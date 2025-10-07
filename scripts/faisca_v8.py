import os
import typing as t
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Optional, Self

import matplotlib.pyplot as plt
import random
import tiktoken
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
import datasets as hf_datasets


@dataclass
class Config:
    batch_size: int
    chart_path: str
    context_length: int
    dataloader_shuffle: bool
    device: str
    drop_last: bool
    dropout_rate: float
    embedding_dimension: int
    eot_token: str
    eval_freq: int
    eval_iter: int
    eval_num_samples: int
    eval_temperature: float
    eval_max_new_tokens: int
    eval_text: str
    eval_top_k: int
    learning_rate: float
    max_length: int
    train_language: str
    max_new_tokens: int
    max_test_size: int
    max_train_size: int
    num_epochs: int
    num_heads: int
    num_layers: int
    num_workers: int
    qkv_bias: bool
    save_path: str
    stride: int
    vocab_size: int
    weight_decay: float
    url_filter: str | None = None

    def __post_init__(self):
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            print(f"Using device: {self.device}")


class CCTitleDataset(Dataset):
    def __init__(
        self,
        hf_split,
        tokenizer: t.Any,
        max_length: int,
        stride: int,
        language: str | None = None,
        url_filter: str | None = None,
        shuffle_titles: bool = True,
        seed: int = 1337,
        max_size: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.seed = seed
        self.vocab_size = tokenizer.n_vocab
        self.EOT_TOKEN = "<|endoftext|>"

        def process_row(row):
            title = (row.get("title") or "").strip()
            if language and row.get("language") != language:
                return {"title": None}
            if url_filter and url_filter not in row.get("requested_url"):
                return {"title": None}
            return {"title": title if title else None}

        hf_split = hf_split.map(process_row)

        titles = [t for t in hf_split["title"] if t is not None]

        # Apply size limit if specified
        if max_size is not None and len(titles) > max_size:
            titles = titles[:max_size]

        if shuffle_titles:
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(len(titles), generator=g).tolist()
            titles = [titles[i] for i in perm]

        eot_id = tokenizer.encode(self.EOT_TOKEN, allowed_special={self.EOT_TOKEN})[0]
        ids: list[int] = []
        for doc in titles:
            toks = tokenizer.encode(doc, allowed_special={self.EOT_TOKEN})
            if toks:
                ids.extend(toks)
                ids.append(eot_id)

        self.inputs: list[list[int]] = []
        self.targets: list[list[int]] = []
        limit = max(0, len(ids) - max_length - 1)
        for i in range(0, limit, stride):
            x = ids[i : i + max_length]
            y = ids[i + 1 : i + 1 + max_length]
            if len(x) == max_length and len(y) == max_length:
                self.inputs.append(x)
                self.targets.append(y)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])


class FeedForward(nn.Module):
    def __init__(self, embedding_dimension: int, hidden_expansion_factor: int = 4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(
                embedding_dimension,
                hidden_expansion_factor * embedding_dimension,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_expansion_factor * embedding_dimension,
                embedding_dimension,
            ),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        num_heads: int,
        qkv_bias: bool,
        dropout_rate: float,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dimension,
            num_heads=num_heads,
            dropout=dropout_rate,
            bias=qkv_bias,
            batch_first=True,
        )
        self.feed_forward = FeedForward(
            embedding_dimension=embedding_dimension,
        )
        self.norm1 = nn.LayerNorm(embedding_dimension)
        self.norm2 = nn.LayerNorm(embedding_dimension)
        self.drop_shortcut = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)

        L = x.size(1)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
        x, _ = self.attention(x, x, x, attn_mask=causal)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # shortcut connection // feed forward
        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class FaiscaGPT(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.embedding_dimension
        )
        self.positional_embedding = nn.Embedding(
            config.context_length, config.embedding_dimension
        )
        self.dropout_embedding = nn.Dropout(p=config.dropout_rate)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_dimension=config.embedding_dimension,
                    num_heads=config.num_heads,
                    qkv_bias=config.qkv_bias,
                    dropout_rate=config.dropout_rate,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(config.embedding_dimension)
        self.out_head = nn.Linear(
            config.embedding_dimension, config.vocab_size, bias=False
        )

        n_params_all = sum(p.numel() for p in self.parameters())
        n_params_all_million = n_params_all / 1e6
        print(f"Total number of params: {n_params_all_million:.2f}M")

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        _, sequence_length = in_idx.shape
        token_embeddings = self.token_embedding(in_idx)
        positional_embeddings = self.positional_embedding(
            torch.arange(sequence_length, device=in_idx.device)
        )
        x = token_embeddings + positional_embeddings
        x = self.dropout_embedding(x)
        x = self.transformer_blocks(x)
        x = self.final_layer_norm(x)
        logits = self.out_head(x)
        return logits


def create_dataloaders(
    train_split,
    val_split,
    config: Config,
    seed: int = 1337,
) -> tuple[DataLoader, DataLoader]:
    train_ds = CCTitleDataset(
        hf_split=train_split,
        tokenizer=tokenizer,
        max_length=config.max_length,
        stride=config.stride,
        language=config.train_language,
        url_filter=config.url_filter,
        shuffle_titles=True,
        seed=seed,
        max_size=config.max_train_size,
    )
    val_ds = CCTitleDataset(
        hf_split=val_split,
        tokenizer=tokenizer,
        max_length=config.max_length,
        stride=config.stride,
        language=config.train_language,
        url_filter=config.url_filter,
        shuffle_titles=False,  # keep this deterministic
        seed=seed,
        max_size=config.max_test_size,
    )

    common = dict(
        batch_size=config.batch_size,
        drop_last=config.drop_last,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(train_ds, shuffle=config.dataloader_shuffle, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader


def calculate_loss(input_batch, target_batch, model, device):
    input_batch, target_batch = (
        input_batch.to(device),
        target_batch.to(device),
    )
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def generate_samples(
    model: FaiscaGPT,
    tokenizer: tiktoken.Encoding,
    config: Config,
) -> None:
    model.eval()

    with torch.no_grad():
        encoded = (
            torch.tensor(
                tokenizer.encode(config.eval_text, allowed_special={config.eot_token}),
                dtype=torch.long,
            )
            .unsqueeze(0)
            .to(config.device)
        )

        for gen_num in range(config.eval_num_samples):
            encoded_completion = generate_single_sample(
                model=model,
                encoded=encoded,
                max_new_tokens=config.eval_max_new_tokens,
                temperature=config.eval_temperature,
                top_k=config.eval_top_k,
                context_length=config.context_length,
            )

            decoded = tokenizer.decode(encoded_completion[0].tolist())
            decoded = decoded.replace("\n", " ").strip()
            instances = decoded.split(config.eot_token)  # split at eot token

            print(f"**** GENERATION {gen_num + 1} OF {config.eval_num_samples} ****")
            for instance in instances:
                print(f"> '{instance}'")
            print("*" * 25)


def generate_single_sample(
    model: FaiscaGPT,
    encoded: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    context_length: int,
    top_k: int,
    stop_at_eot: int | None = None,
) -> torch.Tensor:
    for _ in range(max_new_tokens):
        idx_cond = encoded[:, -context_length:]

        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        encoded = torch.cat((encoded, idx_next), dim=1)
        if stop_at_eot is not None and idx_next.item() == stop_at_eot:
            break

    return encoded


def train(
    model: FaiscaGPT,
    config: Config,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    tokenizer: tiktoken.Encoding,
):
    training_losses, validation_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    model.to(config.device)

    for epoch in range(config.num_epochs):
        model.train()
        size_train_dataloader = len(train_dataloader)
        batch_num = 0

        for input_batch, target_batch in train_dataloader:
            optimizer.zero_grad()
            loss = calculate_loss(
                input_batch=input_batch,
                target_batch=target_batch,
                model=model,
                device=config.device,
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            batch_num += 1

            # run evaluation
            if global_step % config.eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    losses = dict()
                    for split, dataloader in [
                        ("train", train_dataloader),
                        ("validation", val_dataloader),
                    ]:
                        total_split_loss = 0
                        for i, (input_batch, target_batch) in enumerate(dataloader):
                            if i < config.eval_iter:
                                loss = calculate_loss(
                                    input_batch=input_batch,
                                    target_batch=target_batch,
                                    model=model,
                                    device=config.device,
                                )
                                total_split_loss += loss.item()
                            else:
                                break

                        losses[split] = total_split_loss / config.eval_iter

                    train_loss = losses["train"]
                    validation_loss = losses["validation"]

                    print(
                        (
                            f"Epoch: {epoch} - Step: {global_step} - "
                            f"Train Loss: {train_loss:.4f} - "
                            f"Validation Loss: {validation_loss:.4f} - "
                            f"Batch: {batch_num} / {size_train_dataloader} - "
                            f"Tokens seen: {tokens_seen}"
                        )
                    )

                    training_losses.append(train_loss)
                    validation_losses.append(validation_loss)
                    track_tokens_seen.append(tokens_seen)

                    model.train()

        generate_samples(
            model=model,
            tokenizer=tokenizer,
            config=config,
        )

        model.train()

    return model, training_losses, validation_losses, track_tokens_seen


def save_plots_and_model(
    config: Config,
    training_losses: list[float],
    validation_losses: list[float],
    track_tokens_seen: list[float],
    model: FaiscaGPT,
):
    fig, ax1 = plt.subplots()

    epochs_tensor = torch.linspace(0, config.num_epochs, len(training_losses))
    # Plot training and validation loss against epochs
    ax1.plot(epochs_tensor, training_losses, label="Training loss")
    ax1.plot(
        epochs_tensor,
        validation_losses,
        linestyle="-.",
        label="Validation loss",
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(
        track_tokens_seen, training_losses, alpha=0
    )  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room

    if not os.path.exists(os.path.dirname(config.chart_path)):
        os.makedirs(os.path.dirname(config.chart_path))
    fig.savefig(config.chart_path)
    print(f"Chart saved to {config.chart_path}")

    if not os.path.exists(os.path.dirname(config.save_path)):
        os.makedirs(os.path.dirname(config.save_path))
    torch.save(model.state_dict(), config.save_path)
    print(f"Model saved to {config.save_path}")


def calculate_reward_for(text: str) -> float:
    target_words = [
        "futebol",
        "benfica",
        "porto",
        "sporting",
        "bola",
        "liga",
        "campeão",
        "taça",
        "golo",
        "jogo",
        "jogador",
        "treinador",
        "fifa",
        "uefa",
        "euro",
        "messi",
        "ronaldo",
    ]
    has_word = set(text.lower().split()).intersection(set(target_words))
    return 1.0 if has_word else 0.0


@dataclass
class Experience:
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    log_probs_ref: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]
    action_mask: torch.Tensor
    kl: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> Self:
        members = {}
        for field in fields(self):
            v = getattr(self, field.name)
            if isinstance(v, torch.Tensor):
                v = v.to(device=device)
            members[field.name] = v
        return Experience(**members)


def split_experience_batch(experience: Experience) -> list[Experience]:
    batch_size = experience.sequences.size(0)
    batch_data = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        "action_log_probs",
        "log_probs_ref",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        value = getattr(experience, key)
        if value is None:
            vals = [None] * batch_size
        else:
            vals = torch.unbind(value)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_data[i][key] = v

    return [Experience(**data) for data in batch_data]


class ReplayBuffer:
    def __init__(self, limit: int = 0) -> None:
        self.limit = limit
        self.items: list[Experience] = []

    def append(self, experience: Experience) -> None:
        items = split_experience_batch(experience)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Experience:
        return self.items[idx]


def rollout(
    prompt: str,
    model: FaiscaGPT,
    tokenizer: tiktoken.Encoding,
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    # Get token IDs
    eot_token_id = tokenizer.encode(
        config["eot_token"], allowed_special={config["eot_token"]}
    )[0]

    # Encode prompt and create tensors
    input_ids = tokenizer.encode(prompt, allowed_special={config["eot_token"]})
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    # Repeat for multiple rollouts
    attention_mask = attention_mask.repeat(config["num_rollouts"], 1)
    input_ids = input_ids.repeat(config["num_rollouts"], 1)

    input_ids = input_ids.to(config["device"])
    attention_mask = attention_mask.to(config["device"])

    # Generate sequences
    model.eval()
    sequence_ids = generate_single_sample(
        model=model,
        encoded=input_ids,
        max_new_tokens=config["max_new_tokens"],
        temperature=config["temperature"],
        context_length=config["context_length"],
        top_k=config["top_k"],
    )

    # Decode all completions
    completions = []
    for i in range(sequence_ids.size(0)):
        completion = tokenizer.decode(sequence_ids[i].tolist())
        completions.append(completion)

    # Create action mask
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_ids.shape[1] :] = True
    action_mask[sequence_ids == eot_token_id] = False
    action_mask = action_mask[:, 1:]

    # Calculate rewards
    returns = torch.zeros(config["num_rollouts"], 1, dtype=torch.float)
    for i, completion in enumerate(completions):
        reward = calculate_reward_for(completion)
        returns[i] = reward

    return sequence_ids, returns, action_mask, completions


def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: t.Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Monte-Carlo approximation of KL divergence, k3 estimator, see: http://joschu.net/blog/kl-approx.html
    """

    log_ratio = log_probs_ref.float() - log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    return log_ratio.exp() - log_ratio - 1


def sequences_log_probs(
    model: FaiscaGPT,
    sequence_ids: torch.Tensor,
) -> torch.Tensor:
    # Get logits from model
    logits = model(sequence_ids)  # (batch_size, seq_len, vocab_size)

    # Calculate log probabilities
    log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)

    # Get log probabilities for the actual tokens
    # We shift by 1 because we want log prob of next token
    target_ids = sequence_ids[:, 1:]  # (batch_size, seq_len-1)
    log_probs = log_probs[:, :-1, :]  # (batch_size, seq_len-1, vocab_size)

    # Gather log probabilities for the target tokens
    log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(
        -1
    )  # (batch_size, seq_len-1)

    return log_probs


def zero_pad_sequences(
    sequences: list[torch.Tensor], side: str = "left"
) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


class GRPOLoss(nn.Module):
    """GRPO actor loss"""

    def __init__(self, clip_eps: float, kl_weight: float) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight

    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        old_log_probs = experience.action_log_probs
        log_probs_ref = experience.log_probs_ref
        action_mask = experience.action_mask
        advantages = experience.advantages

        kl = approx_kl_divergence(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            action_mask=action_mask,
        )

        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2) + self.kl_weight * kl
        loss = masked_mean(loss, action_mask, dim=-1).mean()

        return loss, kl.mean()


def join_experience_batch(items: list[Experience]) -> Experience:
    batch_data = {}
    keys = (
        "sequences",
        "action_log_probs",
        "log_probs_ref",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        vals = [getattr(item, key) for item in items]
        if all(v is not None for v in vals):
            data = zero_pad_sequences(vals, "left")
        else:
            data = None
        batch_data[key] = data
    return Experience(**batch_data)


def grpo(
    config: dict,
    model: FaiscaGPT,
    tokenizer: tiktoken.Encoding,
):
    prompt = "Alerta "
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    objective = GRPOLoss(clip_eps=config["clip_eps"], kl_weight=config["kl_weight"])
    cpu_device = torch.device("cpu")

    reference_model = deepcopy(model)

    model.to(config["device"])
    reference_model.to(config["device"])

    eot_token_id = tokenizer.encode(
        config["eot_token"], allowed_special={config["eot_token"]}
    )[0]

    step_rewards = []
    step_kl_divergences = []

    for k in range(config["training_steps"]):
        replay_buffer = ReplayBuffer()
        rollout_returns = []

        with torch.no_grad():
            (
                sequence_ids,
                returns,
                action_mask,
                rollout_completions,
            ) = rollout(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                config=config,
            )

            print(f"Step {k} generated {len(rollout_completions)} completions")
            print(
                "Sample completions: ",
                "\n - ".join(random.choices(rollout_completions, k=10)),
            )
            print(f"Returns: {returns.sum().item():.2f}/{len(returns)}")

            advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
            attention_mask = sequence_ids != eot_token_id
            rollout_returns.append(returns.cpu())

            # TODO: Check if this needs
            # attention mask or if we are ok
            log_probs = sequences_log_probs(
                model=model,
                sequence_ids=sequence_ids,
            )

            log_probs_reference = sequences_log_probs(
                model=reference_model,
                sequence_ids=sequence_ids,
            )

            kl = approx_kl_divergence(
                log_probs=log_probs,
                log_probs_ref=log_probs_reference,
                action_mask=action_mask,
            )

            exp = Experience(
                sequences=sequence_ids,
                action_log_probs=log_probs,
                log_probs_ref=log_probs_reference,
                returns=returns,
                advantages=advantages,
                attention_mask=attention_mask,
                action_mask=action_mask,
                kl=kl,
            )
            replay_buffer.append(exp.to(cpu_device))

        torch.mps.empty_cache()
        episode_return_sum = torch.stack(rollout_returns).sum()
        print(f"Returns of step {k}: {episode_return_sum:.4f}")

        step_rewards.append(episode_return_sum.item())

        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=config["train_batch_size"],
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )

        step_kl_values = []

        for step_epoch in range(config["epochs_per_step"]):
            model.train()

            for index, exp in enumerate(experience_sampler):
                print(f"Processing experience {index} of {len(experience_sampler)}")
                exp: Experience
                optimizer.zero_grad()
                exp = exp.to(torch.device(config["device"]))

                log_probs = sequences_log_probs(
                    model=model,
                    sequence_ids=exp.sequences,
                )

                loss, kl = objective(log_probs=log_probs, experience=exp)

                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    continue

                loss.backward()
                grad_norm = clip_grad_norm_(
                    model.parameters(), max_norm=config["max_norm"]
                )
                print(f"{step_epoch}: kl={kl: .4f}, grad_norm={grad_norm: .4f}")

                optimizer.step()
                step_kl_values.append(kl.item())

        if step_kl_values:
            avg_kl = sum(step_kl_values) / len(step_kl_values)
            step_kl_divergences.append(avg_kl)

    print("\nGenerating training plots...")
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot rewards
    steps = list(range(len(step_rewards)))
    ax1.plot(steps, step_rewards, "b-", linewidth=2, marker="o", markersize=4)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Total Rewards")
    ax1.set_title("Sentiment Training: Rewards Over Time")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Plot KL divergence
    if step_kl_divergences:
        kl_steps = list(range(len(step_kl_divergences)))
        ax2.plot(
            kl_steps,
            step_kl_divergences,
            "r-",
            linewidth=2,
            marker="s",
            markersize=4,
        )
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("Average KL Divergence")
        ax2.set_title("Sentiment Training: KL Divergence Over Time")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(config["chart_path"], dpi=300, bbox_inches="tight")
    plt.close()

    print("Training completed! Final metrics:")
    print(f"  - Final reward: {step_rewards[-1]:.2f}")
    print(
        f"  - Final KL divergence: {step_kl_divergences[-1]:.4f}"
        if step_kl_divergences
        else "  - No KL divergence data"
    )
    print("  - Training plots saved to: training_metrics.png")

    return model, step_rewards, step_kl_divergences


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    ds = hf_datasets.load_dataset("duarteocarmo/ccnews-titles-2016")
    ds_train, ds_val = ds["train"], ds["test"]

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    torch.manual_seed(42)

    config = Config(
        batch_size=64,
        chart_path=f"charts/faisca_{current_time}.png",
        context_length=128,
        device="auto",
        drop_last=True,
        dropout_rate=0.1,
        eval_freq=20,
        eval_iter=5,
        eval_text="Presidente",
        learning_rate=3e-4,
        max_length=256,
        max_test_size=6000,
        max_train_size=30_000,
        train_language="pt",
        url_filter=None,
        num_epochs=10,
        num_workers=0,
        qkv_bias=False,
        save_path=f"models/faisca_{current_time}.pt",
        stride=64,
        vocab_size=tokenizer.n_vocab,
        weight_decay=0.1,
        dataloader_shuffle=True,
        max_new_tokens=250,
        embedding_dimension=128,
        num_heads=4,
        num_layers=4,
        eot_token="<|endoftext|>",
        eval_num_samples=1,
        eval_temperature=0.5,
        eval_top_k=30,
        eval_max_new_tokens=60,
    )
    #
    # model = FaiscaGPT(
    #     config=config,
    # )
    # print("\n========== PRE-TRAINING ==========")
    #
    # train_dataloader, val_dataloader = create_dataloaders(
    #     train_split=ds_train,
    #     val_split=ds_val,
    #     config=config,
    # )
    #
    # model, training_losses, validation_losses, track_tokens_seen = train(
    #     model=model,
    #     config=config,
    #     train_dataloader=train_dataloader,
    #     val_dataloader=val_dataloader,
    #     tokenizer=tokenizer,
    # )
    #
    # save_plots_and_model(
    #     config=config,
    #     validation_losses=validation_losses,
    #     training_losses=training_losses,
    #     track_tokens_seen=track_tokens_seen,
    #     model=model,
    # )
    #
    # print("\n========== PRE-TRAINING COMPLETED ==========")
    #
    # print("\n========== SUPERVISED FINE-TUNING ==========")
    #
    sft_config = deepcopy(config)
    sft_config.url_filter = ".pt/"
    sft_config.save_path = f"models/faisca_{current_time}_sft.pt"
    sft_config.num_epochs = 5
    sft_config.chart_path = f"charts/faisca_{current_time}_sft.png"
    sft_config.max_train_size = 20_000
    sft_config.max_test_size = 4000
    sft_config.eval_freq = 5

    # sft_train_dataloader, sft_val_dataloader = create_dataloaders(
    #     train_split=ds_train,
    #     val_split=ds_val,
    #     config=sft_config,
    # )
    #
    # (
    #     sft_model,
    #     sft_training_losses,
    #     sft_validation_losses,
    #     sft_track_tokens_seen,
    # ) = train(
    #     model=model,
    #     config=sft_config,
    #     train_dataloader=sft_train_dataloader,
    #     val_dataloader=sft_val_dataloader,
    #     tokenizer=tokenizer,
    # )
    #
    # save_plots_and_model(
    #     config=sft_config,
    #     validation_losses=sft_validation_losses,
    #     training_losses=sft_training_losses,
    #     track_tokens_seen=sft_track_tokens_seen,
    #     model=sft_model,
    # )
    #
    # print("\n========== SUPERVISED FINE-TUNING COMPLETED ==========")
    #
    # generate_samples(
    #     model=sft_model,
    #     tokenizer=tokenizer,
    #     config=sft_config,
    # )

    print("\n========== REINFORCEMENT LEARNING ==========")

    # load model from file
    sft_model = FaiscaGPT(
        config=sft_config,
    )
    sft_model.load_state_dict(
        torch.load("models/faisca_2025-10-04_22-02-58_sft.pt", map_location="cpu")
    )

    rl_config = {
        "num_rollouts": 24,
        "max_new_tokens": 80,
        "temperature": 0.8,
        "context_length": 128,
        "top_k": 30,
        "eot_token": "<|endoftext|>",
        "device": "mps",
        "training_steps": 18,
        "train_batch_size": 12,
        "epochs_per_step": 4,
        "learning_rate": 1e-5,
        "clip_eps": 0.2,
        "kl_weight": 0.02,
        "max_norm": 1.0,
        "chart_path": f"charts/faisca_{current_time}_rl.png",
    }

    rl_model = grpo(
        config=rl_config,
        model=sft_model,
        tokenizer=tokenizer,
    )

    print("\n========== REINFORCEMENT LEARNING COMPLETED ==========")
