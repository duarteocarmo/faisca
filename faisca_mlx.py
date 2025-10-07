# /// script
# requires-python = "==3.12.6"
# dependencies = [
#     "tiktoken",
#     "mlx",
#     "matplotlib",
#     "datasets",
# ]
# ///
import math
import os
import pathlib
import random
import typing as t
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime

import datasets as hf_datasets
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import tiktoken


@contextmanager
def using_training_mode(module: nn.Module, training: bool):
    previous = module.training
    module.train(training)
    try:
        yield
    finally:
        module.train(previous)


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
            self.device = "mlx"
        print("Using MLX with Apple Silicon acceleration")


class CCTitleDataset:
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
        self.eot_token = "<|endoftext|>"

        if max_size is not None:
            sample_limit = min(len(hf_split), max_size * 20)
            hf_split = hf_split.select(range(sample_limit))

        def process_row(row):
            title = (row.get("title") or "").strip()
            if language and row.get("language") != language:
                return {"title": None}
            requested_url = row.get("requested_url") or ""
            if url_filter and url_filter not in requested_url:
                return {"title": None}
            return {"title": title if title else None}

        hf_split = hf_split.map(process_row)

        titles = [t for t in hf_split["title"] if t is not None]

        if max_size is not None and len(titles) > max_size:
            titles = titles[:max_size]

        if shuffle_titles:
            rng = random.Random(seed)
            rng.shuffle(titles)

        eot_id = tokenizer.encode(self.eot_token, allowed_special={self.eot_token})[0]
        ids: list[int] = []
        for doc in titles:
            toks = tokenizer.encode(doc, allowed_special={self.eot_token})
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

    def __getitem__(self, idx: int) -> tuple[mx.array, mx.array]:
        return (
            mx.array(self.inputs[idx], dtype=mx.int32),
            mx.array(self.targets[idx], dtype=mx.int32),
        )


class SimpleDataLoader:
    def __init__(
        self,
        dataset: CCTitleDataset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)
            self.seed += 1

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            inputs = [self.dataset[i][0] for i in batch_indices]
            targets = [self.dataset[i][1] for i in batch_indices]

            yield mx.stack(inputs), mx.stack(targets)


class FeedForward(nn.Module):
    def __init__(self, embedding_dimension: int, hidden_expansion_factor: int = 4):
        super().__init__()
        self.linear1 = nn.Linear(
            embedding_dimension,
            hidden_expansion_factor * embedding_dimension,
        )
        self.linear2 = nn.Linear(
            hidden_expansion_factor * embedding_dimension,
            embedding_dimension,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = nn.gelu(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        num_heads: int,
        qkv_bias: bool,
        dropout_rate: float,
    ):
        super().__init__()

        self.attention = nn.MultiHeadAttention(
            dims=embedding_dimension,
            num_heads=num_heads,
        )
        self.feed_forward = FeedForward(
            embedding_dimension=embedding_dimension,
        )
        self.norm1 = nn.LayerNorm(embedding_dimension)
        self.norm2 = nn.LayerNorm(embedding_dimension)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def __call__(self, x: mx.array) -> mx.array:
        shortcut = x
        x = self.norm1(x)

        seq_len = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        x = self.attention(x, x, x, mask=mask)
        x = self.dropout1(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x = x + shortcut

        return x


class FaiscaGPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.embedding_dimension
        )
        self.positional_embedding = nn.Embedding(
            config.context_length, config.embedding_dimension
        )
        self.dropout_embedding = nn.Dropout(p=config.dropout_rate)
        self.transformer_blocks = [
            TransformerBlock(
                embedding_dimension=config.embedding_dimension,
                num_heads=config.num_heads,
                qkv_bias=config.qkv_bias,
                dropout_rate=config.dropout_rate,
            )
            for _ in range(config.num_layers)
        ]
        self.final_layer_norm = nn.LayerNorm(config.embedding_dimension)
        self.out_head = nn.Linear(
            config.embedding_dimension, config.vocab_size, bias=False
        )

        mx.eval(self.parameters())
        n_params_all = sum(value.size for _, value in tree_flatten(self.parameters()))
        n_params_all_million = n_params_all / 1e6
        print(f"Total number of params: {n_params_all_million:.2f}M")

    def __call__(self, in_idx: mx.array) -> mx.array:
        _, sequence_length = in_idx.shape
        token_embeddings = self.token_embedding(in_idx)
        positional_embeddings = self.positional_embedding(
            mx.arange(sequence_length, dtype=mx.int32)
        )
        positional_embeddings = mx.expand_dims(positional_embeddings, axis=0)
        x = token_embeddings + positional_embeddings
        x = self.dropout_embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_layer_norm(x)
        logits = self.out_head(x)
        return logits


def create_dataloaders(
    train_split,
    val_split,
    config: Config,
    tokenizer: tiktoken.Encoding,
    seed: int = 1337,
) -> tuple[SimpleDataLoader, SimpleDataLoader]:
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
        shuffle_titles=False,
        seed=seed,
        max_size=config.max_test_size,
    )

    train_loader = SimpleDataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=config.dataloader_shuffle,
        drop_last=config.drop_last,
        seed=seed,
    )
    val_loader = SimpleDataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=config.drop_last,
        seed=seed,
    )
    return train_loader, val_loader


def calculate_loss(model: FaiscaGPT, input_batch: mx.array, target_batch: mx.array):
    logits = model(input_batch)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = target_batch.reshape(-1)
    loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="mean")
    return loss


def generate_samples(
    model: FaiscaGPT,
    tokenizer: tiktoken.Encoding,
    config: Config,
) -> None:
    encoded_prompt = tokenizer.encode(
        config.eval_text, allowed_special={config.eot_token}
    )
    encoded = mx.array(encoded_prompt, dtype=mx.int32).reshape(1, -1)

    with using_training_mode(model, False):
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
            instances = decoded.split(config.eot_token)

            print(f"**** GENERATION {gen_num + 1} OF {config.eval_num_samples} ****")
            for instance in instances:
                print(f"> '{instance.strip()}'")
            print("*" * 25)


def generate_single_sample(
    model: FaiscaGPT,
    encoded: mx.array,
    max_new_tokens: int,
    temperature: float,
    context_length: int,
    top_k: int,
    stop_at_eot: int | None = None,
) -> mx.array:
    generated = encoded
    for _ in range(max_new_tokens):
        idx_cond = generated[:, -context_length:]

        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        top_values = mx.topk(logits, top_k)
        thresholds = top_values[:, [-1]]
        logits = mx.where(logits < thresholds, -mx.inf, logits)

        probs = mx.softmax(logits, axis=-1)
        idx_next = mx.random.categorical(probs, num_samples=1)
        idx_next = idx_next.astype(mx.int32)

        generated = mx.concatenate([generated, idx_next], axis=1)

        if stop_at_eot is not None and int(idx_next[0, 0].item()) == stop_at_eot:
            break

    return generated


def train(
    model: FaiscaGPT,
    config: Config,
    train_dataloader: SimpleDataLoader,
    val_dataloader: SimpleDataLoader,
    tokenizer: tiktoken.Encoding,
):
    training_losses: list[float] = []
    validation_losses: list[float] = []
    track_tokens_seen: list[int] = []
    tokens_seen = 0
    global_step = -1

    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    def loss_fn(model: FaiscaGPT, input_batch: mx.array, target_batch: mx.array):
        return calculate_loss(model, input_batch, target_batch)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    for epoch in range(config.num_epochs):
        model.train(True)
        size_train_dataloader = len(train_dataloader)
        batch_num = 0

        for input_batch, target_batch in train_dataloader:
            loss_value, grads = loss_and_grad_fn(model, input_batch, target_batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            tokens_seen += int(input_batch.size)
            global_step += 1
            batch_num += 1

            if global_step % config.eval_freq == 0:
                model.train(False)
                losses: dict[str, float] = {}
                for split, dataloader in [
                    ("train", train_dataloader),
                    ("validation", val_dataloader),
                ]:
                    total_split_loss = 0.0
                    eval_batches = 0
                    for i, (eval_input, eval_target) in enumerate(dataloader):
                        if i < config.eval_iter:
                            loss_eval = calculate_loss(model, eval_input, eval_target)
                            total_split_loss += float(loss_eval.item())
                            eval_batches += 1
                        else:
                            break

                    if eval_batches == 0:
                        losses[split] = 0.0
                    else:
                        losses[split] = total_split_loss / eval_batches

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

                model.train(True)

        generate_samples(
            model=model,
            tokenizer=tokenizer,
            config=config,
        )

    return model, training_losses, validation_losses, track_tokens_seen


def save_plots_and_model(
    config: Config,
    training_losses: list[float],
    validation_losses: list[float],
    track_tokens_seen: list[int],
    model: FaiscaGPT,
):
    fig, ax1 = plt.subplots()

    if training_losses:
        epochs_array = mx.linspace(0, config.num_epochs, len(training_losses)).tolist()
        ax1.plot(epochs_array, training_losses, label="Training loss")
        ax1.plot(
            epochs_array,
            validation_losses,
            linestyle="-.",
            label="Validation loss",
        )
        ax2 = ax1.twiny()
        ax2.plot(track_tokens_seen, training_losses, alpha=0)
        ax2.set_xlabel("Tokens seen")
        ax1.legend(loc="upper right")
    else:
        ax1.plot([], [])
        ax2 = ax1.twiny()
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    fig.tight_layout()

    if not os.path.exists(os.path.dirname(config.chart_path)):
        os.makedirs(os.path.dirname(config.chart_path))
    fig.savefig(config.chart_path)
    print(f"Chart saved to {config.chart_path}")

    if not os.path.exists(os.path.dirname(config.save_path)):
        os.makedirs(os.path.dirname(config.save_path))

    params = {name: value for name, value in tree_flatten(model.parameters())}
    mx.savez(config.save_path, **params)
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
    sequences: mx.array
    action_log_probs: mx.array
    log_probs_ref: mx.array
    returns: mx.array | None
    advantages: mx.array | None
    attention_mask: mx.array | None
    action_mask: mx.array
    kl: mx.array | None = None


def split_experience_batch(experience: Experience) -> list[Experience]:
    batch_size = experience.sequences.shape[0]
    batch_items = []
    keys = (
        "sequences",
        "action_log_probs",
        "log_probs_ref",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
        "kl",
    )
    for i in range(batch_size):
        data = {}
        for key in keys:
            value = getattr(experience, key)
            data[key] = value[i] if value is not None else None
        batch_items.append(Experience(**data))
    return batch_items


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
) -> tuple[mx.array, mx.array, mx.array, list[str]]:
    eot_token_id = tokenizer.encode(
        config["eot_token"], allowed_special={config["eot_token"]}
    )[0]

    prompt_ids = tokenizer.encode(prompt, allowed_special={config["eot_token"]})
    if not prompt_ids:
        prompt_ids = [eot_token_id]

    repeated_ids = [prompt_ids for _ in range(config["num_rollouts"])]
    input_ids = mx.array(repeated_ids, dtype=mx.int32)

    with using_training_mode(model, False):
        sequence_ids = generate_single_sample(
            model=model,
            encoded=input_ids,
            max_new_tokens=config["max_new_tokens"],
            temperature=config["temperature"],
            context_length=config["context_length"],
            top_k=config["top_k"],
        )

    completions = [
        tokenizer.decode(sequence_ids[i].tolist()) for i in range(sequence_ids.shape[0])
    ]

    seq_len = sequence_ids.shape[1]
    positions = mx.arange(seq_len, dtype=mx.int32)
    positions = mx.broadcast_to(positions, sequence_ids.shape)
    prompt_length = len(prompt_ids)
    action_mask = positions >= prompt_length
    action_mask = mx.logical_and(action_mask, mx.not_equal(sequence_ids, eot_token_id))
    action_mask = action_mask[:, 1:]

    returns_values = [calculate_reward_for(text) for text in completions]
    returns = mx.array(returns_values, dtype=mx.float32).reshape(-1, 1)

    return sequence_ids, returns, action_mask, completions


def approx_kl_divergence(
    log_probs: mx.array,
    log_probs_ref: mx.array,
    action_mask: t.Optional[mx.array],
) -> mx.array:
    log_ratio = log_probs_ref.astype(mx.float32) - log_probs.astype(mx.float32)
    if action_mask is not None:
        log_ratio = log_ratio * action_mask.astype(log_ratio.dtype)
    return mx.exp(log_ratio) - log_ratio - 1


def sequences_log_probs(
    model: FaiscaGPT,
    sequence_ids: mx.array,
) -> mx.array:
    with using_training_mode(model, False):
        logits = model(sequence_ids)
    log_probs = nn.log_softmax(logits, axis=-1)
    target_ids = sequence_ids[:, 1:]
    log_probs = log_probs[:, :-1, :]
    gathered = mx.take_along_axis(
        log_probs, mx.expand_dims(target_ids, axis=-1), axis=-1
    )
    return mx.squeeze(gathered, axis=-1)


def zero_pad_sequences(sequences: list[mx.array], side: str = "left") -> mx.array:
    assert side in ("left", "right")
    if not sequences:
        raise ValueError("Expected non-empty sequence list")
    max_len = max(seq.shape[0] for seq in sequences)
    dtype = sequences[0].dtype
    padded_sequences = []
    fill_value = False if dtype == mx.bool_ else 0
    for seq in sequences:
        pad_len = max_len - seq.shape[0]
        pad_width = [(0, 0)] * seq.ndim
        if seq.ndim == 0:
            pad_width = [(pad_len, 0)]
        else:
            pad_width[0] = (pad_len, 0) if side == "left" else (0, pad_len)
        padded = mx.pad(seq, pad_width, constant_values=fill_value)
        padded_sequences.append(padded)
    return mx.stack(padded_sequences, axis=0)


def masked_mean(
    tensor: mx.array,
    mask: t.Optional[mx.array],
    axis: int | None = None,
) -> mx.array:
    if mask is None:
        return mx.mean(tensor, axis=axis)
    mask = mask.astype(tensor.dtype)
    numerator = mx.sum(tensor * mask, axis=axis)
    denominator = mx.sum(mask, axis=axis) + 1e-8
    return numerator / denominator


class GRPOLoss:
    def __init__(self, clip_eps: float, kl_weight: float) -> None:
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight

    def __call__(
        self,
        log_probs: mx.array,
        experience: Experience,
    ) -> tuple[mx.array, mx.array]:
        old_log_probs = experience.action_log_probs
        log_probs_ref = experience.log_probs_ref
        action_mask = experience.action_mask
        advantages = experience.advantages

        kl = approx_kl_divergence(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            action_mask=action_mask,
        )

        ratio = mx.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = mx.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -mx.minimum(surr1, surr2) + self.kl_weight * kl
        loss = masked_mean(loss, action_mask, axis=-1)
        loss = mx.mean(loss)

        return loss, mx.mean(kl)


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
        "kl",
    )
    for key in keys:
        vals = [getattr(item, key) for item in items]
        if all(v is not None for v in vals):
            batch_data[key] = zero_pad_sequences(vals, "left")
        else:
            batch_data[key] = None
    return Experience(**batch_data)


class ExperienceSampler:
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int = 0,
    ):
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.replay_buffer) // self.batch_size
        return math.ceil(len(self.replay_buffer) / self.batch_size)

    def __iter__(self):
        indices = list(range(len(self.replay_buffer)))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)
            self.seed += 1

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            batch_items = [self.replay_buffer[idx] for idx in batch_indices]
            yield join_experience_batch(batch_items)


def clip_gradients(grads: dict, max_norm: float) -> tuple[dict, float]:
    leaves = tree_flatten(grads)
    total = 0.0
    for _, grad in leaves:
        if grad is None:
            continue
        total += float(mx.sum(grad * grad).item())
    total_norm = math.sqrt(total)
    if total_norm > max_norm and total_norm > 0:
        scale = max_norm / (total_norm + 1e-6)
        grads = tree_map(
            lambda g: g * scale if isinstance(g, mx.array) else g,
            grads,
        )
    return grads, total_norm


def grpo(
    config: dict,
    model: FaiscaGPT,
    tokenizer: tiktoken.Encoding,
):
    prompt = "Alerta "
    optimizer = optim.Adam(learning_rate=config["learning_rate"])
    objective = GRPOLoss(clip_eps=config["clip_eps"], kl_weight=config["kl_weight"])

    reference_model = deepcopy(model)
    mx.eval(reference_model.parameters())

    eot_token_id = tokenizer.encode(
        config["eot_token"], allowed_special={config["eot_token"]}
    )[0]

    step_rewards: list[float] = []
    step_kl_divergences: list[float] = []

    def loss_fn(model: FaiscaGPT, exp_batch: Experience):
        log_probs = sequences_log_probs(model=model, sequence_ids=exp_batch.sequences)
        loss, _ = objective(log_probs, exp_batch)
        return loss

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    for k in range(config["training_steps"]):
        replay_buffer = ReplayBuffer()
        rollout_returns: list[mx.array] = []

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

        sample_headlines = random.choices(
            [
                headline
                for rc in rollout_completions
                for headline in rc.split(config["eot_token"])
            ],
            k=min(10, len(rollout_completions)),
        )

        print("=== Sample completions ===")
        for sh in sample_headlines:
            print(f"> {sh.strip()}")
        print("=========================")

        print(f"Returns: {float(mx.sum(returns).item()):.2f}/{returns.shape[0]}")

        returns_mean = mx.mean(returns)
        returns_std = mx.std(returns)
        advantages = (returns - returns_mean) / (returns_std + 1e-8)
        attention_mask = mx.not_equal(sequence_ids, eot_token_id)
        rollout_returns.append(returns)

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
        replay_buffer.append(exp)

        episode_return_sum = float(mx.sum(mx.stack(rollout_returns)).item())
        print(f"Returns of step {k}: {episode_return_sum:.4f}")

        step_rewards.append(episode_return_sum)

        experience_sampler = ExperienceSampler(
            replay_buffer,
            batch_size=config["train_batch_size"],
            shuffle=True,
            drop_last=True,
            seed=k,
        )

        step_kl_values: list[float] = []

        for step_epoch in range(config["epochs_per_step"]):
            for index, exp_batch in enumerate(experience_sampler):
                print(f"Processing experience {index} of {len(experience_sampler)}")
                loss_value, grads = loss_and_grad_fn(model, exp_batch)

                grads, grad_norm = clip_gradients(grads, config["max_norm"])
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)

                log_probs_updated = sequences_log_probs(
                    model=model,
                    sequence_ids=exp_batch.sequences,
                )
                _, kl_value = objective(log_probs_updated, exp_batch)

                print(
                    f"{step_epoch}: kl={float(kl_value.item()): .4f}, "
                    f"grad_norm={grad_norm: .4f}, loss={float(loss_value.item()): .4f}"
                )

                step_kl_values.append(float(kl_value.item()))

        if step_kl_values:
            avg_kl = sum(step_kl_values) / len(step_kl_values)
            step_kl_divergences.append(avg_kl)

    print("\nGenerating training plots...")
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    steps = list(range(len(step_rewards)))
    ax1.plot(steps, step_rewards, "b-", linewidth=2, marker="o", markersize=4)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Total Rewards")
    ax1.set_title("Rewards Over Time")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

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
        ax2.set_title("KL Divergence Over Time")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
    else:
        ax2.plot([], [])
        ax2.set_title("KL Divergence Over Time")

    plt.tight_layout()
    plt.savefig(config["chart_path"], dpi=300, bbox_inches="tight")
    plt.close()

    print("Training completed! Final metrics:")
    print(
        f"  - Final reward: {step_rewards[-1]:.2f}"
        if step_rewards
        else "  - No reward data"
    )
    if step_kl_divergences:
        print(f"  - Final KL divergence: {step_kl_divergences[-1]:.4f}")
    else:
        print("  - No KL divergence data")
    print(f"  - Training plots saved to: {config['chart_path']}")

    return model, step_rewards, step_kl_divergences


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    ds = hf_datasets.load_dataset("duarteocarmo/ccnews-titles-2016")
    ds_train, ds_val = ds["train"], ds["test"]

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mx.random.seed(42)

    if not pathlib.Path("models").exists():
        print("Creating models/ directory")
        pathlib.Path("models").mkdir(parents=True, exist_ok=True)
    if not pathlib.Path("charts").exists():
        print("Creating charts/ directory")
        pathlib.Path("charts").mkdir(parents=True, exist_ok=True)

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
        max_test_size=200,
        max_train_size=1_000,
        train_language="pt",
        url_filter=None,
        num_epochs=1,
        num_workers=0,
        qkv_bias=False,
        save_path=f"models/faisca_{current_time}.npz",
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

    model = FaiscaGPT(
        config=config,
    )
    print("\n========== PRE-TRAINING ==========")

    train_dataloader, val_dataloader = create_dataloaders(
        train_split=ds_train,
        val_split=ds_val,
        config=config,
        tokenizer=tokenizer,
    )

    model, training_losses, validation_losses, track_tokens_seen = train(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
    )

    save_plots_and_model(
        config=config,
        validation_losses=validation_losses,
        training_losses=training_losses,
        track_tokens_seen=track_tokens_seen,
        model=model,
    )

    print("\n========== PRE-TRAINING COMPLETED ==========")

    print("\n========== SUPERVISED FINE-TUNING ==========")

    sft_config = deepcopy(config)
    sft_config.url_filter = ".pt/"
    sft_config.save_path = f"models/faisca_{current_time}_sft.npz"
    sft_config.num_epochs = 1
    sft_config.chart_path = f"charts/faisca_{current_time}_sft.png"
    sft_config.max_train_size = 1000
    sft_config.max_test_size = 200
    sft_config.eval_freq = 5

    sft_train_dataloader, sft_val_dataloader = create_dataloaders(
        train_split=ds_train,
        val_split=ds_val,
        config=sft_config,
        tokenizer=tokenizer,
    )

    (
        sft_model,
        sft_training_losses,
        sft_validation_losses,
        sft_track_tokens_seen,
    ) = train(
        model=model,
        config=sft_config,
        train_dataloader=sft_train_dataloader,
        val_dataloader=sft_val_dataloader,
        tokenizer=tokenizer,
    )

    save_plots_and_model(
        config=sft_config,
        validation_losses=sft_validation_losses,
        training_losses=sft_training_losses,
        track_tokens_seen=sft_track_tokens_seen,
        model=sft_model,
    )

    print("\n========== SUPERVISED FINE-TUNING COMPLETED ==========")

    generate_samples(
        model=sft_model,
        tokenizer=tokenizer,
        config=sft_config,
    )

    print("\n========== REINFORCEMENT LEARNING ==========")

    rl_config = {
        "num_rollouts": 24,
        "max_new_tokens": 80,
        "temperature": 0.8,
        "context_length": 128,
        "top_k": 30,
        "eot_token": "<|endoftext|>",
        "training_steps": 2,
        "train_batch_size": 12,
        "epochs_per_step": 4,
        "learning_rate": 1e-5,
        "clip_eps": 0.2,
        "kl_weight": 0.02,
        "max_norm": 1.0,
        "chart_path": f"charts/faisca_{current_time}_rl.png",
    }

    rl_model, _, _ = grpo(
        config=rl_config,
        model=sft_model,
        tokenizer=tokenizer,
    )

    print("\n========== REINFORCEMENT LEARNING COMPLETED ==========")
