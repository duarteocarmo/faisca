import os
import typing as t
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import tiktoken
import torch
from torch import nn
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
        vocab_size: int,
        embedding_dimension: int,
        context_length: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float,
        qkv_bias: bool,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.positional_embedding = nn.Embedding(context_length, embedding_dimension)
        self.dropout_embedding = nn.Dropout(p=dropout_rate)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_dimension=embedding_dimension,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(embedding_dimension)
        self.out_head = nn.Linear(embedding_dimension, vocab_size, bias=False)

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
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    max_length: int,
    stride: int,
    language: str | None = None,
    seed: int = 1337,
    max_train_size: int | None = None,
    max_test_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_ds = CCTitleDataset(
        hf_split=train_split,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        language=language,
        shuffle_titles=True,
        seed=seed,
        max_size=max_train_size,
    )
    val_ds = CCTitleDataset(
        hf_split=val_split,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        language=language,
        shuffle_titles=False,  # keep this deterministic
        seed=seed,
        max_size=max_test_size,
    )

    common = dict(
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(train_ds, shuffle=shuffle, **common)
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
    max_new_tokens: int,
    num_samples: int,
    prompt: str,
    temperature: float,
    top_k: int,
    device: str,
    eot_token: str,
    context_length: int,
) -> None:
    model.eval()

    with torch.no_grad():
        encoded = (
            torch.tensor(
                tokenizer.encode(prompt, allowed_special={eot_token}),
                dtype=torch.long,
            )
            .unsqueeze(0)
            .to(device)
        )

        for gen_num in range(num_samples):
            encoded_completion = generate_single_sample(
                model=model,
                encoded=encoded,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                context_length=context_length,
            )

            decoded = tokenizer.decode(encoded_completion[0].tolist())
            decoded = decoded.replace("\n", " ").strip()
            instances = decoded.split(eot_token)  # split at eot token

            print(f"**** GENERATION {gen_num + 1} OF {num_samples} ****")
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
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    context_length: int,
    device: str,
    eval_freq: int,
    eval_iter: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    tokenizer: tiktoken.Encoding,
    eot_token: str,
    eval_text: str,
    eval_num_samples: int,
    eval_temperature: float,
    eval_top_k: int,
    eval_max_new_tokens: int,
):
    training_losses, validation_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        size_train_dataloader = len(train_dataloader)
        batch_num = 0

        for input_batch, target_batch in train_dataloader:
            optimizer.zero_grad()
            loss = calculate_loss(
                input_batch=input_batch,
                target_batch=target_batch,
                model=model,
                device=device,
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            batch_num += 1

            # run evaluation
            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    losses = dict()
                    for split, dataloader in [
                        ("train", train_dataloader),
                        ("validation", val_dataloader),
                    ]:
                        total_split_loss = 0
                        for i, (input_batch, target_batch) in enumerate(dataloader):
                            if i < eval_iter:
                                loss = calculate_loss(
                                    input_batch=input_batch,
                                    target_batch=target_batch,
                                    model=model,
                                    device=device,
                                )
                                total_split_loss += loss.item()
                            else:
                                break

                        losses[split] = total_split_loss / eval_iter

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
            max_new_tokens=eval_max_new_tokens,
            num_samples=eval_num_samples,
            prompt=eval_text,
            temperature=eval_temperature,
            top_k=eval_top_k,
            device=device,
            eot_token=eot_token,
            context_length=context_length,
        )

        model.train()

    return model, training_losses, validation_losses, track_tokens_seen


def save_plots_and_model(
    num_epochs: int,
    chart_path: str,
    save_path: str,
    training_losses: list[float],
    validation_losses: list[float],
    track_tokens_seen: list[float],
    model: FaiscaGPT,
):
    fig, ax1 = plt.subplots()

    epochs_tensor = torch.linspace(0, num_epochs, len(training_losses))
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

    if not os.path.exists(os.path.dirname(chart_path)):
        os.makedirs(os.path.dirname(chart_path))
    fig.savefig(chart_path)
    print(f"Chart saved to {chart_path}")

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


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
        max_test_size=5000,
        max_train_size=25000,
        train_language="pt",
        num_epochs=15,
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
        eval_num_samples=3,
        eval_temperature=1.0,
        eval_top_k=30,
        eval_max_new_tokens=60,
    )

    train_dataloader, val_dataloader = create_dataloaders(
        train_split=ds_train,
        val_split=ds_val,
        language=config.train_language,
        batch_size=config.batch_size,
        drop_last=config.drop_last,
        shuffle=config.dataloader_shuffle,
        num_workers=config.num_workers,
        max_length=config.max_length,
        stride=config.stride,
        max_train_size=config.max_train_size,
        max_test_size=config.max_test_size,
    )

    model = FaiscaGPT(
        vocab_size=config.vocab_size,
        embedding_dimension=config.embedding_dimension,
        context_length=config.context_length,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout_rate=config.dropout_rate,
        qkv_bias=config.qkv_bias,
    )

    model, training_losses, validation_losses, track_tokens_seen = train(
        model=model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_epochs=config.num_epochs,
        context_length=config.context_length,
        device=config.device,
        eval_freq=config.eval_freq,
        eval_iter=config.eval_iter,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
        eot_token=config.eot_token,
        eval_max_new_tokens=config.eval_max_new_tokens,
        eval_text=config.eval_text,
        eval_num_samples=config.eval_num_samples,
        eval_temperature=config.eval_temperature,
        eval_top_k=config.eval_top_k,
    )

    save_plots_and_model(
        num_epochs=config.num_epochs,
        chart_path=config.chart_path,
        save_path=config.save_path,
        validation_losses=validation_losses,
        training_losses=training_losses,
        track_tokens_seen=track_tokens_seen,
        model=model,
    )
