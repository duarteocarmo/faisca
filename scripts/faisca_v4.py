import os
import typing as t
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import tiktoken
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class Config:
    batch_size: int
    chart_path: str
    context_length: int
    device: str
    drop_last: bool
    dropout_rate: float
    embedding_dimension: int
    eval_freq: int
    eval_iter: int
    eval_text: str
    max_new_tokens: int
    learning_rate: float
    max_length: int
    num_epochs: int
    num_heads: int
    num_layers: int
    num_workers: int
    qkv_bias: bool
    save_path: str
    dataloader_shuffle: bool
    stride: int
    val_split: float
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


class FaiscaDataset(Dataset):
    def __init__(self, data: str, max_length: int, tokenizer: t.Any, stride: int):
        self.tokenizer = tokenizer
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)

        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(data, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_ids = token_ids[i : i + max_length]
            target_ids = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(input_ids)
            self.target_ids.append(target_ids)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx]), torch.tensor(self.target_ids[idx])

    def encode(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer.encode(text)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        return encoded_tensor

    def decode(self, token_ids: torch.Tensor) -> str:
        flat = token_ids.squeeze(0)
        return self.tokenizer.decode(flat.tolist())


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

        seq_length = x.size(1)
        attn_mask = torch.triu(
            torch.ones(seq_length, seq_length, dtype=torch.bool, device=x.device),
            1,
        )
        x, _ = self.attention(x, x, x, attn_mask=attn_mask)
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
    train_data: str,
    val_data: str,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    max_length: int,
    stride: int,
) -> tuple[DataLoader, DataLoader]:
    base_args = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    dataset_args = dict(
        max_length=max_length,
        tokenizer=tokenizer,
        stride=stride,
    )

    train_dataset_args = dataset_args.copy()
    train_dataset_args["data"] = train_data

    train_dataloader = DataLoader(FaiscaDataset(**train_dataset_args), **base_args)

    val_dataset_args = dataset_args.copy()
    val_dataset_args["data"] = val_data

    val_dataloader = DataLoader(FaiscaDataset(**val_dataset_args), **base_args)

    return train_dataloader, val_dataloader


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


def train(
    model: FaiscaGPT,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    context_length: int,
    device: str,
    eval_freq: int,
    eval_iter: int,
    max_new_tokens: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
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

        model.eval()
        encoded = train_dataloader.dataset.encode(config.eval_text).to(device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = encoded[:, -context_length:]
                logits = model(idx_cond)
                logits = logits[:, -1, :]
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                encoded = torch.cat((encoded, idx_next), dim=1)

        encoded_pretty = encoded.squeeze(0).tolist()
        print(f"\nGenerated: {encoded_pretty}")
        decoded = train_dataloader.dataset.decode(encoded[0])
        decoded = decoded.replace("\n", " ")
        print(f"Decoded: {decoded}\n")

        model.train()

    return model, training_losses, validation_losses, track_tokens_seen


def save_plots_and_model(
    num_epochs: int,
    chart_path: str,
    save_path: str,
    training_losses: list[float],
    validation_losses: list[float],
    track_tokens_seen: list[float],
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
    file_path = "datasets/os-maias-mini.txt"
    bocage_text = Path(file_path).read_text()
    tokenizer = tiktoken.get_encoding("gpt2")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    torch.manual_seed(42)

    config = Config(
        batch_size=2,
        chart_path=f"charts/faisca_{current_time}.png",
        context_length=256,
        device="auto",
        drop_last=True,
        dropout_rate=0.1,
        eval_freq=5,
        eval_iter=1,
        eval_text="Quando pensares",
        learning_rate=5e-4,
        max_length=256,
        num_epochs=10,
        num_workers=0,
        qkv_bias=False,
        save_path=f"models/faisca_{current_time}.pt",
        stride=128,
        val_split=0.1,
        vocab_size=50257,
        weight_decay=0.1,
        dataloader_shuffle=True,
        max_new_tokens=250,
        # GPT configurations
        embedding_dimension=768,
        num_heads=12,
        num_layers=12,
    )

    split_idx = int(len(bocage_text) * (1 - config.val_split))
    train_data = bocage_text[:split_idx]
    val_data = bocage_text[split_idx:]

    train_dataloader, val_dataloader = create_dataloaders(
        train_data=train_data,
        val_data=val_data,
        batch_size=config.batch_size,
        drop_last=config.drop_last,
        shuffle=config.dataloader_shuffle,
        num_workers=config.num_workers,
        max_length=config.max_length,
        stride=config.stride,
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
        max_new_tokens=config.max_new_tokens,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    save_plots_and_model(
        num_epochs=config.num_epochs,
        chart_path=config.chart_path,
        save_path=config.save_path,
        validation_losses=validation_losses,
        training_losses=training_losses,
        track_tokens_seen=track_tokens_seen,
    )
