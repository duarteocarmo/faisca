# /// script
# requires-python = "==3.12.6"
# dependencies = [
#     "tiktoken",
#     "mlx",
#     "matplotlib",
#     "datasets",
# ]
# ///
import os
import typing as t
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import tiktoken
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

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
        # MLX automatically uses the best available device (Apple Silicon GPU)
        print("Using MLX with Apple Silicon acceleration")


class CCTitleDataset:
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
            mx.random.seed(seed)
            perm = mx.random.permutation(len(titles)).tolist()
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

    def __getitem__(self, idx: int) -> tuple[mx.array, mx.array]:
        return mx.array(self.inputs[idx]), mx.array(self.targets[idx])


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

    def __call__(self, x):
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

        # Create causal mask
        L = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        x = self.attention(x, x, x, mask=mask)
        x = self.dropout1(x)
        x = x + shortcut

        # shortcut connection // feed forward
        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout2(x)
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

        self.transformer_blocks = [
            TransformerBlock(
                embedding_dimension=embedding_dimension,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_layers)
        ]

        self.final_layer_norm = nn.LayerNorm(embedding_dimension)
        self.out_head = nn.Linear(embedding_dimension, vocab_size, bias=False)

        # Count parameters after initialization
        mx.eval(self.parameters())
        n_params_all = sum(v.size for _, v in tree_flatten(self.parameters()))
        n_params_all_million = n_params_all / 1e6
        print(f"Total number of params: {n_params_all_million:.2f}M")

    def __call__(self, in_idx: mx.array) -> mx.array:
        _, sequence_length = in_idx.shape
        token_embeddings = self.token_embedding(in_idx)
        positional_embeddings = self.positional_embedding(mx.arange(sequence_length))
        x = token_embeddings + positional_embeddings
        x = self.dropout_embedding(x)

        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_layer_norm(x)
        logits = self.out_head(x)
        return logits


def create_datasets(
    train_split,
    val_split,
    max_length: int,
    stride: int,
    language: str | None = None,
    seed: int = 1337,
    max_train_size: int | None = None,
    max_test_size: int | None = None,
) -> tuple[CCTitleDataset, CCTitleDataset]:
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
    return train_ds, val_ds


def batch_iterate(dataset: CCTitleDataset, batch_size: int, shuffle: bool = True):
    """Simple batch iterator for MLX datasets"""
    indices = list(range(len(dataset)))
    if shuffle:
        mx.random.seed(42)  # Fixed seed for reproducibility
        perm = mx.random.permutation(len(indices)).tolist()
        indices = [indices[i] for i in perm]

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        if len(batch_indices) < batch_size:
            continue  # Skip incomplete batches

        inputs = []
        targets = []
        for idx in batch_indices:
            inp, tgt = dataset[idx]
            inputs.append(inp)
            targets.append(tgt)

        # Stack into batch
        input_batch = mx.stack(inputs)
        target_batch = mx.stack(targets)
        yield input_batch, target_batch


def calculate_loss(model, input_batch, target_batch):
    logits = model(input_batch)
    # Flatten for cross entropy
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = target_batch.reshape(-1)
    loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="mean")
    return loss


def generate_samples(
    model: FaiscaGPT,
    tokenizer: tiktoken.Encoding,
    max_new_tokens: int,
    num_samples: int,
    prompt: str,
    temperature: float,
    top_k: int,
    eot_token: str,
    context_length: int,
) -> None:
    encoded = mx.array(
        tokenizer.encode(prompt, allowed_special={eot_token}),
        dtype=mx.int32,
    ).reshape(1, -1)

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
    encoded: mx.array,
    max_new_tokens: int,
    temperature: float,
    context_length: int,
    top_k: int,
    stop_at_eot: int | None = None,
) -> mx.array:
    for _ in range(max_new_tokens):
        idx_cond = encoded[:, -context_length:]

        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        v, _ = mx.topk(logits, top_k)
        logits = mx.where(logits < v[:, [-1]], -mx.inf, logits)

        probs = mx.softmax(logits, axis=-1)
        idx_next = mx.random.categorical(probs, num_samples=1)

        encoded = mx.concatenate([encoded, idx_next], axis=1)
        if stop_at_eot is not None and idx_next.item() == stop_at_eot:
            break

    return encoded


def train(
    model: FaiscaGPT,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    context_length: int,
    eval_freq: int,
    eval_iter: int,
    train_dataset: CCTitleDataset,
    val_dataset: CCTitleDataset,
    tokenizer: tiktoken.Encoding,
    eot_token: str,
    eval_text: str,
    eval_num_samples: int,
    eval_temperature: float,
    eval_top_k: int,
    eval_max_new_tokens: int,
    batch_size: int,
):
    training_losses, validation_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    # Create loss and gradient function
    def loss_fn(model, input_batch, target_batch):
        return calculate_loss(model, input_batch, target_batch)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    for epoch in range(num_epochs):
        batch_num = 0
        total_batches = len(train_dataset) // batch_size

        for input_batch, target_batch in batch_iterate(
            train_dataset, batch_size, shuffle=True
        ):
            loss, grads = loss_and_grad_fn(model, input_batch, target_batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            tokens_seen += input_batch.size
            global_step += 1
            batch_num += 1

            # run evaluation
            if global_step % eval_freq == 0:
                losses = dict()
                for split, dataset in [
                    ("train", train_dataset),
                    ("validation", val_dataset),
                ]:
                    total_split_loss = 0
                    eval_batches = 0
                    for input_batch, target_batch in batch_iterate(
                        dataset, batch_size, shuffle=False
                    ):
                        if eval_batches < eval_iter:
                            loss = calculate_loss(model, input_batch, target_batch)
                            total_split_loss += loss.item()
                            eval_batches += 1
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
                        f"Batch: {batch_num} / {total_batches} - "
                        f"Tokens seen: {tokens_seen}"
                    )
                )

                training_losses.append(train_loss)
                validation_losses.append(validation_loss)
                track_tokens_seen.append(tokens_seen)

        generate_samples(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=eval_max_new_tokens,
            num_samples=eval_num_samples,
            prompt=eval_text,
            temperature=eval_temperature,
            top_k=eval_top_k,
            eot_token=eot_token,
            context_length=context_length,
        )

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

    epochs_array = mx.linspace(0, num_epochs, len(training_losses))
    # Plot training and validation loss against epochs
    ax1.plot(epochs_array, training_losses, label="Training loss")
    ax1.plot(
        epochs_array,
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

    # Save model parameters in MLX format
    mx.savez(save_path, **model.parameters())
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    ds = hf_datasets.load_dataset("duarteocarmo/ccnews-titles-2016")
    ds_train, ds_val = ds["train"], ds["test"]

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mx.random.seed(42)

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
        eval_num_samples=3,
        eval_temperature=1.0,
        eval_top_k=30,
        eval_max_new_tokens=60,
    )

    train_dataset, val_dataset = create_datasets(
        train_split=ds_train,
        val_split=ds_val,
        language=config.train_language,
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

    # PRE-TRAINING

    model, training_losses, validation_losses, track_tokens_seen = train(
        model=model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_epochs=config.num_epochs,
        context_length=config.context_length,
        eval_freq=config.eval_freq,
        eval_iter=config.eval_iter,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        eot_token=config.eot_token,
        eval_max_new_tokens=config.eval_max_new_tokens,
        eval_text=config.eval_text,
        eval_num_samples=config.eval_num_samples,
        eval_temperature=config.eval_temperature,
        eval_top_k=config.eval_top_k,
        batch_size=config.batch_size,
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

    # SUPERVISED FINE-TUNING
