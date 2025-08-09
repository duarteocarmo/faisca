from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def update_config(config, dictionary):
    for k, v in dictionary.items():
        setattr(config, k, v)
    return config


def show_config(config: SimpleNamespace):
    for k in sorted(config.__dict__):
        print(f"{k}: {getattr(config, k)}")


class FaiscaDataset(Dataset):
    def __init__(self, data: str, max_length: int, tokenizer, stride):
        self.tokenizer = tokenizer
        chars = sorted(list(set(data)))
        data_size, self.vocab_size = len(data), len(chars)
        print(f"Vocab size: {self.vocab_size}")
        print(f"Data size: {data_size}")

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
        """
        Encode a string into token ids.
        """
        encoded = self.tokenizer.encode(text)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        return encoded_tensor

    def decode(self, token_ids: torch.Tensor) -> str:
        flat = token_ids.squeeze(0)
        return self.tokenizer.decode(flat.tolist())


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dimension_in: int,
        dimension_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool,
    ):
        super().__init__()
        if dimension_out % num_heads != 0:
            raise ValueError(
                f"dimension_out must be divisible by num_heads, got {dimension_out} and {num_heads}"
            )

        self.dimension_out = dimension_out
        self.num_heads = num_heads
        self.head_dim = dimension_out // num_heads

        self.W_query = nn.Linear(dimension_in, dimension_out, bias=qkv_bias)
        self.W_key = nn.Linear(dimension_in, dimension_out, bias=qkv_bias)
        self.W_value = nn.Linear(dimension_in, dimension_out, bias=qkv_bias)
        self.out_projection = nn.Linear(dimension_out, dimension_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, number_of_tokens, dimension_in = x.shape

        # x shape: (batch_size, number_of_tokens, dimension_in)

        # pass through the linear layers
        keys = self.W_key(x)  # shape (batch_size, number_of_tokens, dimension_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # split into heads
        keys = keys.view(
            batch_size, number_of_tokens, self.num_heads, self.head_dim
        )  # shape (batch_size, number_of_tokens, num_heads, head_dim)
        queries = queries.view(
            batch_size, number_of_tokens, self.num_heads, self.head_dim
        )
        values = values.view(
            batch_size, number_of_tokens, self.num_heads, self.head_dim
        )

        # transpose to get the shape right for the attention scores
        keys = keys.transpose(
            1, 2
        )  # shape (batch_size, num_heads, number_of_tokens, head_dim)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # compute the attention scores
        attention_scores = queries @ keys.transpose(2, 3)

        # create the mask
        mask_bool = self.mask.bool()[:number_of_tokens, :number_of_tokens]

        # apply the mask
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        # compute the attention weights
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)

        # compute the context vector
        context_vector = (attention_weights @ values).transpose(1, 2)

        # combine the heads
        context_vector = context_vector.contiguous().view(
            batch_size, number_of_tokens, self.dimension_out
        )
        context_vector = self.out_projection(context_vector)

        return context_vector


class LayerNorm(nn.Module):
    def __init__(self, embedding_dimension: int):
        super().__init__()
        self.eps = 1e-5  # small constant to avoid division by zero
        self.scale = nn.Parameter(torch.ones(embedding_dimension))
        self.shift = nn.Parameter(torch.zeros(embedding_dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    Gelu activation:
    xi > 0 -> output ~ xi
    -2 < xi < 2 -> output ~ 0
    xi << -2 -> output ~ 0
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, embedding_dimension: int, hidden_expansion_factor: int = 4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(
                embedding_dimension, hidden_expansion_factor * embedding_dimension
            ),
            GELU(),
            nn.Linear(
                hidden_expansion_factor * embedding_dimension, embedding_dimension
            ),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        context_length: int,
        num_heads: int,
        qkv_bias: bool,
        dropout_rate: float,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            dimension_in=embedding_dimension,
            dimension_out=embedding_dimension,
            context_length=context_length,
            dropout=dropout_rate,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.feed_forward = FeedForward(
            embedding_dimension=embedding_dimension,
        )
        self.norm1 = LayerNorm(embedding_dimension=embedding_dimension)
        self.norm2 = LayerNorm(embedding_dimension=embedding_dimension)
        self.drop_shortcut = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shortcut connection // attention
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
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
                    context_length=context_length,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer_norm = LayerNorm(embedding_dimension=embedding_dimension)
        self.out_head = nn.Linear(embedding_dimension, vocab_size, bias=False)

        n_params = sum(p.numel() for p in self.transformer_blocks.parameters())
        n_params_million = n_params / 1e6
        print(f"Number of parameters in transformer blocks: {n_params_million:.2f}M")

        n_params_all = sum(p.numel() for p in self.parameters())
        n_params_all_million = n_params_all / 1e6
        print(f"Number of parameters in all layers: {n_params_all_million:.2f}M")

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = in_idx.shape
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


if __name__ == "__main__":
    # bocage_text = pathlib.Path("./datasets/bocage-mini.txt").read_text(
    #     encoding="ISO-8859-1"
    # )

    import os
    import urllib.request

    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            bocage_text = file.read()
    context_length = 256

    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")

    config = SimpleNamespace()

    config = update_config(
        config,
        {
            "vocab_size": 50257,  # Vocabulary size
            "context_length": context_length,  # Context length
            "embedding_dimension": 768,  # Embedding dimension
            "num_heads": 12,  # Number of attention heads
            "num_layers": 12,  # Number of layers
            "dropout_rate": 0.1,  # Dropout rate
            "qkv_bias": False,  # QKV bias
            "num_epochs": 10,
            "learning_rate": 1e-3,
            "weight_decay": 0.1,
            "device": "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu",
            "eval_freq": 5,
            "eval_iter": 1,
            "batch_size": 2,
            "val_split": 0.1,
            "num_workers": 0,
            "drop_last": True,
            "max_length": 256,
            "stride": 128,
        },
    )

    split_idx = int(len(bocage_text) * (1 - config.val_split))

    train_dataloader = DataLoader(
        FaiscaDataset(
            bocage_text[:split_idx],
            max_length=config.max_length,
            tokenizer=tokenizer,
            stride=config.stride,
        ),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=config.drop_last,
        num_workers=config.num_workers,
    )
    val_dataloader = DataLoader(
        FaiscaDataset(
            bocage_text[split_idx:],
            max_length=config.max_length,
            tokenizer=tokenizer,
            stride=config.stride,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=config.drop_last,
        num_workers=config.num_workers,
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

    training_losses, validation_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    model.to(config.device)

    def calculate_loss(input_batch, target_batch, model):
        input_batch, target_batch = (
            input_batch.to(config.device),
            target_batch.to(config.device),
        )
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch.flatten()
        )
        return loss

    for epoch in range(config.num_epochs):
        model.train()
        size_train_dataloader = len(train_dataloader)
        batch_num = 0

        for input_batch, target_batch in train_dataloader:
            optimizer.zero_grad()
            loss = calculate_loss(input_batch, target_batch, model)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            batch_num += 1

            # run evaluation
            if global_step % config.eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    total_train_loss = 0
                    for i, (input_batch, target_batch) in enumerate(train_dataloader):
                        if i < config.eval_iter:
                            loss = calculate_loss(input_batch, target_batch, model)
                            total_train_loss += loss.item()
                        else:
                            break

                    train_loss = total_train_loss / config.eval_iter

                    total_validation_loss = 0
                    for i, (input_batch, target_batch) in enumerate(val_dataloader):
                        if i < config.eval_iter:
                            loss = calculate_loss(input_batch, target_batch, model)
                            total_validation_loss += loss.item()
                        else:
                            break

                    validation_loss = total_validation_loss / config.eval_iter

                    print(
                        f"Epoch: {epoch} - Step: {global_step} - Train Loss: {train_loss} - Validation Loss: {validation_loss} - Batch: {batch_num} / {size_train_dataloader} - Tokens seen: {tokens_seen}"
                    )

                    training_losses.append(train_loss)
                    validation_losses.append(validation_loss)
                    track_tokens_seen.append(tokens_seen)

                    model.train()

        # print sample
        model.eval()
        start_context = "He showed it to me with"
        context_size = model.positional_embedding.weight.shape[0]
        encoded = train_dataloader.dataset.encode(start_context).to(config.device)

        max_new_tokens = 125

        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = encoded[:, -config.context_length :]
                logits = model(idx_cond)
                logits = logits[:, -1, :]
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                encoded = torch.cat((encoded, idx_next), dim=1)

        print(f"\nGenerated: {encoded}")
        decoded = train_dataloader.dataset.decode(encoded[0])
        decoded = decoded.replace("\n", " ")
        print(f"Decoded: {decoded}\n")

        model.train()

    fig, ax1 = plt.subplots()

    epochs_tensor = torch.linspace(0, config.num_epochs, len(training_losses))
    # Plot training and validation loss against epochs
    ax1.plot(epochs_tensor, training_losses, label="Training loss")
    ax1.plot(epochs_tensor, validation_losses, linestyle="-.", label="Validation loss")
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
    # plt.show()
