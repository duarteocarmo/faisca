# /// script
# dependencies = [
#   "transformers",
#   "torch",
#   "accelerate",
#   "matplotlib",
# ]
# ///
# Notes: Mostly stolen from https://github.com/open-thought/tiny-grpo

import random
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Self

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import (
    GenerationConfig,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)


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


def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Monte-Carlo approximation of KL divergence, k3 estimator, see: http://joschu.net/blog/kl-approx.html
    """

    log_ratio = log_probs_ref.float() - log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    return log_ratio.exp() - log_ratio - 1


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


def load_model(
    model_name_or_path: str = "gpt2",
    device_map=None,
) -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(
        model_name_or_path,
        dtype=torch.float32,  # Use float32 for MPS compatibility
        device_map=device_map,
    )
    return model, tokenizer


# Positive sentiment keywords for reward calculation
POSITIVE_KEYWORDS = {
    "happy",
    "joyful",
    "excellent",
    "fantastic",
    "great",
    "positive",
    "good",
    "wonderful",
    "delightful",
    "pleasant",
    "amazing",
    "awesome",
    "brilliant",
    "beautiful",
    "perfect",
    "lovely",
    "cheerful",
    "bright",
    "optimistic",
    "successful",
    "victorious",
    "triumphant",
    "glorious",
    "magnificent",
    "outstanding",
    "superb",
    "marvelous",
    "incredible",
    "fabulous",
    "splendid",
}


def calculate_reward_for(text: str) -> float:
    """Calculate binary reward based on positive keywords."""
    words = set(text.lower().split())
    return 1.0 if words & POSITIVE_KEYWORDS else 0.0


@torch.no_grad()
def rollout(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    num_rollouts: int,
    max_length: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    model.eval()

    # Generate sentences from scratch with a simple prompt
    prompt = "Today I feel"
    model_inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to("mps")

    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )

    input_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
    model_inputs["input_ids"] = input_ids

    # Generate completions
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    sequence_ids = model.generate(
        **model_inputs, generation_config=generation_config
    )
    completions = tokenizer.batch_decode(
        sequence_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )

    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_ids.shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    # Calculate rewards based on sentiment
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    for i, completion in enumerate(completions):
        reward = calculate_reward_for(completion)
        returns[i] = reward

    return (
        sequence_ids,
        returns.to(sequence_ids.device),
        action_mask,
        completions,
    )


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: GPT2LMHeadModel,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs


def main():
    # default config
    # config = {
    #     "group_size": 12,
    #     "rollouts_per_step": 32,
    #     "train_batch_size": 16,
    #     "max_length": 50,
    #     "training_steps": 10,
    # }

    config = {
        "group_size": 8,
        "rollouts_per_step": 24,
        "train_batch_size": 8,
        "max_length": 25,
        "training_steps": 10,
        "seed": 42,
    }

    seed = config["seed"]
    model_name = "gpt2"
    checkpoint_path = Path("./output")
    checkpoint_interval = 20
    train_batch_size = config["train_batch_size"]
    lr = 5e-6
    kl_weight = 0.01
    clip_eps = 0.2
    training_steps = config["training_steps"]

    group_size = config["group_size"]
    rollouts_per_step = config["rollouts_per_step"]
    epochs_per_step = 1
    max_norm = 1.0  # gradient clipping

    # rollout params
    max_length = config["max_length"]  # Shorter for sentiment sentences
    top_p = 1.0
    temperature = 1.0

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    cpu_device = torch.device("cpu")
    init_rng(seed)

    print(f"Using device: {device}")

    reference_model, _ = load_model(model_name, device_map=device)
    model, tokenizer = load_model(model_name, device_map=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    reference_model.eval()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    pad_token_id = tokenizer.eos_token_id

    # No need for external data - we generate from scratch
    print("Starting sentiment training with GPT-2...")

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    # Track metrics for plotting
    step_rewards = []
    step_kl_divergences = []

    for k in range(training_steps):  # Fixed number of training steps
        rollout_returns = []
        replay_buffer.clear()

        with torch.no_grad():
            # Generate rollouts for sentiment training
            sequence_ids, returns, action_mask, completions = rollout(
                model,
                tokenizer,
                num_rollouts=group_size * rollouts_per_step,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
            )

            print(f"Step {k}: Generated {len(completions)} sentences")
            print(f"Sample completions: {completions[:3]}")
            print(f"Returns: {returns.sum().item():.2f}/{len(returns)}")

            rollout_returns.append(returns.cpu())

            advantages = group_advantages(returns)
            attention_mask = sequence_ids != pad_token_id

            log_probs = sequences_log_probs(
                model=model,
                sequence_ids=sequence_ids,
                attention_mask=attention_mask,
            )
            log_probs_ref = sequences_log_probs(
                model=reference_model,
                sequence_ids=sequence_ids,
                attention_mask=attention_mask,
            )
            kl = approx_kl_divergence(
                log_probs=log_probs,
                log_probs_ref=log_probs_ref,
                action_mask=action_mask,
            )

            experience = Experience(
                sequences=sequence_ids,
                action_log_probs=log_probs,
                log_probs_ref=log_probs_ref,
                returns=returns,
                advantages=advantages,
                attention_mask=attention_mask,
                action_mask=action_mask,
                kl=kl,
            )
            replay_buffer.append(experience.to(cpu_device))

        torch.mps.empty_cache() if device.type == "mps" else torch.cuda.empty_cache()
        episode_return_sum = torch.stack(rollout_returns).sum()
        print(f"Returns of step {k}: {episode_return_sum:.4f}")

        # Track metrics for plotting
        step_rewards.append(episode_return_sum.item())

        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )

        # Track average KL divergence for this step
        step_kl_values = []

        for step_epoch in range(epochs_per_step):
            model.train()

            for exp in experience_sampler:
                exp: Experience

                exp = exp.to(device)

                optimizer.zero_grad()

                log_probs = sequences_log_probs(
                    model,
                    sequence_ids=exp.sequences,
                    attention_mask=exp.attention_mask,
                )

                loss, kl = objective(log_probs=log_probs, experience=exp)

                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    continue

                loss.backward()
                grad_norm = clip_grad_norm_(
                    model.parameters(), max_norm=max_norm
                )
                print(
                    f"{step_epoch}: kl={kl: .4f}, grad_norm={grad_norm: .4f}"
                )

                optimizer.step()
                step_kl_values.append(kl.item())

        # Track average KL divergence for this step
        if step_kl_values:
            avg_kl = sum(step_kl_values) / len(step_kl_values)
            step_kl_divergences.append(avg_kl)

        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and (k + 1) % checkpoint_interval == 0
        ):
            model.save_pretrained(checkpoint_path / f"step_{k}")

    if checkpoint_path is not None:
        model.save_pretrained(checkpoint_path / f"step_{k}")

    # Plot training metrics
    print("\nGenerating training plots...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

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
    plt.savefig("training_metrics.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Training completed! Final metrics:")
    print(f"  - Final reward: {step_rewards[-1]:.2f}")
    print(
        f"  - Final KL divergence: {step_kl_divergences[-1]:.4f}"
        if step_kl_divergences
        else "  - No KL divergence data"
    )
    print("  - Training plots saved to: training_metrics.png")


if __name__ == "__main__":
    main()
