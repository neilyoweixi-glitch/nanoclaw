"""
Autoresearch training script - Mac Mini Edition (Apple Silicon compatible)
Adapted from karpathy/autoresearch for MPS/CPU execution.

Usage: python train.py
"""

import os
import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Device selection - MPS for Apple Silicon, CPU fallback
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print(f"Using MPS (Apple Silicon)")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"Using CUDA")
else:
    DEVICE = "cpu"
    print(f"Using CPU (slow)")

# ---------------------------------------------------------------------------
# Configuration (agent can modify these)
# ---------------------------------------------------------------------------

DEPTH = 1  # EXP 6: single layer
VOCAB_SIZE = 4096  # Smaller vocab for Mac
SEQUENCE_LEN = 256  # Shorter sequences for limited memory
BATCH_SIZE = 64  # EXP 6: large batch
TOTAL_BATCH_SIZE = 2**14  # Grad accumulation target (~16K tokens)
TIME_BUDGET = 300  # 5 minutes for Mac Mini
LEARNING_RATE = 1e-3  # EXP 6: aggressive constant LR
WEIGHT_DECAY = 0.0  # EXP 6: no weight decay
WARMUP_STEPS = 5  # EXP 6: minimal warmup
MIN_LR = 1e-3  # EXP 6: same as max for constant LR

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = SEQUENCE_LEN
    vocab_size: int = VOCAB_SIZE
    n_layer: int = DEPTH
    n_head: int = 4  # EXP 6: same heads
    n_embd: int = 32  # EXP 6: tiny embedding
    dropout: float = 0.0

    def __post_init__(self):
        # Auto-compute derived values
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    """Apply rotary position embeddings."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = -x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim

        # Query, Key, Value projections
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Scaled dot-product attention (Mac compatible)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # SwiGLU-style MLP (simpler than gated version)
        hidden = 4 * config.n_embd
        self.c_fc1 = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU: gate * silu(x)
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.layer_idx = layer_idx

    def forward(self, x, cos, sin):
        x = x + self.attn(norm(x), cos, sin)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'h': nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.transformer['wte'].weight

        # Init
        self.apply(self._init_weights)

        # Count params
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model: {n_params/1e6:.2f}M params, {config.n_layer} layers")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.config.n_layer)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()

        # Token embeddings
        tok_emb = self.transformer['wte'](idx)

        # Rotary embeddings precompute
        freqs = 1.0 / (10000 ** (torch.arange(0, self.config.head_dim, 2, device=idx.device).float() / self.config.head_dim))
        pos = torch.arange(T, device=idx.device)
        freqs = torch.outer(pos, freqs)
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]

        x = tok_emb
        for block in self.transformer['h']:
            x = block(x, cos, sin)
        x = norm(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


# ---------------------------------------------------------------------------
# Data (synthetic for Mac - no download needed)
# ---------------------------------------------------------------------------

def get_batch(batch_size=BATCH_SIZE, seq_len=SEQUENCE_LEN):
    """Generate synthetic batch (replace with real data later)."""
    idx = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device=DEVICE)
    targets = torch.roll(idx, -1, dims=1)
    targets[:, -1] = -100  # Ignore last position
    return idx, targets


def estimate_loss(model, eval_iters=10):
    """Estimate validation loss."""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            _, loss = model(*get_batch())
            if loss is not None:
                losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float('inf')


def bpb_from_loss(loss):
    """Convert cross-entropy loss to bits per byte."""
    # log2(e) * loss / (vocab_size factor)
    return loss * math.log2(math.e)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def get_lr(step, warmup_steps=WARMUP_STEPS, max_lr=LEARNING_RATE, min_lr=MIN_LR, total_steps=1000):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    # Cosine decay from max_lr to min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def configure_optimizer(model, lr=LEARNING_RATE, wd=WEIGHT_DECAY):
    """Configure AdamW optimizer with proper weight decay."""
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': wd},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), fused=False)
    return optimizer


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    print(f"Device: {DEVICE}")
    print(f"Config: depth={DEPTH}, vocab={VOCAB_SIZE}, seq={SEQUENCE_LEN}")

    config = GPTConfig()
    model = GPT(config).to(DEVICE)
    optimizer = configure_optimizer(model)

    # Training
    grad_accum_steps = TOTAL_BATCH_SIZE // (BATCH_SIZE * SEQUENCE_LEN)
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"LR schedule: warmup={WARMUP_STEPS}, max_lr={LEARNING_RATE}, min_lr={MIN_LR}")

    start_time = time.time()
    step = 0
    tokens_seen = 0
    best_val_bpb = float('inf')

    # Estimate total steps for LR schedule
    total_steps = int(TIME_BUDGET / 0.7)

    # Initialize learning rate
    lr = get_lr(0, warmup_steps=WARMUP_STEPS, max_lr=LEARNING_RATE, min_lr=MIN_LR, total_steps=total_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print(f"\nStarting training ({TIME_BUDGET}s budget)...")

    while True:
        # Check time budget
        elapsed = time.time() - start_time
        if elapsed >= TIME_BUDGET:
            break

        # Gradient accumulation
        optimizer.zero_grad()
        for _ in range(grad_accum_steps):
            idx, targets = get_batch()
            _, loss = model(idx, targets)
            loss = loss / grad_accum_steps
            loss.backward()
        optimizer.step()

        tokens_seen += BATCH_SIZE * SEQUENCE_LEN * grad_accum_steps
        step += 1

        # Update learning rate for next step
        lr = get_lr(step, warmup_steps=WARMUP_STEPS, max_lr=LEARNING_RATE, min_lr=MIN_LR, total_steps=total_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        step += 1

        # Log every 10 steps
        if step % 10 == 0:
            val_loss = estimate_loss(model)
            val_bpb = bpb_from_loss(val_loss)
            print(f"step {step:4d} | loss {val_loss:.4f} | val_bpb {val_bpb:.6f} | lr {lr:.2e} | tokens {tokens_seen/1e6:.1f}M")

            if val_bpb < best_val_bpb:
                best_val_bpb = val_bpb

    # Final evaluation
    final_loss = estimate_loss(model, eval_iters=50)
    final_bpb = bpb_from_loss(final_loss)
    duration = time.time() - start_time

    # Get memory usage
    if DEVICE == "mps":
        mem_mb = torch.mps.current_allocated_memory() / 1024 / 1024
    elif DEVICE == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        mem_mb = 0

    # Print results in standard format
    print("\n---")
    print(f"val_bpb:          {final_bpb:.6f}")
    print(f"training_seconds: {duration:.1f}")
    print(f"peak_vram_mb:     {mem_mb:.1f}")
    print(f"total_tokens_M:   {tokens_seen/1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {sum(p.numel() for p in model.parameters())/1e6:.1f}")
    print(f"depth:            {config.n_layer}")
    print("---")

    return final_bpb


if __name__ == "__main__":
    train()
