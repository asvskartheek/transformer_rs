"""
MLX transformer — architecture mirrors the Rust implementation:
  • RMSNorm (pre-norm)
  • Multi-head self-attention with RoPE
  • GELU feed-forward
  • Causal language model head
"""
import math
import mlx.core as mx
import mlx.nn as nn

from configs import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.scale = math.sqrt(cfg.head_dim)

        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = nn.RoPE(cfg.head_dim, traditional=False, base=10_000.0)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        B, L, D = x.shape

        q = self.wq(x).reshape(B, L, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(B, L, self.n_heads, self.head_dim)
        v = self.wv(x).reshape(B, L, self.n_heads, self.head_dim)

        # RoPE expects [B, n_heads, L, head_dim]
        q = self.rope(q.transpose(0, 2, 1, 3))
        k = self.rope(k.transpose(0, 2, 1, 3))
        v = v.transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) / self.scale  # [B, H, L, L]
        if mask is not None:
            scores = scores + mask

        scores = mx.softmax(scores, axis=-1)
        out = (scores @ v).transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.gelu(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        x = x + self.attn(self.attn_norm(x), mask=mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def __call__(self, tokens: mx.array, mask: mx.array | None = None) -> mx.array:
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.lm_head(self.norm(x))

    def loss(self, tokens: mx.array) -> mx.array:
        inputs  = tokens[:, :-1]   # [B, L-1]
        targets = tokens[:, 1:]    # [B, L-1]

        L = inputs.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        mask = mask.astype(self.embed.weight.dtype)

        logits = self(inputs, mask=mask)                  # [B, L-1, V]
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size),
            targets.reshape(-1),
        )
        return loss.mean()
