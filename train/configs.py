"""
Model size configs — every entry maps to one training run.
'medium' matches the Rust default (4 layers, d_model=256, 8 heads, d_ff=1024).
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    max_seq_len: int
    vocab_size: int

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_params(self) -> int:
        """Rough parameter count (embedding + layers + lm_head)."""
        embed = self.vocab_size * self.d_model
        per_layer = (
            4 * self.d_model * self.d_model   # Q K V O projections
            + 2 * self.d_model * self.d_ff    # FFN up+down
            + 2 * self.d_model                # two RMSNorm weights
        )
        lm_head = self.d_model * self.vocab_size
        norm = self.d_model
        return embed + self.n_layers * per_layer + lm_head + norm


# Rust default: Config::new(256, 8, 4, 1024, 512, 8192)
CONFIGS: dict[str, ModelConfig] = {
    "tiny": ModelConfig(
        name="tiny",
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=256,
        max_seq_len=512,
        vocab_size=4096,
    ),
    "small": ModelConfig(
        name="small",
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=512,
        vocab_size=4096,
    ),
    "medium": ModelConfig(       # ← matches Rust Config::default()
        name="medium",
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        max_seq_len=512,
        vocab_size=8192,
    ),
    "large": ModelConfig(
        name="large",
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_seq_len=512,
        vocab_size=8192,
    ),
}
