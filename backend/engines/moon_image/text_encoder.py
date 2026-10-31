"""
backend/engines/moon_image/text_encoder.py
Original learned Text Conditioning Transformer for Moon Image.
Trained from scratch from random initialization. Zero imported pretrained weights.
"""
import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

class SimpleTextTokenizer:
    """Byte-level fallback tokenizer ensuring 100% deterministic, zero-dependency tokenization."""
    def __init__(self, vocab_size: int = 8192, max_length: int = 77):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

    def encode(self, text: str) -> torch.Tensor:
        if not text:
            tokens = [self.bos_token_id, self.eos_token_id]
        else:
            raw_bytes = text.strip().encode("utf-8", errors="replace")
            # Map UTF-8 bytes [0..255] offset by 4
            tokens = [self.bos_token_id] + [b + 4 for b in raw_bytes[: self.max_length - 2]] + [self.eos_token_id]
        
        # Pad or truncate to max_length
        if len(tokens) < self.max_length:
            tokens = tokens + [self.pad_token_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[: self.max_length]
            tokens[-1] = self.eos_token_id
            
        return torch.tensor(tokens, dtype=torch.long)

    def batch_encode(self, texts: list) -> torch.Tensor:
        return torch.stack([self.encode(t) for t in texts], dim=0)


class MoonTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class MoonTextEncoder(nn.Module):
    """
    Learned text conditioning model for Moon Image.
    Produces both sequence-level cross-attention context and pooled text embeddings.
    """
    def __init__(
        self,
        vocab_size: int = 8192,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        max_length: int = 77
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        self.blocks = nn.ModuleList([
            MoonTransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.tokenizer = SimpleTextTokenizer(vocab_size=vocab_size, max_length=max_length)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: LongTensor [batch_size, seq_len]
        Returns:
            context: FloatTensor [batch_size, seq_len, embed_dim]
            pooled: FloatTensor [batch_size, embed_dim]
        """
        b, seq_len = input_ids.shape
        device = input_ids.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(b, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)

        context = self.final_layer_norm(x)
        # Pooled embedding is the representation at EOS token / first non-pad index
        pooled = context[:, 0, :]
        return context, pooled
