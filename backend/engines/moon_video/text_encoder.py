"""
backend/engines/moon_video/text_encoder.py
Text and Camera Motion Conditioning Encoder for Moon Video.
Learned from random initialization. Zero imported pretrained weights.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
from ..moon_image.text_encoder import SimpleTextTokenizer, MoonTransformerBlock

class MoonVideoTextEncoder(nn.Module):
    """
    Conditions video generation on both textual prompt and camera motion trajectories.
    """
    def __init__(
        self,
        vocab_size: int = 8192,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        max_length: int = 77,
        num_camera_motions: int = 10
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        self.camera_embedding = nn.Embedding(num_camera_motions, embed_dim)

        self.blocks = nn.ModuleList([
            MoonTransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.tokenizer = SimpleTextTokenizer(vocab_size=vocab_size, max_length=max_length)

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

    def forward(self, input_ids: torch.Tensor, camera_motion_id: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, seq_len = input_ids.shape
        device = input_ids.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(b, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # Inject camera motion vector if specified
        if camera_motion_id is not None:
            cam_id = camera_motion_id.clamp(0, self.camera_embedding.num_embeddings - 1)
            cam_emb = self.camera_embedding(cam_id).unsqueeze(1)
            x = x + cam_emb

        for block in self.blocks:
            x = block(x)

        context = self.final_layer_norm(x)
        pooled = context[:, 0, :]
        return context, pooled
