from typing import Any

import flax.linen as nn

from jepa.embeddings import PatchEmbedding, PositionalEmbedding


class ViTInput(nn.Module):
    patch_size: int
    num_patches: int
    embed_dim: int

    @nn.compact
    def __call__(self, x) -> Any:
        x = PatchEmbedding(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
        )(x)
        return PositionalEmbedding(
            num_patches=self.num_patches,
            embed_dim=self.embed_dim,
        )(x)