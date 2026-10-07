import flax.linen as nn

from jepa.embeddings import PatchEmbedding, PositionalEmbedding
from .transformer import TransformerBlock


class ViTEncoder(nn.Module):
    patch_size: int
    embed_dim: int
    num_patches: int
    num_heads: int
    num_layers: int
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x):
        x = PatchEmbedding(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
        )(x)
        x = PositionalEmbedding(
            num_patches=self.num_patches,
            embed_dim=self.embed_dim,
        )(x)

        for _ in range(self.num_layers):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
            )(x)

        return nn.LayerNorm()(x)