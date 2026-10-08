import flax.linen as nn

from jepa.utils import patchify


class PatchEmbedding(nn.Module):
    patch_size: int
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        patches = patchify(x, self.patch_size)
        return nn.Dense(self.embed_dim)(patches)

class PositionalEmbedding(nn.Module):
    num_patches: int
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_patches, self.embed_dim),
        )
        return x + pos_embedding