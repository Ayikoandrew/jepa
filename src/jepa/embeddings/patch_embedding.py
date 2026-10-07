import flax.linen as nn

from jepa.utils import patchify


class PatchEmbedding(nn.Module):
    patch_size: int
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        patches = patchify(x, self.patch_size)
        return nn.Dense(self.embed_dim)(patches)