import flax.linen as nn


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