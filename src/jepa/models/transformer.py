from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn


class SelfAttention(nn.Module):
    embed_dim: int

    @nn.compact
    def __call__(self, x) -> Any:
        q = nn.Dense(self.embed_dim)(x)
        k = nn.Dense(self.embed_dim)(x)
        v = nn.Dense(self.embed_dim)(x)
        scale = jnp.sqrt(self.embed_dim)

        scores = q @ jnp.swapaxes(k, -1, -2)
        scores = scores / scale
        attention_weights = nn.softmax(scores, axis=-1)
        return attention_weights @ v


class MultiHeadSelfAttention(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x) -> Any:
        B, N, D = x.shape
        assert D % self.num_heads == 0
        head_dim = D // self.num_heads

        q = nn.Dense(self.embed_dim)(x)
        k = nn.Dense(self.embed_dim)(x)
        v = nn.Dense(self.embed_dim)(x)

        q = q.reshape(B, N, self.num_heads, head_dim)
        k = k.reshape(B, N, self.num_heads, head_dim)
        v = v.reshape(B, N, self.num_heads, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scores = q @ jnp.swapaxes(k, -1, -2)
        scores = scores / jnp.sqrt(head_dim)
        output = nn.softmax(scores, axis=-1) @ v
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(B, N, D)
        return nn.Dense(self.embed_dim)(output)


class MLP(nn.Module):
    embed_dim: int
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x) -> Any:
        hidden_dim = self.embed_dim * self.mlp_ratio
        x = nn.Dense(hidden_dim)(x)
        x = nn.gelu(x)
        return nn.Dense(self.embed_dim)(x)


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.LayerNorm()(x)
        x = MultiHeadSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
        )(x)
        x = residual + x

        residual = x
        x = nn.LayerNorm()(x)
        x = MLP(embed_dim=self.embed_dim, mlp_ratio=self.mlp_ratio)(x)
        return residual + x


def self_attention(q, k, v):
    key_dim = q.shape[-1]
    scores = q @ k.T
    weights = jax.nn.softmax(scores / jnp.sqrt(key_dim), axis=-1)
    return weights @ v