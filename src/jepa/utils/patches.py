import jax.numpy as jnp


def patchify(x: jnp.ndarray, patch_size: int) -> jnp.ndarray:
    B, H, W, C = x.shape
    assert H % patch_size == 0
    assert W % patch_size == 0

    grid_height = H // patch_size
    grid_width = W // patch_size
    x = x.reshape(
        B,
        grid_height,
        patch_size,
        grid_width,
        patch_size,
        C,
    )
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(
        B,
        grid_height * grid_width,
        patch_size * patch_size * C,
    )