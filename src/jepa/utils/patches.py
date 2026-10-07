import jax.numpy as jnp


def patchify(x: jnp.ndarray, patch_size: int) -> jnp.ndarray:
    batch_size, height, width, channels = x.shape
    assert height % patch_size == 0
    assert width % patch_size == 0

    grid_height = height // patch_size
    grid_width = width // patch_size
    x = x.reshape(
        batch_size,
        grid_height,
        patch_size,
        grid_width,
        patch_size,
        channels,
    )
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(
        batch_size,
        grid_height * grid_width,
        patch_size * patch_size * channels,
    )