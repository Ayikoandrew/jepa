import jax.numpy as jnp


def make_synthetic_image(
    height: int = 224,
    width: int = 224,
    channels: int = 3,
) -> jnp.ndarray:
    values = jnp.arange(height * width * channels, dtype=jnp.float32)
    image = values.reshape(1, height, width, channels)
    return image / values.size