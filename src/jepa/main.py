from jax import random

from jepa.examples.synthetic import make_synthetic_image
from jepa.models import ViTEncoder


def main() -> None:
    image = make_synthetic_image()
    model = ViTEncoder(
        patch_size=16,
        embed_dim=384,
        num_patches=196,
        num_heads=6,
        num_layers=6,
    )
    params = model.init(random.PRNGKey(0), image)
    tokens = model.apply(params, image)
    print(f"Output token shape: {tokens.shape}") # type: ignore