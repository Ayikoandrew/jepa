# JEPA (Work in progress!)

## Project layout

- `src/jepa/models/`: ViT and transformer modules.
- `src/jepa/embeddings/`: patch and positional embeddings.
- `src/jepa/utils/`: shared tensor utilities.
- `src/jepa/examples/`: example input generation.

## Run

Run the example encoder from the project root:

```bash
uv run jepa
```

The example builds a synthetic 224 x 224 image and prints the encoder output
shape. You can also run it as a module with `uv run python -m jepa`.

## Note:
This is me learning how Joint Embedding Predictive Architecture works. And I am going to explore virtually every literature on JEPA.