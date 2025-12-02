# Playhouse

WIP port of PufferLib, for self-educational purposes

Differences from PufferLib:

- Tetris env is written in parallel Rust (no vector.py required for 1-GPU training)
- Type checks with `basedpyright`
- Typed config files

## Instructions

Rebuild Rust dependencies

```sh
uv sync --reinstall-package tetris_rust
```

Check Tetris environment performance (~11M SPS on a M4 Max)

```sh
uv run -m playhouse.environments.tetris
```

Watch Tetris render

```sh
uv run -m playhouse.environments.tetris.watch
```
