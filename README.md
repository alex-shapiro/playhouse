# Playhouse

WIP Subset of PufferLib, ported for self-educational purposes

## Instructions

Rebuild Rust dependencies

```sh
uv sync --reinstall-package tetris_rust
```

Check Tetris environment performance (~9.3M SPS on a M4 Max)

```sh
uv run -m playhouse.environments.tetris
```

Watch Tetris render

```sh
uv run -m playhouse.environments.tetris.watch
```
