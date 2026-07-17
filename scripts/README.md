# Script guide

Executable files directly in `scripts/` are stable user-facing entry points.
Run them from any working directory; every script resolves the repository root
from its own location.

## Normal deployment commands

| Command | Purpose |
|---|---|
| `./scripts/setup.sh` | Bootstrap a fresh clone: initialize dependencies, acquire a checkpoint, verify the host, and build |
| `./scripts/download_models.sh` | Download or offline-import GGML checkpoints |
| `./scripts/build.sh` | Configure and build CPU, CUDA, or CUDA + TensorRT backends |
| `./scripts/verify_setup.sh` | Read-only check of tools, submodules, CUDA, model assets, and optional TensorRT assets |

The usual new-system command is:

```bash
./scripts/setup.sh --backend auto --model q8_0
```

For an explicit CUDA 12 installation and TensorRT:

```bash
./scripts/setup.sh --backend cuda --cuda-root /usr/local/cuda-12 --trt
```

Use `--help` on any public command for all options.

## Developer commands

| Command | Purpose |
|---|---|
| `./scripts/development/export_onnx.sh` | Export the optional TensorRT ONNX graphs from a GGML checkpoint |
| `./scripts/development/make_goldens.sh` | Regenerate CPU correctness baselines |

Developer entry points live in `development/`. The Python implementation
details used by ONNX export live in `convert/`; neither directory is part of
the normal deployment flow.

## Setup internals

Low-level environment and dependency provisioning scripts live in `setup/`.
Normal deployments should invoke `scripts/setup.sh` or `scripts/build.sh`,
which call these helpers with the correct paths. See
[`setup/README.md`](setup/README.md) before invoking them directly.
