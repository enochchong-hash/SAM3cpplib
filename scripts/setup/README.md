# Setup helpers

These scripts implement low-level host setup for the public commands one
directory above. They are kept here so the main `scripts/` directory contains
only intentional user entry points.

## CUDA environment discovery

`cuda_env.sh` locates versioned CUDA installations and exports the selected
toolkit when sourced. Selection order is an explicit path, CUDA environment
variables, `/usr/local/cuda-12`, `/usr/local/cuda`, then other versioned
installations under `/usr/local` and `/opt`.

```bash
# Inspect the toolkit chosen automatically
./scripts/setup/cuda_env.sh

# Activate it in the current shell
source scripts/setup/cuda_env.sh

# Select a specific installation and print reusable exports
./scripts/setup/cuda_env.sh --cuda-root /usr/local/cuda-12 --print-env
```

Layouts using `lib`, `lib64`, or `targets/<architecture>/lib` are supported.
Most users do not need to source this manually because `scripts/build.sh`
does it automatically.

## TensorRT SDK provisioning

`setup_tensorrt.sh` vendors the pinned TensorRT headers and runtime libraries
required to compile the optional backend. It is a large, one-time network
operation. The recommended public workflow is:

```bash
./scripts/setup.sh --backend cuda --trt
```

For manual or offline reuse:

```bash
./scripts/setup/setup_tensorrt.sh
./scripts/setup/setup_tensorrt.sh --copy-from /path/to/other/3rdparty
./scripts/build.sh cuda --trt
```

TensorRT engines are GPU- and TensorRT-version-specific runtime caches. This
helper installs build/runtime dependencies; it does not replace the GGML
checkpoint and does not make engine caches portable between unlike systems.
