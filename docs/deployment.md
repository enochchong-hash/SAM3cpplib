# Standalone deployment

`sam3cpplib` does not need `release/sam3` at runtime. The reference project
stores large checkpoints outside Git and downloads pre-converted GGML files
from the `PABannier/sam3.cpp` Hugging Face repository. This library now owns
the same acquisition workflow under `scripts/download_models.sh`.

## Fresh system

Install a C++17 compiler, CMake, Git, and either curl or wget. Then run:

```bash
git clone --recurse-submodules <sam3cpplib-url>
cd sam3cpplib
./scripts/setup.sh --backend auto --model q8_0
```

If the repository was cloned without submodules, `setup.sh` initializes them.
It downloads `resources/models/sam3-q8_0.ggml`, detects a CUDA toolkit when
available, and otherwise produces a CPU build.

Useful explicit configurations:

```bash
./scripts/setup.sh --backend cpu
./scripts/setup.sh --backend cuda --cuda-root /usr/local/cuda-12
./scripts/setup.sh --backend cuda --cuda-arch '86;89;120'
```

The CUDA selector supports `/usr/local/cuda-12/bin` with libraries in
`/usr/local/cuda-12/lib`, as well as `lib64`, target-specific library folders,
other `/usr/local/cuda-*` versions, `/opt/cuda-*`, and the usual CUDA
environment variables. To activate it in an interactive shell:

```bash
source scripts/setup/cuda_env.sh
```

## Checkpoints and offline hosts

The default q8_0 checkpoint is approximately 1.1 GB. F16 is approximately
1.8 GB. List every published variant with:

```bash
./scripts/download_models.sh --list
```

Use a mirror or copy from removable storage without editing the script:

```bash
SAM3_MODEL_BASE_URL=https://mirror.example/models ./scripts/download_models.sh q8_0
./scripts/download_models.sh --source-dir /mnt/models q8_0
```

Only the checkpoint is required for the portable CPU/CUDA GGML backend.
The source library is MIT-licensed, but the model weights remain under Meta's
SAM model license. Check those terms before redistributing the checkpoint
itself. Keeping weights out of Git and acquiring them on the target host also
keeps the code package small and separates these licenses.

## TensorRT is optional

TensorRT does not replace the GGML checkpoint. The GGML file supplies model
metadata, tokenizer data, and weights used by the complete implementation.
TensorRT accelerates selected subgraphs using separately exported ONNX graphs
and locally GPU-specific engine caches. Engine caches should be rebuilt on the
deployment GPU rather than copied between unlike GPUs or TensorRT versions.

To vendor the pinned TensorRT development/runtime files and build support:

```bash
./scripts/setup.sh --backend cuda --trt
```

This is a large optional download. ONNX/FP8 generation and cache configuration
are documented in `docs/tensorrt.md`. A plain `setup.sh` deliberately keeps
TensorRT off so a new deployment is immediately usable with GGML.

## Verification

```bash
./scripts/verify_setup.sh --backend auto
ctest --test-dir build --output-on-failure
./build/examples/sam3cpp_segment_image \
  --model resources/models/sam3-q8_0.ggml \
  --image tests/data/cat.jpg --text cat
```

Applications embedding the source tree continue to use
`add_subdirectory(path/to/sam3cpplib)` and link `sam3cpp::sam3cpp`.
