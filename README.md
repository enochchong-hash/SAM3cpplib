# sam3cpplib

First-party C++ library for **SAM3 single-image promptable segmentation** — text,
exemplar-box, point, and box prompts — with three backends:

- **TensorRT** (FP16 default / FP8 opt-in): primary NVIDIA fast path
- **ggml CUDA**: portable GPU path
- **ggml CPU**: worst-case fallback and the golden-sample reference

Raw RGB in, raw 0/255 masks out — no image codecs, no HTTP, no Python at runtime.
Designed to be embedded in a larger application as a git submodule.

This library supersedes the `release/sam3` submodule-plus-patches arrangement; the code
is owned here (originally derived from MIT-licensed `PABannier/sam3.cpp`, see LICENSE).

**Status: scaffold.** See [docs/PLAN.md](docs/PLAN.md) for the full structure, the
port mapping, and the phase gates.
