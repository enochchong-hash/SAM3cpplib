// TensorRT engine build/cache for the SAM3 image encoder (ViT + SimpleFPN
// neck). Part of the TensorRT image-encoder migration (see docs/sam3/PLAN.md,
// Phase 1). Builds a serialized engine from the ONNX graph produced by
// convert_sam3_encoder_to_onnx.py (an offline, one-time conversion tool --
// this header/its .cpp are the only TensorRT-dependent code that ships in
// the deployed server) and caches it to disk, keyed by ONNX-file content
// hash + GPU name + TensorRT version, so the (expensive) build only runs
// once per (model, GPU, TensorRT version) combination.
//
// Compiled only when SAM3_TRT_ENCODER is defined (see CMakeLists.txt).
#pragma once

#include <string>
#include <vector>

#include <NvInferRuntime.h>

// Builds (or loads a cached copy of) a serialized TensorRT engine for the
// ONNX model at `onnx_path`. `cache_dir` is created if missing. Force a
// rebuild (ignoring any cache hit) by setting SAM3_TRT_REBUILD=1.
//
// `allow_fp16`: if true (the default), FP16 tactics are permitted (fastest,
// and validated as numerically fine for the image encoder and PVS -- cosine
// similarity 0.994-0.999 / near-exact box+mask match on a real image).
// SAM3_TRT_FP32=1 forces this off process-wide, for debugging.
//
// `fp32_name_substrings`: mixed-precision override -- any network layer
// whose ONNX-derived name contains one of these substrings is pinned to
// FP32 (via ILayer::setPrecision + BuilderFlag::kPREFER_PRECISION_CONSTRAINTS,
// "prefer" so an unsupported combination falls back rather than failing the
// build) while the rest of the network still runs FP16. Use this instead of
// `allow_fp16=false` when only specific numerically-delicate sections need
// full precision -- e.g. PCS's box-relative-positional-bias log-distance
// transform (sign(d)*log(...), where FP16 rounding near d=0 can flip the
// sign term) and its 6-iteration inverse-sigmoid/sigmoid box-refinement
// recurrence, both found to be more FP16-sensitive than the bulk of the
// network on a real-image test (0.96 vs 0.76 detection confidence with the
// whole graph in FP16, box/mask otherwise matching closely).
bool sam3_trt_build_or_load_engine(const std::string& onnx_path,
                                    const std::string& cache_dir,
                                    nvinfer1::ILogger& logger,
                                    std::vector<char>& out_engine_blob,
                                    bool allow_fp16 = true,
                                    const std::vector<std::string>& fp32_name_substrings = {});
