// Generic TensorRT execution wrapper -- shared by the image encoder, PCS
// (text head), and PVS (point head) TensorRT paths (see docs/sam3/PLAN.md).
// Compiled only when SAM3_TRT_ENCODER is defined.
//
// Each engine (built from a distinct ONNX graph) has its own I/O tensor
// names/shapes/dtypes -- rather than hardcoding those per use site, this
// wrapper discovers them from the engine itself at creation time and binds
// device buffers generically by name.
#pragma once

#include <string>
#include <vector>

#include <NvInferRuntime.h>

struct sam3_trt_engine;

// A named tensor buffer. Exactly one of f32/i32 should be populated,
// matching the engine's declared dtype for that tensor name (checked at
// create time against the actual ONNX-declared type; mismatches fail loudly
// rather than silently misinterpreting bytes).
//
// `device_ptr`, if set, bypasses the host round-trip entirely -- direct
// GPU-to-GPU chaining between engines (or between an engine and a ggml CUDA
// tensor's ->data), instead of downloading to a host std::vector and
// re-uploading:
//   - As an input: the pointer must already hold valid, fully-written
//     device data of the right dtype/element-count by the time
//     sam3_trt_engine_run is called (f32/i32 are ignored). The caller is
//     responsible for shape/dtype matching -- this is internal plumbing
//     between our own call sites, not a user-facing boundary.
//   - As an output: pre-populate `outputs` (before calling) with an entry
//     whose name matches a real output tensor and device_ptr set to a
//     buffer the caller owns; the engine writes its result directly there
//     and sam3_trt_engine_run leaves it untouched (no download). Any output
//     not pre-populated this way is downloaded to host f32/i32 as usual.
struct sam3_trt_tensor {
    std::string name;
    std::vector<float> f32;
    std::vector<int32_t> i32;
    void* device_ptr = nullptr;
    // Concrete shape for an input whose engine-declared shape has dynamic
    // (-1) dims (e.g. the PVS sparse-token count). Required for such inputs
    // (sam3_trt_engine_run calls setInputShape with it and validates the
    // element count); leave empty for static inputs.
    std::vector<int64_t> dims;
};

// Deserializes `engine_blob` (from sam3_trt_build_or_load_engine) and
// discovers every I/O tensor the engine declares. Staging device buffers
// are NOT allocated here -- sam3_trt_engine_run allocates them lazily, and
// only for tensors actually used through the host path (a tensor always
// bound via device_ptr never allocates one). Returns nullptr on failure.
sam3_trt_engine* sam3_trt_engine_create(const std::vector<char>& engine_blob,
                                        nvinfer1::ILogger& logger);

// Runs the engine once. `inputs` must include every input tensor the engine
// declares (extras are ignored). `outputs` is replaced with one entry per
// output tensor the engine declares -- pre-populate entries with device_ptr
// set to request direct GPU-to-GPU binding for specific outputs (see
// sam3_trt_tensor's doc above); anything not pre-populated is downloaded to
// host as usual. Returns false on a real execution failure (never partially
// fills `outputs`).
bool sam3_trt_engine_run(sam3_trt_engine* eng,
                         const std::vector<sam3_trt_tensor>& inputs,
                         std::vector<sam3_trt_tensor>& outputs);

void sam3_trt_engine_free(sam3_trt_engine* eng);
