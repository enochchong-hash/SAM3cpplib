#include "trt_runtime.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>

namespace {

struct sam3_trt_deleter {
    template <typename T>
    void operator()(T* p) const { delete p; }
};
template <typename T>
using sam3_trt_ptr = std::unique_ptr<T, sam3_trt_deleter>;

int64_t sam3_dims_volume(const nvinfer1::Dims& d) {
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= d.d[i];
    return v;
}

bool sam3_cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "sam3_trt_runtime: %s failed: %s\n", what, cudaGetErrorString(err));
        return false;
    }
    return true;
}

}  // namespace

struct sam3_trt_engine {
    sam3_trt_ptr<nvinfer1::IRuntime> runtime;
    sam3_trt_ptr<nvinfer1::ICudaEngine> engine;
    sam3_trt_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream = nullptr;

    struct Binding {
        std::string name;
        void* dev = nullptr;
        int64_t elems = 0;      // static shapes: exact; dynamic: profile-max (staging bound)
        nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
        bool is_input = false;
        bool is_dynamic = false;  // any -1 in the engine-declared shape
    };
    std::vector<Binding> bindings;
};

sam3_trt_engine* sam3_trt_engine_create(const std::vector<char>& engine_blob,
                                        nvinfer1::ILogger& logger) {
    auto eng = std::make_unique<sam3_trt_engine>();

    eng->runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!eng->runtime) {
        fprintf(stderr, "%s: createInferRuntime failed\n", __func__);
        return nullptr;
    }
    eng->engine.reset(eng->runtime->deserializeCudaEngine(engine_blob.data(), engine_blob.size()));
    if (!eng->engine) {
        fprintf(stderr, "%s: deserializeCudaEngine failed\n", __func__);
        return nullptr;
    }
    eng->context.reset(eng->engine->createExecutionContext());
    if (!eng->context) {
        fprintf(stderr, "%s: createExecutionContext failed\n", __func__);
        return nullptr;
    }
    if (!sam3_cuda_check(cudaStreamCreate(&eng->stream), "cudaStreamCreate")) {
        return nullptr;
    }

    const int n_io = eng->engine->getNbIOTensors();
    for (int i = 0; i < n_io; ++i) {
        const char* name = eng->engine->getIOTensorName(i);
        sam3_trt_engine::Binding b;
        b.name = name;
        nvinfer1::Dims shape = eng->engine->getTensorShape(name);
        for (int d = 0; d < shape.nbDims; ++d) b.is_dynamic = b.is_dynamic || (shape.d[d] == -1);
        if (b.is_dynamic) {
            // Size any lazily-allocated staging to the optimization
            // profile's maximum, so one buffer serves every runtime shape.
            shape = eng->engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
        }
        b.elems = sam3_dims_volume(shape);
        b.dtype = eng->engine->getTensorDataType(name);
        b.is_input = eng->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;

        const size_t elem_size = (b.dtype == nvinfer1::DataType::kINT32 ||
                                  b.dtype == nvinfer1::DataType::kFLOAT) ? 4 : 0;
        if (elem_size == 0) {
            fprintf(stderr, "%s: tensor '%s' has unsupported dtype %d (only FLOAT/INT32 handled)\n",
                    __func__, name, static_cast<int>(b.dtype));
            return nullptr;
        }
        // No staging buffer allocated here -- sam3_trt_engine_run allocates
        // one lazily the first time this tensor is actually used through the
        // host path. Tensors always bound via device_ptr (the common case
        // for the chained encoder->PCS/PVS tensors: encoder outputs ~247MB,
        // PCS inputs ~111MB, PVS inputs ~111MB) never allocate staging at
        // all. Addresses are (re)bound every run() call, so nothing is set
        // on the context here either.
        eng->bindings.push_back(b);
    }

    return eng.release();
}

bool sam3_trt_engine_run(sam3_trt_engine* eng,
                         const std::vector<sam3_trt_tensor>& inputs,
                         std::vector<sam3_trt_tensor>& outputs) {
    if (!eng) return false;

    auto find_input = [&](const std::string& name) -> const sam3_trt_tensor* {
        for (auto& t : inputs) {
            if (t.name == name) return &t;
        }
        return nullptr;
    };
    // Callers may pre-populate `outputs` with entries carrying a device_ptr,
    // to request direct GPU-to-GPU binding for that output (skip the D2H
    // download) -- e.g. writing straight into a ggml CUDA tensor another
    // stage already owns, instead of bouncing through host memory.
    auto find_preset_output = [&](const std::string& name) -> const sam3_trt_tensor* {
        for (auto& t : outputs) {
            if (t.name == name && t.device_ptr) return &t;
        }
        return nullptr;
    };

    for (auto& b : eng->bindings) {
        if (!b.is_input) continue;
        const sam3_trt_tensor* src = find_input(b.name);
        if (!src) {
            fprintf(stderr, "%s: missing required input '%s'\n", __func__, b.name.c_str());
            return false;
        }
        // Dynamic-shape inputs need their concrete shape set every run (and
        // the caller must provide it). `want` is the element count actually
        // transferred this run -- for dynamic inputs that's the concrete
        // shape's volume, bounded by b.elems (the profile max, which sizes
        // the staging buffer).
        int64_t want = b.elems;
        if (b.is_dynamic) {
            if (src->dims.empty()) {
                fprintf(stderr, "%s: input '%s' has dynamic shape -- caller must set dims\n",
                        __func__, b.name.c_str());
                return false;
            }
            nvinfer1::Dims d{};
            d.nbDims = static_cast<int32_t>(src->dims.size());
            int64_t vol = 1;
            for (size_t i = 0; i < src->dims.size(); ++i) {
                d.d[i] = static_cast<int64_t>(src->dims[i]);
                vol *= src->dims[i];
            }
            if (vol > b.elems) {
                fprintf(stderr, "%s: input '%s' shape volume %lld exceeds profile max %lld\n",
                        __func__, b.name.c_str(), (long long)vol, (long long)b.elems);
                return false;
            }
            if (!eng->context->setInputShape(b.name.c_str(), d)) {
                fprintf(stderr, "%s: setInputShape('%s') rejected (outside profile range?)\n",
                        __func__, b.name.c_str());
                return false;
            }
            want = vol;
        }
        if (src->device_ptr) {
            // Direct GPU-to-GPU chaining -- see sam3_trt_tensor's doc.
            eng->context->setTensorAddress(b.name.c_str(), src->device_ptr);
            continue;
        }
        // Lazy staging: allocated on first host-path use at the max size
        // this binding can ever need, reused afterwards. FLOAT and INT32
        // (the only dtypes engine_create accepts) are both 4 bytes/element.
        if (!b.dev && !sam3_cuda_check(cudaMalloc(&b.dev, b.elems * 4), "cudaMalloc(staging in)")) {
            return false;
        }
        eng->context->setTensorAddress(b.name.c_str(), b.dev);
        if (b.dtype == nvinfer1::DataType::kFLOAT) {
            if (static_cast<int64_t>(src->f32.size()) != want) {
                fprintf(stderr, "%s: input '%s' size mismatch (got %zu, expected %lld floats)\n",
                        __func__, b.name.c_str(), src->f32.size(), (long long)want);
                return false;
            }
            if (!sam3_cuda_check(cudaMemcpyAsync(b.dev, src->f32.data(), want * sizeof(float),
                                                cudaMemcpyHostToDevice, eng->stream),
                                "cudaMemcpyAsync(H2D f32)")) {
                return false;
            }
        } else {  // kINT32
            if (static_cast<int64_t>(src->i32.size()) != want) {
                fprintf(stderr, "%s: input '%s' size mismatch (got %zu, expected %lld int32s)\n",
                        __func__, b.name.c_str(), src->i32.size(), (long long)want);
                return false;
            }
            if (!sam3_cuda_check(cudaMemcpyAsync(b.dev, src->i32.data(), want * sizeof(int32_t),
                                                cudaMemcpyHostToDevice, eng->stream),
                                "cudaMemcpyAsync(H2D i32)")) {
                return false;
            }
        }
    }

    // Bind output addresses before enqueue: either the caller's own device
    // pointer (direct chaining) or a lazily-allocated staging buffer for
    // outputs that will be downloaded to host below.
    for (auto& b : eng->bindings) {
        if (b.is_input) continue;
        const sam3_trt_tensor* preset = find_preset_output(b.name);
        if (!preset && !b.dev &&
            !sam3_cuda_check(cudaMalloc(&b.dev, b.elems * 4), "cudaMalloc(staging out)")) {
            return false;
        }
        eng->context->setTensorAddress(b.name.c_str(), preset ? preset->device_ptr : b.dev);
    }

    if (!eng->context->enqueueV3(eng->stream)) {
        fprintf(stderr, "%s: enqueueV3 failed\n", __func__);
        return false;
    }

    std::vector<sam3_trt_tensor> result;
    for (auto& b : eng->bindings) {
        if (b.is_input) continue;
        const sam3_trt_tensor* preset = find_preset_output(b.name);
        if (preset) {
            // Data already landed exactly where the caller wanted it (their
            // own device buffer) -- nothing to download, keep their entry.
            result.push_back(*preset);
            continue;
        }
        sam3_trt_tensor t;
        t.name = b.name;
        if (b.dtype == nvinfer1::DataType::kFLOAT) {
            t.f32.resize(static_cast<size_t>(b.elems));
            if (!sam3_cuda_check(cudaMemcpyAsync(t.f32.data(), b.dev, b.elems * sizeof(float),
                                                cudaMemcpyDeviceToHost, eng->stream),
                                "cudaMemcpyAsync(D2H f32)")) {
                return false;
            }
        } else {
            t.i32.resize(static_cast<size_t>(b.elems));
            if (!sam3_cuda_check(cudaMemcpyAsync(t.i32.data(), b.dev, b.elems * sizeof(int32_t),
                                                cudaMemcpyDeviceToHost, eng->stream),
                                "cudaMemcpyAsync(D2H i32)")) {
                return false;
            }
        }
        result.push_back(std::move(t));
    }

    // sam3_encode_image and friends assume a synchronous handoff -- every
    // output must be materialized before returning, since the caller may
    // immediately hand these buffers to ggml_backend_tensor_set.
    if (!sam3_cuda_check(cudaStreamSynchronize(eng->stream), "cudaStreamSynchronize")) {
        return false;
    }
    outputs = std::move(result);
    return true;
}

void sam3_trt_engine_free(sam3_trt_engine* eng) {
    if (!eng) return;
    for (auto& b : eng->bindings) {
        if (b.dev) cudaFree(b.dev);
    }
    if (eng->stream) cudaStreamDestroy(eng->stream);
    delete eng;
}
