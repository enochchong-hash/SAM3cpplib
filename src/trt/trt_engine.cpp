#include "trt_engine.h"

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>

#include <sys/stat.h>

namespace {

// Deleter for TensorRT interface objects (all derive from INoCopy, which
// declares a virtual destructor -- safe to `delete` directly, no destroy()
// method needed since TensorRT 8.0's API cleanup).
struct sam3_trt_deleter {
    template <typename T>
    void operator()(T* p) const { delete p; }
};
template <typename T>
using sam3_trt_ptr = std::unique_ptr<T, sam3_trt_deleter>;

uint64_t sam3_fnv1a(const void* data, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

bool sam3_read_file(const std::string& path, std::vector<char>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    std::streamsize sz = f.tellg();
    if (sz < 0) return false;
    f.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(sz));
    if (sz > 0 && !f.read(out.data(), sz)) return false;
    return true;
}

bool sam3_write_file(const std::string& path, const void* data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    if (n > 0) f.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
    return f.good();
}

}  // namespace

bool sam3_trt_build_or_load_engine(const std::string& onnx_path,
                                    const std::string& cache_dir,
                                    nvinfer1::ILogger& logger,
                                    std::vector<char>& out_engine_blob,
                                    bool allow_fp16,
                                    const std::vector<std::string>& fp32_name_substrings) {
    if (getenv("SAM3_TRT_FP32")) allow_fp16 = false;

    std::vector<char> onnx_bytes;
    if (!sam3_read_file(onnx_path, onnx_bytes)) {
        fprintf(stderr, "%s: failed to read '%s'\n", __func__, onnx_path.c_str());
        return false;
    }
    // Cache key includes the FP16/FP32 choice and the mixed-precision
    // override list -- an engine built one way must never be silently
    // reused for a differently-configured build.
    uint64_t hash = sam3_fnv1a(onnx_bytes.data(), onnx_bytes.size()) ^ (allow_fp16 ? 0 : 0x9E3779B97F4A7C15ULL);
    for (const auto& s : fp32_name_substrings) {
        hash ^= sam3_fnv1a(s.data(), s.size());
    }

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    std::string gpu_name = prop.name;
    for (char& c : gpu_name) {
        if (c == ' ' || c == '/') c = '_';
    }

    char cache_name[1024];
    snprintf(cache_name, sizeof(cache_name),
             "%s/sam3_encoder-%s-trt%d.%d.%d.%d-%016llx.engine",
             cache_dir.c_str(), gpu_name.c_str(),
             NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, NV_TENSORRT_BUILD,
             static_cast<unsigned long long>(hash));

    const bool force_rebuild = getenv("SAM3_TRT_REBUILD") != nullptr;
    if (!force_rebuild && sam3_read_file(cache_name, out_engine_blob)) {
        fprintf(stderr, "%s: loaded cached engine '%s' (%zu bytes)\n",
                __func__, cache_name, out_engine_blob.size());
        return true;
    }

    fprintf(stderr, "%s: building engine from '%s' (%s, %s)...\n", __func__, onnx_path.c_str(),
            force_rebuild ? "forced rebuild" : "no cache hit", allow_fp16 ? "FP16" : "FP32");

    sam3_trt_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) {
        fprintf(stderr, "%s: createInferBuilder failed\n", __func__);
        return false;
    }
    // Explicit-batch is the only mode in TensorRT 10.x -- flags=0 is
    // equivalent to (deprecated) NetworkDefinitionCreationFlag::kEXPLICIT_BATCH.
    sam3_trt_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    if (!network) {
        fprintf(stderr, "%s: createNetworkV2 failed\n", __func__);
        return false;
    }
    sam3_trt_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser) {
        fprintf(stderr, "%s: nvonnxparser::createParser failed\n", __func__);
        return false;
    }
    if (!parser->parseFromFile(onnx_path.c_str(),
                                static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        fprintf(stderr, "%s: ONNX parse failed (%d errors):\n", __func__, parser->getNbErrors());
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            fprintf(stderr, "  %s\n", parser->getError(i)->desc());
        }
        return false;
    }

    sam3_trt_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) {
        fprintf(stderr, "%s: createBuilderConfig failed\n", __func__);
        return false;
    }
    // Dynamic input dims (e.g. the PVS graph's sparse-token count) need an
    // optimization profile. Every -1 dim on every input gets the same
    // [1, 2, 16] (min, opt, max) range -- the only dynamic dim in any of our
    // graphs is the PVS sparse-token count, where 2 (one point + pad) is the
    // most common request shape and 16 comfortably covers realistic
    // multi-point/box prompts. Static graphs (encoder, PCS) have no dynamic
    // dims and skip this entirely, keeping their behavior unchanged.
    {
        bool have_dynamic = false;
        nvinfer1::IOptimizationProfile* profile = nullptr;
        for (int i = 0; i < network->getNbInputs(); ++i) {
            nvinfer1::ITensor* input = network->getInput(i);
            nvinfer1::Dims dims = input->getDimensions();
            bool dyn = false;
            for (int d = 0; d < dims.nbDims; ++d) dyn = dyn || (dims.d[d] == -1);
            if (!dyn) continue;
            if (!profile) profile = builder->createOptimizationProfile();
            nvinfer1::Dims mn = dims, op = dims, mx = dims;
            for (int d = 0; d < dims.nbDims; ++d) {
                if (dims.d[d] == -1) { mn.d[d] = 1; op.d[d] = 2; mx.d[d] = 16; }
            }
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, mn);
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, op);
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, mx);
            have_dynamic = true;
            fprintf(stderr, "%s: dynamic input '%s' -> profile min/opt/max = 1/2/16 on dynamic dims\n",
                    __func__, input->getName());
        }
        if (have_dynamic) {
            config->addOptimizationProfile(profile);
        }
    }
    // FP16 tactics allowed when requested; the ONNX graph itself is authored
    // in plain F32 (see convert_sam3_*_to_onnx.py) -- TensorRT casts as needed.
    if (allow_fp16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    if (allow_fp16 && !fp32_name_substrings.empty()) {
        int n_pinned = 0;
        const int n_layers = network->getNbLayers();
        for (int i = 0; i < n_layers; ++i) {
            nvinfer1::ILayer* layer = network->getLayer(i);
            const char* name = layer->getName();
            if (!name) continue;
            // Index/shape constants (e.g. Slice's starts/ends/axes/steps,
            // Reshape's target shape) are INT32/INT64/BOOL -- forcing FP32
            // precision on those is a TensorRT API error, not just a
            // precision no-op, so only pin layers whose outputs are all
            // float-typed.
            bool all_float = layer->getNbOutputs() > 0;
            for (int o = 0; o < layer->getNbOutputs(); ++o) {
                if (layer->getOutput(o)->getType() != nvinfer1::DataType::kFLOAT) {
                    all_float = false;
                    break;
                }
            }
            if (!all_float) continue;
            for (const auto& pat : fp32_name_substrings) {
                if (strstr(name, pat.c_str())) {
                    layer->setPrecision(nvinfer1::DataType::kFLOAT);
                    ++n_pinned;
                    break;
                }
            }
        }
        // "Prefer" (not "obey") -- fall back to FP16 for a layer/op
        // combination that can't actually run in FP32, rather than failing
        // the whole build.
        config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
        fprintf(stderr, "%s: pinned %d/%d layers to FP32 (mixed precision)\n",
                __func__, n_pinned, n_layers);
    }
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 2ULL << 30);  // 2 GiB

    sam3_trt_ptr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
        fprintf(stderr, "%s: buildSerializedNetwork failed\n", __func__);
        return false;
    }

    out_engine_blob.assign(static_cast<const char*>(serialized->data()),
                           static_cast<const char*>(serialized->data()) + serialized->size());

    mkdir(cache_dir.c_str(), 0755);
    if (!sam3_write_file(cache_name, out_engine_blob.data(), out_engine_blob.size())) {
        fprintf(stderr, "%s: warning: failed to write cache file '%s' (continuing without cache)\n",
                __func__, cache_name);
    } else {
        fprintf(stderr, "%s: built and cached engine '%s' (%zu bytes)\n",
                __func__, cache_name, out_engine_blob.size());
    }
    return true;
}
