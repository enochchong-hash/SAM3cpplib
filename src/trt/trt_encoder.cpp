// sam3cpplib -- ported from tools/sam3.cpp @ 049884b (wip/trt-phase1,
// upstream PABannier/sam3.cpp + gemma4 patches 0001-0013). See docs/PLAN.md.
#include "sam3_internal.h"
#ifdef SAM3_TRT_ENCODER
#include "trt_engine.h"
#include "trt_runtime.h"

/*****************************************************************************
** Image backbone — TensorRT path (see docs/sam3/PLAN.md, Phase 1)
**
** Opt-in via SAM3_TRT_ENCODER=1 (+ SAM3_TRT_ONNX_PATH). ggml remains the
** default/fallback path -- unset, this whole section costs one getenv() call.
*****************************************************************************/

struct sam3_trt_stderr_logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            fprintf(stderr, "[TRT] %s\n", msg);
        }
    }
};

// ── Programmatic configuration (sam3_params::trt) ──────────────────────────
// Captured once at sam3_load_model time. Fields are consulted only when
// cfg.enabled is set; otherwise (and for any empty field) the SAM3_TRT_* env
// vars remain authoritative, so env-driven deployments (release/sam3's
// trt_env.sh) behave exactly as before this struct existed.
static sam3_trt_config g_trt_cfg;

void sam3_trt_set_config(const sam3_trt_config& cfg) {
    g_trt_cfg = cfg;
}

bool sam3_trt_enabled() {
    return g_trt_cfg.enabled || getenv("SAM3_TRT_ENCODER") != nullptr;
}

// env-var name -> effective value ("" = unset). The single cfg.cache_dir
// root maps onto the three per-engine cache env vars via subdirectories.
std::string sam3_trt_cfg_value(const char* env_name) {
    if (g_trt_cfg.enabled) {
        const std::string n = env_name;
        if (n == "SAM3_TRT_ONNX_PATH"     && !g_trt_cfg.encoder_onnx.empty())     return g_trt_cfg.encoder_onnx;
        if (n == "SAM3_TRT_ONNX_PATH_FP8" && !g_trt_cfg.encoder_onnx_fp8.empty()) return g_trt_cfg.encoder_onnx_fp8;
        if (n == "SAM3_TRT_PCS_ONNX_PATH" && !g_trt_cfg.pcs_onnx.empty())         return g_trt_cfg.pcs_onnx;
        if (n == "SAM3_TRT_PVS_ONNX_PATH" && !g_trt_cfg.pvs_onnx.empty())         return g_trt_cfg.pvs_onnx;
        if (!g_trt_cfg.cache_dir.empty()) {
            if (n == "SAM3_TRT_CACHE_DIR")     return g_trt_cfg.cache_dir + "/encoder";
            if (n == "SAM3_TRT_PCS_CACHE_DIR") return g_trt_cfg.cache_dir + "/pcs";
            if (n == "SAM3_TRT_PVS_CACHE_DIR") return g_trt_cfg.cache_dir + "/pvs";
        }
        if (n == "SAM3_TRT_PCS_PRECISION" && !g_trt_cfg.pcs_precision.empty())    return g_trt_cfg.pcs_precision;
    }
    const char* v = getenv(env_name);
    return v ? std::string(v) : std::string();
}

// Lazily built/cached once per process, per distinct ONNX-path env var (one
// GPU model per process is already this codebase's own documented constraint
// -- see g_gpu_backend above -- but the image encoder / PCS / PVS engines
// are 3 distinct graphs sharing this same cache-by-env-var-name helper).
sam3_trt_engine* sam3_get_trt_engine_cached(const char* onnx_env_name, const char* cache_env_name,
                                                   bool allow_fp16,
                                                   const std::vector<std::string>& fp32_name_substrings) {
    static std::map<std::string, sam3_trt_engine*> cache;
    auto it = cache.find(onnx_env_name);
    if (it != cache.end()) return it->second;

    sam3_trt_engine* result = nullptr;
    const std::string onnx_path = sam3_trt_cfg_value(onnx_env_name);
    if (onnx_path.empty()) {
        fprintf(stderr, "%s: %s is unset (env or sam3_trt_config)\n", __func__, onnx_env_name);
    } else {
        const std::string cache_dir_cfg = sam3_trt_cfg_value(cache_env_name);
        const std::string cache_dir = !cache_dir_cfg.empty() ? cache_dir_cfg : (onnx_path + ".cache");
        static sam3_trt_stderr_logger logger;
        std::vector<char> engine_blob;
        if (sam3_trt_build_or_load_engine(onnx_path, cache_dir, logger, engine_blob, allow_fp16,
                                          fp32_name_substrings)) {
            result = sam3_trt_engine_create(engine_blob, logger);
            if (!result) {
                fprintf(stderr, "%s: sam3_trt_engine_create failed for %s\n", __func__, onnx_env_name);
            }
        }
    }
    cache[onnx_env_name] = result;  // cache the failure too, so we don't retry every call
    return result;
}

// Attempts the TensorRT image-encoder path on already-preprocessed CHW data.
// Returns false (no state mutated) if TRT isn't usable for this call, so the
// caller falls through to the unchanged ggml path -- never a hard failure.
bool sam3_try_trt_encode_image(sam3_state& state, const sam3_model& model,
                                      const float* chw_data, int img_size) {
    if (!sam3_trt_enabled()) return false;

    const auto& hp = model.hparams;
    if (img_size != hp.img_size) {
        // Phase 1's exported engine is built for one fixed, static input shape
        // (see convert_sam3_encoder_to_onnx.py) -- a resolution override falls
        // back to ggml rather than feeding TensorRT a shape it can't handle.
        fprintf(stderr, "%s: img_size=%d != model native %d, falling back to ggml\n",
                __func__, img_size, hp.img_size);
        return false;
    }

    // Direct GPU-to-GPU chaining requires state's tensors to actually be
    // CUDA-backend (model.backend could be CPU if params.use_gpu was false
    // for this particular load, independent of whether SAM3_TRT_ENCODER is
    // set -- TensorRT/CUDA init in sam3_get_trt_engine_cached doesn't care
    // about ggml's own backend choice). Binding a host pointer as a
    // TensorRT tensor address would silently corrupt memory, so require the
    // GPU backend and fall back to ggml (host round-trip, still correct)
    // otherwise.
    if (model.backend != g_gpu_backend || !g_gpu_backend) {
        fprintf(stderr, "%s: model not on the CUDA backend, falling back to ggml\n", __func__);
        return false;
    }

    sam3_trt_engine* enc = nullptr;
    if (state.trt_encoder_fp8) {
        enc = sam3_get_trt_engine_cached("SAM3_TRT_ONNX_PATH_FP8", "SAM3_TRT_CACHE_DIR");
        if (!enc) {
            fprintf(stderr, "%s: FP8 encoder requested but unavailable "
                            "(SAM3_TRT_ONNX_PATH_FP8 unset or engine failed) -- using FP16\n",
                    __func__);
        }
    }
    if (!enc) enc = sam3_get_trt_engine_cached("SAM3_TRT_ONNX_PATH", "SAM3_TRT_CACHE_DIR");
    if (!enc) return false;

    const int E = hp.vit_embed_dim;
    const int H = hp.n_img_embd();
    const int D = hp.neck_dim;
    const int hw[4] = {H * 4, H * 2, H, H / 2};

    // Drop any previous encode's allocations -- both a leftover TRT-path
    // buffer and a leftover ggml-graph-path buffer (whichever ran last time).
    if (state.trt_out_buf) { ggml_backend_buffer_free(state.trt_out_buf); state.trt_out_buf = nullptr; }
    if (state.trt_out_ctx) { ggml_free(state.trt_out_ctx); state.trt_out_ctx = nullptr; }
    if (state.galloc) { ggml_gallocr_free(state.galloc); state.galloc = nullptr; }
    if (state.ctx) { ggml_free(state.ctx); state.ctx = nullptr; }

    const size_t ctx_size = ggml_tensor_overhead() * 16;
    struct ggml_init_params gparams = {ctx_size, nullptr, true};
    state.trt_out_ctx = ggml_init(gparams);
    if (!state.trt_out_ctx) {
        fprintf(stderr, "%s: ggml_init failed\n", __func__);
        return false;
    }

    // ne=[E,W,H,1] / [D,W,H,1] -- fastest-varying axis first (E or D), same
    // convention sam3_build_vit_graph/sam3_build_neck_graph document, and
    // byte-identical to the ONNX graph's NHWC output layout (reversed(ne) ==
    // numpy shape), so binding the engine's output straight to ->data below
    // needs no reshuffling.
    auto* vit_t = ggml_new_tensor_4d(state.trt_out_ctx, GGML_TYPE_F32, E, H, H, 1);
    ggml_set_name(vit_t, "vit_output");
    struct ggml_tensor* neck_det_t[4] = {};
    struct ggml_tensor* neck_trk_t[4] = {};
    for (int i = 0; i < 4; ++i) {
        char name[32];
        snprintf(name, sizeof(name), "neck_det_%d", i);
        neck_det_t[i] = ggml_new_tensor_4d(state.trt_out_ctx, GGML_TYPE_F32, D, hw[i], hw[i], 1);
        ggml_set_name(neck_det_t[i], name);
        snprintf(name, sizeof(name), "neck_trk_%d", i);
        neck_trk_t[i] = ggml_new_tensor_4d(state.trt_out_ctx, GGML_TYPE_F32, D, hw[i], hw[i], 1);
        ggml_set_name(neck_trk_t[i], name);
    }

    state.trt_out_buf = ggml_backend_alloc_ctx_tensors(state.trt_out_ctx, model.backend);
    if (!state.trt_out_buf) {
        fprintf(stderr, "%s: ggml_backend_alloc_ctx_tensors failed\n", __func__);
        ggml_free(state.trt_out_ctx);
        state.trt_out_ctx = nullptr;
        return false;
    }

    sam3_trt_tensor image_in;
    image_in.name = "image";
    image_in.f32.assign(chw_data, chw_data + 3 * (size_t)img_size * img_size);

    // Bind the engine's outputs directly to these tensors' device buffers --
    // the encoder writes straight into state.vit_output/neck_det/neck_trk,
    // no device->host->device round trip (see sam3_trt_runtime.h's
    // device_ptr doc).
    std::vector<sam3_trt_tensor> trt_outputs;
    trt_outputs.push_back({"vit_output", {}, {}, vit_t->data});
    for (int i = 0; i < 4; ++i) {
        trt_outputs.push_back({"neck_det_" + std::to_string(i), {}, {}, neck_det_t[i]->data});
        trt_outputs.push_back({"neck_trk_" + std::to_string(i), {}, {}, neck_trk_t[i]->data});
    }

    if (!sam3_trt_engine_run(enc, {image_in}, trt_outputs)) {
        fprintf(stderr, "%s: sam3_trt_engine_run failed, falling back to ggml\n", __func__);
        ggml_backend_buffer_free(state.trt_out_buf); state.trt_out_buf = nullptr;
        ggml_free(state.trt_out_ctx); state.trt_out_ctx = nullptr;
        return false;
    }
    // sam3_trt_engine_run only returns one entry per *real* output binding
    // the engine declares -- a preset name with no matching binding simply
    // never appears in trt_outputs (not "present with a null device_ptr").
    // Check the count, not for nulls: if the engine doesn't declare all 9
    // expected outputs, the corresponding ggml tensor was never written and
    // holds uninitialized device memory. Fail loudly rather than proceed
    // silently.
    if (trt_outputs.size() != 9) {
        fprintf(stderr, "%s: engine produced %zu outputs, expected 9 (vit_output + 4 neck_det + "
                        "4 neck_trk), falling back to ggml\n", __func__, trt_outputs.size());
        ggml_backend_buffer_free(state.trt_out_buf); state.trt_out_buf = nullptr;
        ggml_free(state.trt_out_ctx); state.trt_out_ctx = nullptr;
        return false;
    }

    state.vit_output = vit_t;
    for (int i = 0; i < 4; ++i) {
        state.neck_det[i] = neck_det_t[i];
        state.neck_trk[i] = neck_trk_t[i];
    }
    state.backend = model.backend;
    return true;
}
#endif  // SAM3_TRT_ENCODER
