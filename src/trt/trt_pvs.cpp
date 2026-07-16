// sam3cpplib -- ported from tools/sam3.cpp @ 049884b (wip/trt-phase1,
// upstream PABannier/sam3.cpp + gemma4 patches 0001-0013). See docs/PLAN.md.
#include "sam3_internal.h"
#ifdef SAM3_TRT_ENCODER
#include "trt_engine.h"
#include "trt_runtime.h"

/*****************************************************************************
** Image segmentation — PVS TensorRT path (see docs/sam3/PLAN.md)
**
** Opt-in via SAM3_TRT_ENCODER=1 (+ SAM3_TRT_PVS_ONNX_PATH). Handles every
** prompt shape sam3_collect_pvs_prompt_tokens produces -- any mix of
** positive/negative points and a box (corner tokens, labels 2/3) -- via the
** engine's dynamic sparse-token dim (profile 1..16; more tokens than that
** falls back to ggml, as does any TRT failure). Reuses the exact same
** per-token sparse-embedding CPU math (sam3_pe_encode_coord + state's PE
** caches) and the same post-processing (sam3_bilinear_interpolate, mask
** binarization, bbox-from-mask) the ggml path uses -- only the
** two-way-decoder compute itself moves to TensorRT.
*****************************************************************************/
bool sam3_try_trt_segment_pvs(sam3_state& state, const sam3_model& model,
                                     const sam3_pvs_params& params,
                                     sam3_result& out_result) {
    if (!sam3_trt_enabled()) return false;

    // state.neck_trk[i]->data is read directly as a CUDA device pointer
    // below (no ggml_backend_tensor_get round trip) -- only safe if the
    // image was actually encoded on the GPU backend.
    if (model.backend != g_gpu_backend || !g_gpu_backend) {
        fprintf(stderr, "%s: model not on the CUDA backend, falling back to ggml\n", __func__);
        return false;
    }

    const auto& hp = model.hparams;
    const int D = hp.sam_embed_dim;
    const int H = sam3_eff_feat_size(state, hp);
    const int eff_img_size = sam3_eff_img_size(state, hp);
    const int num_mask_tokens = hp.sam_n_multimask + 1;
    const int H0 = H * 4, H1 = H * 2;
    const int mask_hw = H0;

    sam3_populate_pe_cache(state, model);

    // Every prompt shape (any mix of positive/negative points and a box)
    // reduces to a flat token list -- box corners are just tokens with
    // labels 2/3, exactly as the ggml path builds them. The engine's
    // sparse-token dim is dynamic (optimization profile 1..16, see
    // sam3_trt_build_or_load_engine), so the only genuine scope limit left
    // is the profile maximum.
    std::vector<float> all_coords;
    std::vector<int> all_labels;
    sam3_collect_pvs_prompt_tokens(params, all_coords, all_labels);
    const int N_pts = (int)all_labels.size();
    if (N_pts > 16) {
        fprintf(stderr, "%s: %d prompt tokens exceeds the engine profile max (16), "
                        "falling back to ggml\n", __func__, N_pts);
        return false;
    }

    // PVS runs FP32: single-positive-point prompts were fine in FP16, but
    // negative-point prompts diverged badly from the ggml reference (mask
    // area 27% off on a real image -- negatives create large near-zero-logit
    // regions where FP16 noise flips many pixels at the 0 threshold); FP32
    // matches ggml to <2% everywhere. Unlike PCS, FP32 costs almost nothing
    // here: this decoder has no big attention (<=16 tokens x 5184 image), so
    // it's ~120MB ctx and ~7.4ms vs ~7ms warm.
    sam3_trt_engine* eng = sam3_get_trt_engine_cached("SAM3_TRT_PVS_ONNX_PATH", "SAM3_TRT_PVS_CACHE_DIR",
                                                      /*allow_fp16=*/false);
    if (!eng) return false;

    sam3_trt_tensor sparse_in; sparse_in.name = "sparse"; sparse_in.f32.assign(N_pts * D, 0.0f);
    sparse_in.dims = {1, N_pts, D};
    const int num_pos_feats = D / 2;
    for (int p = 0; p < N_pts; ++p) {
        float px = all_coords[p * 2 + 0] / (float)state.orig_width * (float)eff_img_size + 0.5f;
        float py = all_coords[p * 2 + 1] / (float)state.orig_height * (float)eff_img_size + 0.5f;
        float x_norm = px / (float)eff_img_size;
        float y_norm = py / (float)eff_img_size;
        float pe_vec[256];
        sam3_pe_encode_coord(pe_vec, x_norm, y_norm, state.pe_gauss_cache.data(), num_pos_feats);
        int label = all_labels[p];
        if (label == -1) {
            for (int d = 0; d < D; ++d) sparse_in.f32[p * D + d] = state.not_a_point_cache[d];
        } else {
            for (int d = 0; d < D; ++d) sparse_in.f32[p * D + d] = pe_vec[d] + state.point_emb_cache[label][d];
        }
    }

    // image_feats fed RAW (no_mem_embed is added inside the graph, unlike the
    // ggml path which adds it on the CPU before upload -- see
    // convert_sam3_pvs_to_onnx.py's baked no_mem_embed_const). Read straight
    // from the encoder's device buffers -- no device->host->device round
    // trip (see sam3_trt_runtime.h's device_ptr doc).
    sam3_trt_tensor image_feats_in; image_feats_in.name = "image_feats";
    image_feats_in.device_ptr = state.neck_trk[2]->data;
    sam3_trt_tensor feat_s0_in; feat_s0_in.name = "feat_s0"; feat_s0_in.device_ptr = state.neck_trk[0]->data;
    sam3_trt_tensor feat_s1_in; feat_s1_in.name = "feat_s1"; feat_s1_in.device_ptr = state.neck_trk[1]->data;

    std::vector<sam3_trt_tensor> outputs;
    if (!sam3_trt_engine_run(eng, {sparse_in, image_feats_in, feat_s0_in, feat_s1_in}, outputs)) {
        fprintf(stderr, "%s: sam3_trt_engine_run failed, falling back to ggml\n", __func__);
        return false;
    }
    auto find_out = [&](const char* name) -> std::vector<float>* {
        for (auto& t : outputs) if (t.name == name) return &t.f32;
        return nullptr;
    };
    auto* masks_data = find_out("masks");
    auto* iou_data = find_out("iou_pred");
    auto* obj_score_out = find_out("obj_score_logit");
    auto* sam_token_data = find_out("sam_token");
    if (!masks_data || !iou_data || !obj_score_out || !sam_token_data) {
        fprintf(stderr, "%s: engine missing expected outputs, falling back to ggml\n", __func__);
        return false;
    }
    const float obj_logit = (*obj_score_out)[0];
    const float obj_score = 1.0f / (1.0f + expf(-obj_logit));

    static const bool stage_timing_pvs = getenv("SAM3_STAGE_TIMING") != nullptr;
    auto t_post0 = std::chrono::high_resolution_clock::now();
    int start_idx, end_idx;
    if (params.multimask) { start_idx = 1; end_idx = num_mask_tokens; }
    else                  { start_idx = 0; end_idx = 1; }

    for (int m = start_idx; m < end_idx; ++m) {
        sam3_detection det;
        det.sam_token = *sam_token_data;

        const float* mask_ptr = masks_data->data() + m * mask_hw * mask_hw;
        auto mask_resized = sam3_bilinear_interpolate(mask_ptr, mask_hw, mask_hw,
                                                      state.orig_width, state.orig_height);
        det.mask.width = state.orig_width;
        det.mask.height = state.orig_height;
        det.mask.data.resize(state.orig_width * state.orig_height);
        for (int i = 0; i < (int)mask_resized.size(); ++i)
            det.mask.data[i] = (mask_resized[i] > 0.0f) ? 255 : 0;

        det.mask.iou_score = (*iou_data)[m];
        det.mask.obj_score = obj_score;
        det.mask.instance_id = m;
        det.score = (*iou_data)[m];
        det.iou_score = (*iou_data)[m];
        det.instance_id = m;

        int min_x = state.orig_width, min_y = state.orig_height, max_x = 0, max_y = 0;
        for (int y = 0; y < state.orig_height; ++y) {
            for (int x = 0; x < state.orig_width; ++x) {
                if (det.mask.data[y * state.orig_width + x] > 0) {
                    min_x = std::min(min_x, x); min_y = std::min(min_y, y);
                    max_x = std::max(max_x, x); max_y = std::max(max_y, y);
                }
            }
        }
        det.box = {(float)min_x, (float)min_y, (float)max_x, (float)max_y};

        out_result.detections.push_back(std::move(det));
    }
    if (stage_timing_pvs) {
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_post0).count();
        fprintf(stderr, "[stage_timing] pvs_post (CPU mask upscale + binarize + bbox scan, "
                        "%zu masks): %.2f ms\n", out_result.detections.size(), ms);
    }
    return true;
}
#endif  // SAM3_TRT_ENCODER
