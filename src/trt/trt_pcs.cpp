// sam3cpplib -- ported from tools/sam3.cpp @ 049884b (wip/trt-phase1,
// upstream PABannier/sam3.cpp + gemma4 patches 0001-0013). See docs/PLAN.md.
#include "sam3_internal.h"
#ifdef SAM3_TRT_ENCODER
#include "trt_engine.h"
#include "trt_runtime.h"

/*****************************************************************************
** Image segmentation — PCS TensorRT path (see docs/sam3/PLAN.md)
**
** Opt-in via SAM3_TRT_ENCODER=1 (+ SAM3_TRT_PCS_ONNX_PATH). SCOPE: text
** prompt with ZERO exemplar boxes only (see convert_sam3_pcs_to_onnx.py) --
** falls back to the ggml path (unmodified below) for anything else, or on
** any TRT failure. Reuses the exact same post-processing helpers
** (sam3_cxcywh_to_xyxy / sam3_bilinear_interpolate / sam3_nms) the ggml path
** uses, so behavior after the raw class_scores/pred_boxes/presence_score/
** mask_logits are obtained is identical regardless of which path produced them.
*****************************************************************************/
bool sam3_try_trt_segment_pcs(sam3_state& state, const sam3_model& model,
                                     const sam3_pcs_params& params,
                                     const std::vector<int32_t>& token_ids,
                                     sam3_result& out_result) {
    if (!sam3_trt_enabled()) return false;

    const int n_boxes = (int)(params.pos_exemplars.size() + params.neg_exemplars.size());
    const int n_geo = n_boxes + (int)params.exemplar_embeddings.size() + 1;
    if (n_geo > 16) {
        fprintf(stderr, "%s: %d geometry tokens exceeds the engine profile max (16), "
                        "falling back to ggml\n", __func__, n_geo);
        return false;
    }
    // state.neck_det[i]->data is read directly as a CUDA device pointer
    // below (no ggml_backend_tensor_get round trip) -- only safe if the
    // image was actually encoded on the GPU backend.
    if (model.backend != g_gpu_backend || !g_gpu_backend) {
        fprintf(stderr, "%s: model not on the CUDA backend, falling back to ggml\n", __func__);
        return false;
    }

    const auto& hp = model.hparams;
    const int D = hp.neck_dim;
    const int H = hp.n_img_embd();
    const int NQ = hp.ddec_num_queries;
    const int N_spatial = H * H;
    const int H0 = H * 4, H1 = H * 2;
    const int mask_hw = H0;

    // PCS precision, selectable via SAM3_TRT_PCS_PRECISION:
    //   unset / "fp32"          -- whole graph FP32 (default, safest)
    //   "fp16"                  -- whole graph FP16
    //   "mixed:<s1>,<s2>,..."   -- FP16 except layers whose ONNX-derived
    //                              names contain one of the substrings,
    //                              pinned FP32 (component prefixes: text_,
    //                              geom_, fenc_, ddec_, scoring, seg_, pxd_)
    // Background: full FP16 measurably degraded detection confidence on a
    // real image (0.96 vs 0.76 for the same box/mask) even though isolated
    // cosine similarity looked fine; full FP32 matches an onnxruntime FP32
    // reference to ~1e-4. An early mixed attempt pinning 756 *scattered*
    // layers (substring matches like "_rpb" across the whole graph) made
    // things worse -- 0 detections -- so mixed pinning should use whole
    // component prefixes (few large contiguous regions, few FP16<->FP32
    // reformat boundaries), which is what this knob is for.
    bool pcs_allow_fp16 = false;
    std::vector<std::string> pcs_fp32_patterns;
    const std::string pcs_prec = sam3_trt_cfg_value("SAM3_TRT_PCS_PRECISION");
    if (!pcs_prec.empty()) {
        const std::string& p = pcs_prec;
        if (p == "fp16") {
            pcs_allow_fp16 = true;
        } else if (p.rfind("mixed:", 0) == 0) {
            pcs_allow_fp16 = true;
            std::string rest = p.substr(6);
            size_t pos = 0;
            while (pos < rest.size()) {
                size_t comma = rest.find(',', pos);
                if (comma == std::string::npos) comma = rest.size();
                if (comma > pos) pcs_fp32_patterns.push_back(rest.substr(pos, comma - pos));
                pos = comma + 1;
            }
        } else if (p != "fp32") {
            fprintf(stderr, "%s: unknown SAM3_TRT_PCS_PRECISION '%s', using fp32\n", __func__, p.c_str());
        }
    }
    sam3_trt_engine* eng = sam3_get_trt_engine_cached("SAM3_TRT_PCS_ONNX_PATH", "SAM3_TRT_PCS_CACHE_DIR",
                                                      pcs_allow_fp16, pcs_fp32_patterns);
    if (!eng) return false;

    // Geometry input tokens (exemplar boxes' embeddings + precomputed
    // concept rows + CLS, see sam3_precompute_geom_input). ROI pooling needs
    // the backbone features on the CPU only when same-image boxes are
    // present (one 5.3MB D2H copy per such request; plain text prompts and
    // precomputed-embedding prompts skip it).
    std::vector<float> geom_feats_cpu;
    const float* geom_feats_ptr = nullptr;
    if (n_boxes > 0) {
        geom_feats_cpu.resize((size_t)D * N_spatial);
        ggml_backend_tensor_get(state.neck_det[2], geom_feats_cpu.data(), 0,
                                geom_feats_cpu.size() * sizeof(float));
        geom_feats_ptr = geom_feats_cpu.data();
    }
    sam3_trt_tensor geom_in; geom_in.name = "geom_in";
    geom_in.f32 = sam3_precompute_geom_input(model, params, geom_feats_ptr, H, H);
    geom_in.dims = {1, n_geo, D};
    if ((int)geom_in.f32.size() != n_geo * D) {
        fprintf(stderr, "%s: geometry precompute size mismatch (%zu vs %d)\n",
                __func__, geom_in.f32.size(), n_geo * D);
        return false;
    }

    // Read straight from the encoder's device buffers -- no
    // device->host->device round trip (see sam3_trt_runtime.h's device_ptr
    // doc). state.neck_det[i]->data is already a valid CUDA pointer whether
    // the encoder ran via TRT (direct-bound) or ggml (regular backend
    // tensor); either way it holds fully-written data by the time we get
    // here (sam3_segment_pcs's own "image not encoded" check already ran).
    sam3_trt_tensor tok_in; tok_in.name = "token_ids"; tok_in.i32 = token_ids;
    sam3_trt_tensor img_feats_in; img_feats_in.name = "img_feats";
    img_feats_in.device_ptr = state.neck_det[2]->data;
    sam3_trt_tensor fpn0_in; fpn0_in.name = "fpn0"; fpn0_in.device_ptr = state.neck_det[0]->data;
    sam3_trt_tensor fpn1_in; fpn1_in.name = "fpn1"; fpn1_in.device_ptr = state.neck_det[1]->data;

    std::vector<sam3_trt_tensor> outputs;
    if (!sam3_trt_engine_run(eng, {tok_in, geom_in, img_feats_in, fpn0_in, fpn1_in}, outputs)) {
        fprintf(stderr, "%s: sam3_trt_engine_run failed, falling back to ggml\n", __func__);
        return false;
    }
    auto find_out = [&](const char* name) -> std::vector<float>* {
        for (auto& t : outputs) if (t.name == name) return &t.f32;
        return nullptr;
    };
    auto* scores_data = find_out("class_scores");
    auto* boxes_data = find_out("pred_boxes");
    auto* presence_out = find_out("presence_score");
    auto* all_masks = find_out("mask_logits");
    if (!scores_data || !boxes_data || !presence_out || !all_masks) {
        fprintf(stderr, "%s: engine missing expected outputs, falling back to ggml\n", __func__);
        return false;
    }
    const float presence_logit = (*presence_out)[0];
    const float presence_prob = 1.0f / (1.0f + expf(-presence_logit));

    // ── Post-processing: identical to the ggml path (sam3_segment_pcs below) ──
    static const bool stage_timing_pcs = getenv("SAM3_STAGE_TIMING") != nullptr;
    auto t_post0 = std::chrono::high_resolution_clock::now();
    std::vector<sam3_detection> dets;
    for (int q = 0; q < NQ; ++q) {
        float class_prob = 1.0f / (1.0f + expf(-(*scores_data)[q]));
        float score = class_prob * presence_prob;
        if (score < params.score_threshold) continue;

        sam3_detection det;
        float cx = (*boxes_data)[0 + q * 4];
        float cy = (*boxes_data)[1 + q * 4];
        float bw = (*boxes_data)[2 + q * 4];
        float bh = (*boxes_data)[3 + q * 4];

        det.box = sam3_cxcywh_to_xyxy(cx, cy, bw, bh, state.orig_width, state.orig_height);
        det.score = score;

        const float* mask_ptr = all_masks->data() + q * mask_hw * mask_hw;
        auto mask_resized = sam3_bilinear_interpolate(mask_ptr, mask_hw, mask_hw,
                                                      state.orig_width, state.orig_height);
        det.mask.width = state.orig_width;
        det.mask.height = state.orig_height;
        det.mask.data.resize(state.orig_width * state.orig_height);
        for (int i = 0; i < (int)mask_resized.size(); ++i)
            det.mask.data[i] = (mask_resized[i] > 0.0f) ? 255 : 0;
        det.mask.iou_score = score;

        dets.push_back(std::move(det));
    }

    auto keep = sam3_nms(dets, params.nms_threshold);
    for (int i = 0; i < (int)keep.size(); ++i) {
        dets[keep[i]].instance_id = i + 1;
        out_result.detections.push_back(std::move(dets[keep[i]]));
    }
    if (stage_timing_pcs) {
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_post0).count();
        fprintf(stderr, "[stage_timing] pcs_post (CPU mask upscale 288^2->%dx%d + binarize + NMS, "
                        "%zu detections): %.2f ms\n",
                state.orig_width, state.orig_height, out_result.detections.size(), ms);
    }
    return true;
}
#endif  // SAM3_TRT_ENCODER
