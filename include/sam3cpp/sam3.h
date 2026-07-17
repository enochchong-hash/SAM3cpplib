// sam3cpplib -- SAM3 single-image promptable segmentation (text, exemplar-box,
// point, and box prompts). First-party port of tools/sam3.cpp @ 049884b
// (upstream PABannier/sam3.cpp, MIT, + gemma4 patches 0001-0013); see
// docs/PLAN.md. This header is source-compatible with the sam3.h the
// release/sam3 server builds against today, minus the video-tracking /
// SAM2 / EdgeTAM API (deliberately not ported).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/*
** ── Version ─────────────────────────────────────────────────────────────
*/

#define SAM3_VERSION_MAJOR 1
#define SAM3_VERSION_MINOR 0
#define SAM3_VERSION_PATCH 0
#define SAM3_VERSION       "1.0.0"

/*
** ── Forward Declarations ─────────────────────────────────────────────────
*/

struct sam3_model;
struct sam3_state;

/* Custom deleter so unique_ptr works with the forward-declared opaque type. */
struct sam3_state_deleter { void operator()(sam3_state * p) const; };

using sam3_state_ptr = std::unique_ptr<sam3_state, sam3_state_deleter>;

/*
** ── Model Type ──────────────────────────────────────────────────────────
*/

enum sam3_model_type {
    SAM3_MODEL_SAM3        = 0,  // Full SAM3 (ViT + detector)
    SAM3_MODEL_SAM3_VISUAL = 1,  // SAM3 visual-only (ViT, no text/detector)
    // Kept for source compatibility only -- sam3cpplib does not load these
    // model files (sam3_load_model rejects them; use release/sam3's bundled
    // upstream library for video tracking with SAM2/EdgeTAM).
    SAM3_MODEL_SAM2        = 2,
    SAM3_MODEL_EDGETAM     = 3,
};

/*****************************************************************************
** Public Data Types
**
** Geometry primitives, images, masks, and detection results.
*****************************************************************************/

struct sam3_point {
    float x;
    float y;
};

struct sam3_box {
    float x0;  // top-left x
    float y0;  // top-left y
    float x1;  // bottom-right x
    float y1;  // bottom-right y
};

struct sam3_image {
    int width    = 0;
    int height   = 0;
    int channels = 3;
    std::vector<uint8_t> data;
};

struct sam3_mask {
    int   width       = 0;
    int   height      = 0;
    float iou_score   = 0.0f;
    float obj_score   = 0.0f;
    int   instance_id = -1;
    std::vector<uint8_t> data;  // binary mask (0 or 255)
};

struct sam3_detection {
    sam3_box  box;
    float     score     = 0.0f;
    float     iou_score = 0.0f;
    int       instance_id = -1;
    sam3_mask  mask;
    std::vector<float> sam_token;  // raw SAM decoder output token
};

struct sam3_result {
    std::vector<sam3_detection> detections;
};

/*****************************************************************************
** Mask convenience accessors
**
** Pure, CPU-side geometry derived from a raw sam3_mask (0/255 bytes) -- for
** library consumers that want ready-made bounding box / centroid / pixel
** coordinates instead of scanning the mask themselves. Foreground = byte
** > 127. Coordinates are in the mask's own pixel space (mask.width x
** mask.height), which for detections equals the original image resolution.
*****************************************************************************/

// Foreground pixel count.
size_t sam3_mask_area(const sam3_mask & mask);

// Center of mass of foreground pixels (pixel coords). Returns {-1, -1} if
// the mask is empty.
sam3_point sam3_mask_centroid(const sam3_mask & mask);

// Tight axis-aligned bounding box of the foreground (x0,y0 inclusive; x1,y1
// exclusive, i.e. width = x1-x0). May be tighter than sam3_detection::box,
// which is the model's predicted box. Returns {0,0,0,0} if empty.
sam3_box sam3_mask_bbox(const sam3_mask & mask);

// Foreground test at integer pixel (x, y); false if out of range or empty.
bool sam3_mask_at(const sam3_mask & mask, int x, int y);

// All foreground pixel coordinates, row-major (y-major) order.
std::vector<sam3_point> sam3_mask_coords(const sam3_mask & mask);

/*****************************************************************************
** Parameters
**
** Configuration for model loading and segmentation.
*****************************************************************************/

// TensorRT backend configuration -- the programmatic alternative to the
// SAM3_TRT_* environment variables for embedded consumers (a host app should
// not have to mutate its own environment). Only meaningful in builds with
// SAM3CPP_TENSORRT=ON; ignored otherwise. Any field left empty falls back to
// the corresponding environment variable (SAM3_TRT_ONNX_PATH,
// SAM3_TRT_ONNX_PATH_FP8, SAM3_TRT_PCS_ONNX_PATH, SAM3_TRT_PVS_ONNX_PATH,
// SAM3_TRT_CACHE_DIR, SAM3_TRT_PCS_PRECISION, SAM3_TRT_SKIP_GGML_WEIGHTS),
// so env-var-driven deployments keep working untouched.
struct sam3_trt_config {
    bool        enabled = false;      // master switch (equivalent of SAM3_TRT_ENCODER=1)
    std::string encoder_onnx;         // image encoder ONNX (FP16 engine)
    std::string encoder_onnx_fp8;     // optional FP8-quantized encoder ONNX
    std::string pcs_onnx;             // PCS (text/exemplar head) ONNX
    std::string pcs_onnx_fp8;         // optional FP8-quantized PCS ONNX (fenc/ddec GEMMs)
    std::string pvs_onnx;             // PVS (point/box head) ONNX
    std::string cache_dir;            // serialized-engine cache root
    std::string pcs_precision = "mixed:text_";  // fp32 | fp16 | mixed:<substr,...>
    bool        skip_ggml_weights = true;       // TRT-only deployments: don't load
                                                // the ~1.1GB ggml weights at all
};

struct sam3_params {
    std::string model_path;
    int         n_threads       = 4;
    bool        use_gpu         = true;
    int         seed            = 42;
    int         encode_img_size = 0;  // 0 = model default; override input resolution
    sam3_trt_config trt;              // see above (SAM3CPP_TENSORRT builds)
};

struct sam3_tensor_info {
    int64_t ne[4] = {0, 0, 0, 0};
    uint64_t nb[4] = {0, 0, 0, 0};
    int type = 0;
    int op = 0;
    bool is_contiguous = false;
};

enum sam3_vit_block_stage {
    SAM3_VIT_BLOCK_STAGE_NORM1 = 0,
    SAM3_VIT_BLOCK_STAGE_WINDOW_PART,
    SAM3_VIT_BLOCK_STAGE_QKV_PROJ,
    SAM3_VIT_BLOCK_STAGE_ATTN_CORE,
    SAM3_VIT_BLOCK_STAGE_ATTN_PROJ,
    SAM3_VIT_BLOCK_STAGE_WINDOW_UNPART,
    SAM3_VIT_BLOCK_STAGE_NORM2,
    SAM3_VIT_BLOCK_STAGE_MLP_FC1,
    SAM3_VIT_BLOCK_STAGE_MLP_GELU,
    SAM3_VIT_BLOCK_STAGE_MLP_FC2,
    SAM3_VIT_BLOCK_STAGE_MLP,
};

enum sam3_vit_prefix_stage {
    SAM3_VIT_PREFIX_STAGE_PATCH_EMBED = 0,
    SAM3_VIT_PREFIX_STAGE_PATCH_IM2COL,
    SAM3_VIT_PREFIX_STAGE_PATCH_MULMAT_RAW,
    SAM3_VIT_PREFIX_STAGE_PATCH_MULMAT,
    SAM3_VIT_PREFIX_STAGE_POS_ADD,
    SAM3_VIT_PREFIX_STAGE_LN_PRE_NORM,
    SAM3_VIT_PREFIX_STAGE_LN_PRE,
};

struct sam3_pcs_params {
    std::string            text_prompt;
    // Same-image exemplar boxes, NORMALIZED [0,1] XYXY (the geometry encoder
    // ROI-pools backbone features from these regions of the encoded image).
    std::vector<sam3_box>  pos_exemplars;
    std::vector<sam3_box>  neg_exemplars;
    // Precomputed concept-embedding rows (256 floats each), captured earlier
    // via sam3_pcs_compute_exemplar_embedding -- typically from a REFERENCE
    // image -- and injected as additional geometry prompt tokens. Lets a
    // concept library be built offline and applied to live frames without
    // re-encoding the reference image. EXPERIMENTAL: cross-image exemplars
    // are outside the model's training distribution; validate per use case.
    std::vector<std::vector<float>> exemplar_embeddings;
    float                  score_threshold = 0.5f;
    float                  nms_threshold   = 0.1f;
};

struct sam3_pvs_params {
    std::vector<sam3_point> pos_points;
    std::vector<sam3_point> neg_points;
    sam3_box                box      = {0, 0, 0, 0};
    bool                    use_box  = false;
    bool                    multimask = false;
};

/*****************************************************************************
** Public API
**
** Model lifecycle, image encoding, and segmentation.
*****************************************************************************/

/*
** ── Model Lifecycle ──────────────────────────────────────────────────────
*/

/*
** Load a SAM3 model from the file specified in params.model_path.
** Returns nullptr on failure.
*/
std::shared_ptr<sam3_model> sam3_load_model(const sam3_params & params);

/* Free all resources held by a loaded model. */
void sam3_free_model(sam3_model & model);

/* Returns true if the model was loaded as visual-only (no text/detector path). */
bool sam3_is_visual_only(const sam3_model & model);

/* Returns the model type. */
sam3_model_type sam3_get_model_type(const sam3_model & model);

/*
** ── Inference State ──────────────────────────────────────────────────────
*/

/* Allocate inference state (backbone caches, PE buffers). */
sam3_state_ptr sam3_create_state(const sam3_model & model,
                                const sam3_params & params);

/* Free inference state and its GPU buffers. */
void sam3_free_state(sam3_state & state);

/*
** ── Image Backbone ───────────────────────────────────────────────────────
*/

/*
** Encode an image through the ViT backbone and FPN neck.
** Call once per image before segmentation.
** Returns true on success, false on failure.
*/
// Select the image-encoder precision at runtime (TensorRT builds only):
// fp8=true uses the FP8 engine configured via encoder_onnx_fp8 /
// SAM3_TRT_ONNX_PATH_FP8 (lower expected accuracy, ~450MB less VRAM,
// slightly faster); fp8=false (default) uses the FP16 engine. Both engines
// stay resident once used -- switching per request is free after the first
// load. Falls back to FP16 with a log line if the FP8 engine is unavailable.
void sam3_set_encoder_fp8(sam3_state& state, bool fp8);

// Same contract for the PCS (text/exemplar) head: fp8=true uses the FP8
// engine configured via pcs_onnx_fp8 / SAM3_TRT_PCS_ONNX_PATH_FP8. The FP8
// PCS graph quantizes the fusion-encoder and DETR-decoder linear GEMMs only
// -- the text encoder stays FP32 (FP16/FP8-sensitive), attention stays FP16
// (fused MHA), and the bbox/RPB MLPs, geometry encoder, scoring and seg head
// stay FP16 (tiny or numerically delicate). See docs/tensorrt.md for the
// full per-subsystem precision map and how to generate the FP8 graph.
// Falls back to the standard PCS engine with a log line if unavailable.
// The PVS head has no FP8 (or FP16) variant BY DESIGN: it must run FP32
// (FP16 already diverges 27% on negative-point prompts) and its engine is
// only ~32MB, so there is nothing to win.
void sam3_set_pcs_fp8(sam3_state& state, bool fp8);

bool sam3_encode_image(sam3_state       & state,
                       const sam3_model & model,
                       const sam3_image & image);

/*
** ── Image Segmentation ──────────────────────────────────────────────────
*/

/* Segment using text prompt + exemplar boxes (PCS path). */
// Compute a reusable concept-embedding row (256 floats) for one exemplar box
// on the CURRENTLY ENCODED image: call sam3_encode_image on the reference
// image first, then this per box. `box_normalized` is [0,1] XYXY;
// `positive` selects the positive/negative exemplar label embedding. The
// returned row can be persisted (1KB) and later supplied in
// sam3_pcs_params::exemplar_embeddings on any image. Returns empty on error.
std::vector<float> sam3_pcs_compute_exemplar_embedding(sam3_state& state,
                                                       const sam3_model& model,
                                                       const sam3_box& box_normalized,
                                                       bool positive);

sam3_result sam3_segment_pcs(sam3_state             & state,
                             const sam3_model       & model,
                             const sam3_pcs_params  & params);

/* Segment using point/box prompts (PVS path). */
sam3_result sam3_segment_pvs(sam3_state             & state,
                             const sam3_model       & model,
                             const sam3_pvs_params  & params);

/*
** ── Utility ─────────────────────────────────────────────────────────────
**
** Convenience image I/O (stb-backed). Compiled only when the library is
** built with SAM3CPP_IMAGE_IO=ON (the default); the core inference library
** itself consumes raw RGB sam3_image buffers and never decodes files.
*/

sam3_image sam3_load_image(const std::string & path);
bool       sam3_save_mask(const sam3_mask & mask, const std::string & path);

/*****************************************************************************
** Test and Debug API
**
** Standalone tokenizer, intermediate tensor dumps, and debug utilities.
** These functions are intended for testing and development only (they back
** the golden-sample process and the parity tests; see docs/goldens.md).
*****************************************************************************/

bool                  sam3_test_load_tokenizer(const std::string & model_path);
std::vector<int32_t>  sam3_test_tokenize(const std::string & text);

/*
** Run the text encoder on fixed token IDs and dump standard intermediate
** tensors to <output_dir>/<tensor_name>.{bin,shape}.
*/
bool sam3_test_dump_text_encoder(const sam3_model & model,
                                 const std::vector<int32_t> & token_ids,
                                 const std::string & output_dir,
                                 int n_threads = 4);

/*
** Run the full phase 5 detector path (fusion encoder + DETR decoder +
** dot-product scoring + segmentation head) on an already-encoded image
** and dump intermediate tensors.
*/
bool sam3_test_dump_phase5(const sam3_model & model,
                           const sam3_state & state,
                           const std::vector<int32_t> & token_ids,
                           const std::string & output_dir,
                           int n_threads = 4);

/*
** Run the phase 5 detector from pre-dumped inputs instead of re-running
** the image/text encoders.  Isolates detector numerics from earlier phases.
*/
bool sam3_test_dump_phase5_from_ref_inputs(const sam3_model & model,
                                           const std::vector<int32_t> & token_ids,
                                           const std::string & prephase_ref_dir,
                                           const std::string & phase5_ref_dir,
                                           const std::string & output_dir,
                                           int n_threads = 4);

/*
** Run the phase 6 prompt encoder + SAM decoder on an already-encoded
** image state and dump intermediate tensors.
*/
bool sam3_test_dump_phase6(const sam3_model & model,
                           const sam3_state & state,
                           const sam3_pvs_params & params,
                           const std::string & output_dir,
                           int n_threads = 4);

/*
** Run the phase 6 prompt encoder + SAM decoder from pre-dumped phase 3
** tracker features.  Isolates phase 6 numerics from earlier phases.
*/
bool sam3_test_dump_phase6_from_ref_inputs(const sam3_model & model,
                                           const std::string & prephase_ref_dir,
                                           const sam3_pvs_params & params,
                                           const std::string & output_dir,
                                           int n_threads = 4);

/*
** Run the geometry encoder from pre-computed backbone features and dump
** intermediate tensors.  Tests exemplar box coordinate encoding against
** Python reference.
*/
bool sam3_test_dump_geom_enc(const sam3_model   & model,
                              const std::string  & prephase_ref_dir,
                              const sam3_pcs_params & params,
                              const std::string  & output_dir,
                              int                  n_threads = 4);

/*
** Run ONLY the fusion encoder (6 layers) from pre-dumped inputs (image
** features, pos encoding, prompt tokens, attn bias).  Dumps per-layer
** outputs for isolated fenc debugging.
*/
bool sam3_test_fenc_only(const sam3_model  & model,
                          const std::string & ref_dir,
                          const std::string & output_dir,
                          int                 n_threads = 4);

/*
** ── Debug ────────────────────────────────────────────────────────────────
*/

/* Dump a named state tensor to a binary file for verification. */
bool sam3_dump_state_tensor(const sam3_state & state,
                             const std::string & tensor_name,
                             const std::string & output_path);

/* Query metadata for a named state tensor without dumping its payload. */
bool sam3_get_state_tensor_info(const sam3_state & state,
                                const std::string & tensor_name,
                                sam3_tensor_info  & info);

/* Dump a named model tensor (weights/constants) to a binary file. */
bool sam3_dump_model_tensor(const sam3_model   & model,
                            const std::string  & tensor_name,
                            const std::string  & output_path);

/* Query metadata for a named model tensor. */
bool sam3_get_model_tensor_info(const sam3_model  & model,
                                const std::string & tensor_name,
                                sam3_tensor_info  & info);

/*
** Encode an image from pre-preprocessed float data (CHW layout, already
** resized and normalized).  Bypasses C++ preprocessing so that numerical
** comparisons against the Python reference are not polluted by resize
** implementation differences.
*/
bool sam3_encode_image_from_preprocessed(sam3_state       & state,
                                          const sam3_model & model,
                                          const float      * chw_data,
                                          int                img_size);

/*
** Test-only: run ONLY the ViT encoder from preprocessed float data and keep
** only the requested intermediate tensors alive for dumping/comparison.
*/
bool sam3_encode_vit_from_preprocessed_selective(sam3_state                    & state,
                                                 const sam3_model              & model,
                                                 const float                   * chw_data,
                                                 int                             img_size,
                                                 const std::vector<std::string> & output_tensors);

/*
** Test-only: run the exact ViT prefix up to the tensor entering block 0
** (patch embed + pos embed + ln_pre).
*/
bool sam3_test_run_vit_block0_input(const sam3_model   & model,
                                    const float        * chw_data,
                                    int                  img_size,
                                    std::vector<float> & output_data,
                                    int64_t              output_ne[4],
                                    int                  n_threads = 4);

/*
** Test-only: run an exact ViT prefix sub-stage on the real model tensors using
** the model's backend. PATCH_EMBED expects image input [W,H,3,1]. Later stages
** expect feature input [E,W,H,1] in ggml 4D layout.
*/
bool sam3_test_run_vit_prefix_stage(const sam3_model         & model,
                                    sam3_vit_prefix_stage      stage,
                                    const float              * input_data,
                                    const int64_t              input_ne[4],
                                    std::vector<float>       & output_data,
                                    int64_t                    output_ne[4],
                                    int                        n_threads = 4);
bool sam3_test_run_patch_mulmat_host_ref(const sam3_model         & model,
                                         const float              * input_data,
                                         const int64_t              input_ne[4],
                                         bool                       use_double_accum,
                                         std::vector<float>       & output_data,
                                         int64_t                    output_ne[4]);
bool sam3_test_run_vit_block_linear_host_ref(const sam3_model         & model,
                                             int                        block_idx,
                                             sam3_vit_block_stage       stage,
                                             const float              * input_data,
                                             const int64_t              input_ne[4],
                                             bool                       use_double_accum,
                                             std::vector<float>       & output_data,
                                             int64_t                    output_ne[4]);

/*
** Test-only: run an exact ViT block sub-stage on the real model tensors using
** the model's backend. The input is always provided as F32 in ggml 4D layout.
*/
bool sam3_test_run_vit_block_stage(const sam3_model        & model,
                                   int                       block_idx,
                                   sam3_vit_block_stage      stage,
                                   const float             * input_data,
                                   const int64_t             input_ne[4],
                                   std::vector<float>      & output_data,
                                   int64_t                   output_ne[4],
                                   int                       n_threads = 4);
