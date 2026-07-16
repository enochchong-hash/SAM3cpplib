// sam3cpplib -- ported from tools/sam3.cpp @ 049884b (wip/trt-phase1,
// upstream PABannier/sam3.cpp + gemma4 patches 0001-0013). See docs/PLAN.md.
#include "sam3_internal.h"

/*****************************************************************************
** Inference state
*****************************************************************************/

// Deleter implementations for opaque types
void sam3_state_deleter::operator()(sam3_state* p) const {
    if (p) {
        sam3_free_state(*p);
        delete p;
    }
}

sam3_state_ptr sam3_create_state(const sam3_model& model,
                                 const sam3_params& params) {
    sam3_state_ptr state(new sam3_state());
    state->backend = model.backend;
    state->n_threads = (params.n_threads > 0)
                           ? params.n_threads
                           : std::max(1u, std::thread::hardware_concurrency());

    const auto& hp = model.hparams;
    int eis = (params.encode_img_size > 0) ? params.encode_img_size : hp.img_size;
    state->encode_img_size  = eis;
    state->encode_feat_size = sam3_effective_feat_size(hp, eis);
    if (eis != hp.img_size) {
        fprintf(stderr, "%s: encode_img_size=%d (model native=%d), feat_size=%d (native=%d)\n",
                __func__, eis, hp.img_size, state->encode_feat_size, hp.feat_size());
    }

    return state;
}

void sam3_free_state(sam3_state& state) {
    if (state.galloc) {
        ggml_gallocr_free(state.galloc);
        state.galloc = nullptr;
    }
    if (state.buffer) {
        ggml_backend_buffer_free(state.buffer);
        state.buffer = nullptr;
    }
    if (state.pe_buf) {
        ggml_backend_buffer_free(state.pe_buf);
        state.pe_buf = nullptr;
    }
    if (state.pe_ctx) {
        ggml_free(state.pe_ctx);
        state.pe_ctx = nullptr;
    }
    if (state.trt_out_buf) {
        ggml_backend_buffer_free(state.trt_out_buf);
        state.trt_out_buf = nullptr;
    }
    if (state.trt_out_ctx) {
        ggml_free(state.trt_out_ctx);
        state.trt_out_ctx = nullptr;
    }
    if (state.ctx) {
        ggml_free(state.ctx);
        state.ctx = nullptr;
    }
}

