// sam3_dump_encoder_weights — export the ViT backbone + SimpleFPN neck
// weights of a loaded SAM3 model to raw f32 blobs + a manifest.json, for
// the offline TensorRT ONNX-authoring step (convert_sam3_encoder_to_onnx.py).
//
// This is a one-time, dev-machine tool (part of the TensorRT image-encoder
// migration, see docs/sam3/PLAN.md) — it does not run at serving time and
// is not part of the deployed server.
//
// Usage: sam3_dump_encoder_weights --model m.ggml --out encoder_export/
//
// Reuses the existing public sam3_get_model_tensor_info / sam3_dump_model_tensor
// API (sam3.h) rather than reaching into sam3_model internals — every tensor
// this tool touches goes through the same, already-validated dequantization
// path sam3_load_tensors itself uses.

#include "sam3.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <sys/stat.h>

static const char* sam3_ggml_type_name(int type) {
    switch (type) {
        case 0:  return "f32";
        case 1:  return "f16";
        default: return "dequant_f32";  // any quantized source type, already dequantized on dump
    }
}

// Appends a manifest entry for `tensor_name` if it exists in the model, dumping
// its payload to `<out_dir>/<tensor_name>.bin` (+ a redundant .shape file
// written by sam3_dump_model_tensor itself). Returns false only on a real
// dump failure (a missing *optional* tensor, e.g. a detector-neck tensor in a
// visual-only model, is handled by the caller, not here).
static bool sam3_dump_one(const sam3_model& model, const std::string& out_dir,
                           const std::string& name, std::ofstream& manifest, bool& first) {
    sam3_tensor_info info;
    if (!sam3_get_model_tensor_info(model, name, info)) {
        return false;  // not registered in this model -- caller decides if that's fatal
    }
    if (!sam3_dump_model_tensor(model, name, out_dir + "/" + name)) {
        fprintf(stderr, "%s: failed to dump '%s'\n", __func__, name.c_str());
        return false;
    }
    // sam3_dump_tensor_to_path always writes a *dequantized* f32 buffer,
    // regardless of the tensor's in-memory type -- so the manifest always
    // records "f32", matching what's actually on disk.
    if (!first) manifest << ",\n";
    first = false;
    // ggml_ne is always the full 4-tuple in ggml order (ne[0] fastest-varying,
    // trailing entries padded to 1 when the logical tensor has fewer than 4
    // dims). Deliberately NOT also emitting a "squeezed" numpy-order shape
    // here: ggml's own size-1 padding always sits at the *end* of ne[] (e.g.
    // a 2D qkv weight's ne=[1024,3072,1,1]), but a real 1x1 conv kernel's
    // size-1 dims sit at the *front* in ggml order (ne=[1,1,D,D]) -- reversing
    // and blindly trimming trailing 1s can't tell those two apart. The
    // authoring script (Stage B) knows each tensor's role (linear/conv/bias)
    // and reshapes accordingly from this authoritative, never-squeezed tuple.
    manifest << "  {\"name\": \"" << name << "\", \"file\": \"" << name << ".bin\""
             << ", \"dtype\": \"f32\", \"ggml_ne\": ["
             << info.ne[0] << ", " << info.ne[1] << ", " << info.ne[2] << ", " << info.ne[3]
             << "]}";
    return true;
}

int main(int argc, char** argv) {
    sam3_params params;
    params.use_gpu = false;  // pure weight export -- no inference, CPU load is enough
    std::string out_dir = "encoder_export";

    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--model") == 0 && i + 1 < argc) { params.model_path = argv[++i]; }
        else if (strcmp(argv[i], "--out")   == 0 && i + 1 < argc) { out_dir = argv[++i]; }
        else if (strcmp(argv[i], "--gpu")   == 0)                  { params.use_gpu = true; }
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (params.model_path.empty()) {
        fprintf(stderr, "usage: %s --model m.ggml [--out encoder_export/] [--gpu]\n", argv[0]);
        return 1;
    }

    auto model = sam3_load_model(params);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const sam3_model_type mtype = sam3_get_model_type(*model);
    if (mtype != SAM3_MODEL_SAM3 && mtype != SAM3_MODEL_SAM3_VISUAL) {
        fprintf(stderr, "error: --model is not a SAM3 (ViT) model (SAM2/EdgeTAM image encoders "
                         "use a different backbone and are out of scope for this phase)\n");
        return 1;
    }
    const bool has_det_neck = !sam3_is_visual_only(*model);

    mkdir(out_dir.c_str(), 0755);
    std::ofstream manifest(out_dir + "/manifest.json");
    if (!manifest) { fprintf(stderr, "failed to open manifest.json for writing\n"); return 1; }
    manifest << "{\n\"tensors\": [\n";
    bool first = true;
    int n_dumped = 0;

    auto dump_required = [&](const std::string& name) {
        if (!sam3_dump_one(*model, out_dir, name, manifest, first)) {
            fprintf(stderr, "error: required tensor '%s' missing from model\n", name.c_str());
            exit(1);
        }
        n_dumped++;
    };

    // ── ViT prefix ───────────────────────────────────────────────────────
    dump_required("vit.patch_embed.proj.weight");
    dump_required("vit.pos_embed");
    dump_required("vit.ln_pre.weight");
    dump_required("vit.ln_pre.bias");

    // ── ViT blocks (dynamic depth discovery: probe until a block is missing) ──
    for (int i = 0; ; ++i) {
        const std::string p = "vit.blocks." + std::to_string(i) + ".";
        sam3_tensor_info probe;
        if (!sam3_get_model_tensor_info(*model, p + "norm1.weight", probe)) break;
        dump_required(p + "norm1.weight");
        dump_required(p + "norm1.bias");
        dump_required(p + "attn.qkv.weight");
        dump_required(p + "attn.qkv.bias");
        dump_required(p + "attn.proj.weight");
        dump_required(p + "attn.proj.bias");
        dump_required(p + "norm2.weight");
        dump_required(p + "norm2.bias");
        dump_required(p + "mlp.lin1.weight");
        dump_required(p + "mlp.lin1.bias");
        dump_required(p + "mlp.lin2.weight");
        dump_required(p + "mlp.lin2.bias");
        dump_required(p + "attn.freqs_cis");
    }

    // ── SimpleFPN neck(s) ────────────────────────────────────────────────
    auto dump_neck = [&](const std::string& prefix) {
        dump_required(prefix + "0.dconv_2x2_0.weight");
        dump_required(prefix + "0.dconv_2x2_0.bias");
        dump_required(prefix + "0.dconv_2x2_1.weight");
        dump_required(prefix + "0.dconv_2x2_1.bias");
        dump_required(prefix + "0.conv_1x1.weight");
        dump_required(prefix + "0.conv_1x1.bias");
        dump_required(prefix + "0.conv_3x3.weight");
        dump_required(prefix + "0.conv_3x3.bias");

        dump_required(prefix + "1.dconv_2x2.weight");
        dump_required(prefix + "1.dconv_2x2.bias");
        dump_required(prefix + "1.conv_1x1.weight");
        dump_required(prefix + "1.conv_1x1.bias");
        dump_required(prefix + "1.conv_3x3.weight");
        dump_required(prefix + "1.conv_3x3.bias");

        dump_required(prefix + "2.conv_1x1.weight");
        dump_required(prefix + "2.conv_1x1.bias");
        dump_required(prefix + "2.conv_3x3.weight");
        dump_required(prefix + "2.conv_3x3.bias");

        dump_required(prefix + "3.conv_1x1.weight");
        dump_required(prefix + "3.conv_1x1.bias");
        dump_required(prefix + "3.conv_3x3.weight");
        dump_required(prefix + "3.conv_3x3.bias");
    };
    if (has_det_neck) {
        dump_neck("neck.det.");
    }
    dump_neck("neck.trk.");

    manifest << "\n]}\n";
    manifest.close();

    fprintf(stderr, "%s: dumped %d tensors to '%s' (manifest.json)\n", __func__, n_dumped, out_dir.c_str());
    return 0;
}
