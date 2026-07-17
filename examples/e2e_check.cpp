// sam3_pcs_pvs_e2e_check — encode a real image (ggml GPU path) then run PCS
// (text prompt) and PVS (point prompt), printing detections. Run this binary
// twice (once with no SAM3_TRT_* env vars, once with SAM3_TRT_ENCODER=1 +
// SAM3_TRT_PCS_ONNX_PATH/SAM3_TRT_PVS_ONNX_PATH set) and diff the two runs'
// output to validate the TensorRT PCS/PVS paths against the ggml path on a
// real image. Dev-only validation tool for the TensorRT migration
// (docs/sam3/PLAN.md) -- not part of the deployed server.
//
// Usage: sam3_pcs_pvs_e2e_check --model m.ggml --image cat.jpg --text "cat"
//                                --point-x 600 --point-y 600 [--out dir]
#include "sam3.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <sys/stat.h>

// "x,y;x,y;..." -> list of points
static std::vector<sam3_point> parse_points(const char* s) {
    std::vector<sam3_point> pts;
    float x = 0, y = 0;
    while (*s && sscanf(s, "%f,%f", &x, &y) == 2) {
        pts.push_back({x, y});
        const char* semi = strchr(s, ';');
        if (!semi) break;
        s = semi + 1;
    }
    return pts;
}

int main(int argc, char** argv) {
    sam3_params params;
    params.use_gpu = true;
    std::string image_path, text = "cat", out_dir = "pcs_pvs_e2e";
    float point_x = 600, point_y = 600;
    std::vector<sam3_point> pos_points, neg_points;
    bool use_box = false;
    float box[4] = {0, 0, 0, 0};
    // Exemplar boxes in PIXEL coords "x0,y0,x1,y1;..." (normalized to [0,1]
    // after the image loads); embedding files hold one raw 256-float row.
    std::vector<sam3_box> pos_ex_px, neg_ex_px;
    bool use_fp8 = false;
    bool use_pcs_fp8 = false;
    int n_runs = 3;  // cold + warm repeats; goldens use --runs 1
    std::string save_embedding_path, use_embedding_path;

    auto parse_boxes = [](const char* s) {
        std::vector<sam3_box> out;
        float b[4];
        while (*s && sscanf(s, "%f,%f,%f,%f", &b[0], &b[1], &b[2], &b[3]) == 4) {
            out.push_back({b[0], b[1], b[2], b[3]});
            const char* semi = strchr(s, ';');
            if (!semi) break;
            s = semi + 1;
        }
        return out;
    };

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--model") && i + 1 < argc)   { params.model_path = argv[++i]; }
        else if (!strcmp(argv[i], "--image") && i + 1 < argc)   { image_path = argv[++i]; }
        else if (!strcmp(argv[i], "--text") && i + 1 < argc)    { text = argv[++i]; }
        else if (!strcmp(argv[i], "--point-x") && i + 1 < argc) { point_x = strtof(argv[++i], nullptr); }
        else if (!strcmp(argv[i], "--point-y") && i + 1 < argc) { point_y = strtof(argv[++i], nullptr); }
        else if (!strcmp(argv[i], "--points") && i + 1 < argc)     { pos_points = parse_points(argv[++i]); }
        else if (!strcmp(argv[i], "--neg-points") && i + 1 < argc) { neg_points = parse_points(argv[++i]); }
        else if (!strcmp(argv[i], "--box") && i + 1 < argc) {
            if (sscanf(argv[++i], "%f,%f,%f,%f", &box[0], &box[1], &box[2], &box[3]) == 4) use_box = true;
        }
        else if (!strcmp(argv[i], "--pos-exemplars") && i + 1 < argc) { pos_ex_px = parse_boxes(argv[++i]); }
        else if (!strcmp(argv[i], "--neg-exemplars") && i + 1 < argc) { neg_ex_px = parse_boxes(argv[++i]); }
        else if (!strcmp(argv[i], "--save-embedding") && i + 1 < argc) { save_embedding_path = argv[++i]; }
        else if (!strcmp(argv[i], "--use-embedding") && i + 1 < argc)  { use_embedding_path = argv[++i]; }
        else if (!strcmp(argv[i], "--out") && i + 1 < argc)     { out_dir = argv[++i]; }
        else if (!strcmp(argv[i], "--no-gpu"))                  { params.use_gpu = false; }
        else if (!strcmp(argv[i], "--fp8"))                     { use_fp8 = true; }
        else if (!strcmp(argv[i], "--pcs-fp8"))                 { use_pcs_fp8 = true; }
        else if (!strcmp(argv[i], "--trt-onnx-dir") && i + 1 < argc) {
            // Programmatic TRT config (no env vars): expects the standard
            // export names sam3_encoder.onnx / sam3_pcs.onnx / sam3_pvs.onnx.
            std::string d = argv[++i];
            params.trt.enabled = true;
            params.trt.encoder_onnx = d + "/sam3_encoder.onnx";
            params.trt.encoder_onnx_fp8 = d + "/sam3_encoder_fp8.onnx";
            params.trt.pcs_onnx = d + "/sam3_pcs.onnx";
            params.trt.pcs_onnx_fp8 = d + "/sam3_pcs_fp8.onnx";
            params.trt.pvs_onnx = d + "/sam3_pvs.onnx";
        }
        else if (!strcmp(argv[i], "--trt-cache-dir") && i + 1 < argc) { params.trt.cache_dir = argv[++i]; }
        else if (!strcmp(argv[i], "--runs") && i + 1 < argc)    { n_runs = atoi(argv[++i]); if (n_runs < 1) n_runs = 1; }
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) { params.n_threads = atoi(argv[++i]); }
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (params.model_path.empty() || image_path.empty()) {
        fprintf(stderr, "usage: %s --model m.ggml --image i.jpg [--text t] [--point-x x] [--point-y y]\n"
                        "       [--points \"x,y;x,y\"] [--neg-points \"x,y;...\"] [--box x0,y0,x1,y1] [--out dir]\n",
                argv[0]);
        return 1;
    }

    auto model = sam3_load_model(params);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }

    auto state = sam3_create_state(*model, params);
    if (use_fp8) sam3_set_encoder_fp8(*state, true);
    if (use_pcs_fp8) sam3_set_pcs_fp8(*state, true);
    sam3_image image = sam3_load_image(image_path);
    if (!state || image.data.empty()) { fprintf(stderr, "state/image load failed\n"); return 1; }

    for (int i = 0; i < n_runs; ++i) {
        auto t_enc0 = std::chrono::high_resolution_clock::now();
        if (!sam3_encode_image(*state, *model, image)) { fprintf(stderr, "encode failed\n"); return 1; }
        double enc_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_enc0).count();
        fprintf(stderr, "image encoded (%dx%d) call %d in %.2f ms%s\n", image.width, image.height, i, enc_ms,
                i == 0 ? " (cold: includes one-time engine load)" : " (warm)");
    }

    mkdir(out_dir.c_str(), 0755);

    // ── Optional: capture a concept embedding from THIS image's first
    // positive exemplar box, save to file, exit (reference-image workflow;
    // apply on another image via --use-embedding). ──
    if (!save_embedding_path.empty()) {
        if (pos_ex_px.empty()) { fprintf(stderr, "--save-embedding needs --pos-exemplars\n"); return 1; }
        sam3_box nb = {pos_ex_px[0].x0 / image.width, pos_ex_px[0].y0 / image.height,
                       pos_ex_px[0].x1 / image.width, pos_ex_px[0].y1 / image.height};
        auto emb = sam3_pcs_compute_exemplar_embedding(*state, *model, nb, /*positive=*/true);
        if (emb.empty()) { fprintf(stderr, "embedding capture failed\n"); return 1; }
        FILE* f = fopen(save_embedding_path.c_str(), "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", save_embedding_path.c_str()); return 1; }
        fwrite(emb.data(), sizeof(float), emb.size(), f);
        fclose(f);
        fprintf(stderr, "saved %zu-float concept embedding to %s\n", emb.size(), save_embedding_path.c_str());
        return 0;
    }

    // ── PCS -- run 3x in-process: 1st call pays one-time TensorRT engine
    // deserialization (a real server pays this once at startup, not per
    // request), 2nd/3rd calls reflect steady-state per-request latency. ──
    sam3_pcs_params pcs;
    pcs.text_prompt = text;
    for (const auto& b : pos_ex_px)
        pcs.pos_exemplars.push_back({b.x0 / image.width, b.y0 / image.height,
                                     b.x1 / image.width, b.y1 / image.height});
    for (const auto& b : neg_ex_px)
        pcs.neg_exemplars.push_back({b.x0 / image.width, b.y0 / image.height,
                                     b.x1 / image.width, b.y1 / image.height});
    if (!use_embedding_path.empty()) {
        FILE* f = fopen(use_embedding_path.c_str(), "rb");
        if (!f) { fprintf(stderr, "cannot read %s\n", use_embedding_path.c_str()); return 1; }
        std::vector<float> emb(256);
        size_t got = fread(emb.data(), sizeof(float), emb.size(), f);
        fclose(f);
        if (got != emb.size()) { fprintf(stderr, "bad embedding file (%zu floats)\n", got); return 1; }
        pcs.exemplar_embeddings.push_back(std::move(emb));
        fprintf(stderr, "loaded concept embedding from %s\n", use_embedding_path.c_str());
    }
    sam3_result pcs_result;
    for (int i = 0; i < n_runs; ++i) {
        auto t_pcs0 = std::chrono::high_resolution_clock::now();
        pcs_result = sam3_segment_pcs(*state, *model, pcs);
        double pcs_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_pcs0).count();
        fprintf(stderr, "PCS ('%s') call %d: %zu detections in %.2f ms%s\n",
                text.c_str(), i, pcs_result.detections.size(), pcs_ms,
                i == 0 ? " (cold: includes one-time engine load)" : " (warm)");
    }
    {
        std::string path = out_dir + "/pcs_detections.txt";
        FILE* f = fopen(path.c_str(), "w");
        for (auto& d : pcs_result.detections) {
            long mask_sum = 0;
            for (auto b : d.mask.data) mask_sum += b ? 1 : 0;
            fprintf(f, "score=%.6f box=%.2f,%.2f,%.2f,%.2f mask_px=%ld\n",
                    d.score, d.box.x0, d.box.y0, d.box.x1, d.box.y1, mask_sum);
        }
        fclose(f);
        fprintf(stderr, "wrote %s\n", path.c_str());
    }

    // ── PVS -- same cold/warm loop as PCS ──
    sam3_pvs_params pvs;
    if (!pos_points.empty() || !neg_points.empty() || use_box) {
        pvs.pos_points = pos_points;
        pvs.neg_points = neg_points;
        if (use_box) { pvs.use_box = true; pvs.box = {box[0], box[1], box[2], box[3]}; }
    } else {
        pvs.pos_points.push_back({point_x, point_y});
    }
    sam3_result pvs_result;
    for (int i = 0; i < n_runs; ++i) {
        auto t_pvs0 = std::chrono::high_resolution_clock::now();
        pvs_result = sam3_segment_pvs(*state, *model, pvs);
        double pvs_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_pvs0).count();
        fprintf(stderr, "PVS (%zu pos, %zu neg, box=%d) call %d: %zu detections in %.2f ms%s\n",
                pvs.pos_points.size(), pvs.neg_points.size(), (int)pvs.use_box,
                i, pvs_result.detections.size(), pvs_ms,
                i == 0 ? " (cold: includes one-time engine load)" : " (warm)");
    }
    {
        std::string path = out_dir + "/pvs_detections.txt";
        FILE* f = fopen(path.c_str(), "w");
        for (auto& d : pvs_result.detections) {
            long mask_sum = 0;
            for (auto b : d.mask.data) mask_sum += b ? 1 : 0;
            fprintf(f, "iou=%.6f obj_score=%.6f box=%.2f,%.2f,%.2f,%.2f mask_px=%ld\n",
                    d.iou_score, d.mask.obj_score, d.box.x0, d.box.y0, d.box.x1, d.box.y1, mask_sum);
        }
        fclose(f);
        fprintf(stderr, "wrote %s\n", path.c_str());
    }

    return 0;
}
