#!/usr/bin/env python3
"""Author a static ONNX graph for SAM3's image encoder (ViT backbone +
SimpleFPN neck) from the raw weights dumped by `sam3_dump_encoder_weights`
(examples/dump_encoder_weights.cpp).

Offline, one-time conversion tool -- part of the TensorRT image-encoder
migration (see docs/sam3/PLAN.md). Never runs at serving time.

Why hand-authored via onnx.helper (not torch.onnx.export): there is no
PyTorch module for this forward pass anywhere in this codebase to trace --
the only validated implementation is the ggml graph-builder code itself
(sam3_build_vit_prefix_graph / sam3_vit_block_forward / sam3_build_neck_graph
in sam3.cpp). This script is a direct, op-for-op transcription of that ggml
graph into ONNX ops, using the *same* weight file the ggml path uses (no
gated PyTorch checkpoint involved). Shared graph-authoring helpers live in
sam3_onnx_common.py (also used by the text/point-head conversion scripts).

Usage:
    python3 convert_sam3_encoder_to_onnx.py --export-dir encoder_export --out sam3_encoder.onnx
    python3 convert_sam3_encoder_to_onnx.py --export-dir encoder_export --out sam3_encoder.onnx --check
"""
import argparse
import os

import numpy as np
from onnx import TensorProto, helper

from sam3_onnx_common import (
    Manifest, GraphBuilder, add, build_model, gelu_erf, layer_norm, linear, matmul, mul,
    reshape, slice_step, split_n, sub, transpose, unsqueeze,
)

# ── SAM3 ViT image-encoder hyperparameters (native config; see sam3_hparams
#    in sam3.cpp). Phase 1 explicitly targets this fixed resolution only. ──
IMG_SIZE = 1008
PATCH_SIZE = 14
E = 1024
GRID = IMG_SIZE // PATCH_SIZE          # 72
N_TOK = GRID * GRID                    # 5184
NUM_HEADS = 16
HEAD_DIM = E // NUM_HEADS              # 64
HALF = HEAD_DIM // 2                   # 32
MLP_DIM = 4736
DEPTH = 32
WINDOW = 24
GLOBAL_IDX = {7, 15, 23, 31}
NECK_DIM = 256
LN_EPS = 1e-5


def apply_rope(g, x, cos_init, sin_init, batch, n_tok, name):
    # x: (batch,heads,N,head_dim). Interleaved pairing (matches
    # sam3_apply_rope's [2,half,...] reshape: pair i = (x[2i], x[2i+1])).
    # cos_init/sin_init: (1,1,N,half) constants, broadcast over batch/heads.
    x_re = slice_step(g, x, axis=3, start=0, step=2, name=name + "_re")   # (batch,heads,N,half)
    x_im = slice_step(g, x, axis=3, start=1, step=2, name=name + "_im")
    out_re = sub(g, mul(g, x_re, cos_init, name=name + "_recos"),
                 mul(g, x_im, sin_init, name=name + "_imsin"), name=name + "_outre")
    out_im = add(g, mul(g, x_re, sin_init, name=name + "_resin"),
                 mul(g, x_im, cos_init, name=name + "_imcos"), name=name + "_outim")
    re_u = unsqueeze(g, out_re, [4], name=name + "_reu")   # (batch,heads,N,half,1)
    im_u = unsqueeze(g, out_im, [4], name=name + "_imu")
    stacked = g.node("Concat", [re_u, im_u], name=name + "_stack", axis=4)  # (batch,heads,N,half,2)
    return reshape(g, stacked, [batch, NUM_HEADS, n_tok, HEAD_DIM], name=name + "_merge")


def window_partition(g, x, name):
    npx = npy = GRID // WINDOW  # 3
    x = reshape(g, x, [1, GRID, GRID, E], name=name + "_spatial")
    x = reshape(g, x, [1, npy, WINDOW, npx, WINDOW, E], name=name + "_split")
    x = transpose(g, x, [0, 1, 3, 2, 4, 5], name=name + "_perm")
    x = reshape(g, x, [npy * npx, WINDOW * WINDOW, E], name=name + "_windows")
    return x


def window_unpartition(g, x, name):
    npx = npy = GRID // WINDOW
    x = reshape(g, x, [1, npy, npx, WINDOW, WINDOW, E], name=name + "_split")
    x = transpose(g, x, [0, 1, 3, 2, 4, 5], name=name + "_perm")
    x = reshape(g, x, [1, GRID, GRID, E], name=name + "_spatial")
    x = reshape(g, x, [1, N_TOK, E], name=name + "_seq")
    return x


def attention(g, x, w_qkv, b_qkv, w_proj, b_proj, cos_init, sin_init, batch, n_tok, name):
    qkv = linear(g, x, w_qkv, b_qkv, name=name + "_qkv")  # (batch,N,3E)
    q, k, v = split_n(g, qkv, [E, E, E], axis=2, name=name + "_qkvsplit")

    def to_heads(t, nm):
        t = reshape(g, t, [batch, n_tok, NUM_HEADS, HEAD_DIM], name=nm + "_r")
        return transpose(g, t, [0, 2, 1, 3], name=nm + "_t")  # (batch,heads,N,head_dim)

    q = to_heads(q, name + "_q")
    k = to_heads(k, name + "_k")
    v = to_heads(v, name + "_v")

    q = apply_rope(g, q, cos_init, sin_init, batch, n_tok, name=name + "_roq")
    k = apply_rope(g, k, cos_init, sin_init, batch, n_tok, name=name + "_rok")

    kt = transpose(g, k, [0, 1, 3, 2], name=name + "_kt")           # (batch,heads,head_dim,N)
    scores = matmul(g, q, kt, name=name + "_scores")                # (batch,heads,N,N)
    scale_c = g.const(name + "_scale", np.float32(1.0 / np.sqrt(HEAD_DIM)))
    scores = mul(g, scores, scale_c, name=name + "_scaled")
    probs = g.node("Softmax", [scores], name=name + "_softmax", axis=-1)
    out = matmul(g, probs, v, name=name + "_attnv")                 # (batch,heads,N,head_dim)
    out = transpose(g, out, [0, 2, 1, 3], name=name + "_backt")     # (batch,N,heads,head_dim)
    out = reshape(g, out, [batch, n_tok, E], name=name + "_merge")
    out = linear(g, out, w_proj, b_proj, name=name + "_proj")
    return out


def vit_block(g, x, mf, block_idx, name):
    # Direct transcription of sam3_vit_block_forward: pre-norm -> attn
    # (windowed or global, with RoPE) -> residual -> pre-norm -> MLP -> residual.
    is_global = block_idx in GLOBAL_IDX
    n_tok = N_TOK if is_global else WINDOW * WINDOW
    batch = 1 if is_global else (GRID // WINDOW) ** 2

    p = f"vit.blocks.{block_idx}."
    norm1_w = g.const(name + "_norm1_w", mf.bias(p + "norm1.weight"))
    norm1_b = g.const(name + "_norm1_b", mf.bias(p + "norm1.bias"))
    norm2_w = g.const(name + "_norm2_w", mf.bias(p + "norm2.weight"))
    norm2_b = g.const(name + "_norm2_b", mf.bias(p + "norm2.bias"))
    qkv_w = g.const(name + "_qkv_w", mf.linear_w(p + "attn.qkv.weight"))
    qkv_b = g.const(name + "_qkv_b", mf.bias(p + "attn.qkv.bias"))
    proj_w = g.const(name + "_proj_w", mf.linear_w(p + "attn.proj.weight"))
    proj_b = g.const(name + "_proj_b", mf.bias(p + "attn.proj.bias"))
    fc1_w = g.const(name + "_fc1_w", mf.linear_w(p + "mlp.lin1.weight"))
    fc1_b = g.const(name + "_fc1_b", mf.bias(p + "mlp.lin1.bias"))
    fc2_w = g.const(name + "_fc2_w", mf.linear_w(p + "mlp.lin2.weight"))
    fc2_b = g.const(name + "_fc2_b", mf.bias(p + "mlp.lin2.bias"))

    cos_np, sin_np = mf.freqs(p + "attn.freqs_cis")  # each (n_tok, HALF)
    cos = g.const(name + "_cos", cos_np.reshape(1, 1, n_tok, HALF))
    sin = g.const(name + "_sin", sin_np.reshape(1, 1, n_tok, HALF))

    shortcut = x
    xn = layer_norm(g, x, norm1_w, norm1_b, LN_EPS, name=name + "_ln1")
    if not is_global:
        xn = window_partition(g, xn, name=name + "_winpart")
    attn_out = attention(g, xn, qkv_w, qkv_b, proj_w, proj_b, cos, sin, batch, n_tok, name=name + "_attn")
    if not is_global:
        attn_out = window_unpartition(g, attn_out, name=name + "_winunpart")
    x = add(g, shortcut, attn_out, name=name + "_res1")

    shortcut = x
    xn = layer_norm(g, x, norm2_w, norm2_b, LN_EPS, name=name + "_ln2")
    h = linear(g, xn, fc1_w, fc1_b, name=name + "_fc1")
    h = gelu_erf(g, h, name=name + "_gelu")
    h = linear(g, h, fc2_w, fc2_b, name=name + "_fc2")
    x = add(g, shortcut, h, name=name + "_res2")
    return x


def neck_scale(g, x_nchw, mf, prefix, scale_idx, name):
    # Direct transcription of one scale branch of sam3_build_neck_graph.
    def conv_w(nm):
        return g.const(name + "_" + nm + "_w", mf.conv_w(prefix + nm + ".weight"))

    def deconv_w(nm):
        return g.const(name + "_" + nm + "_w", mf.deconv_w(prefix + nm + ".weight"))

    def bias(nm):
        return g.const(name + "_" + nm + "_b", mf.bias(prefix + nm + ".bias"))

    if scale_idx == 0:
        s = g.node("ConvTranspose", [x_nchw, deconv_w("dconv_2x2_0"), bias("dconv_2x2_0")],
                   name=name + "_dc1", kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0])
        s = gelu_erf(g, s, name=name + "_gelu")
        s = g.node("ConvTranspose", [s, deconv_w("dconv_2x2_1"), bias("dconv_2x2_1")],
                   name=name + "_dc2", kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0])
    elif scale_idx == 1:
        s = g.node("ConvTranspose", [x_nchw, deconv_w("dconv_2x2"), bias("dconv_2x2")],
                   name=name + "_dc", kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0])
    elif scale_idx == 2:
        s = x_nchw
    elif scale_idx == 3:
        s = g.node("MaxPool", [x_nchw], name=name + "_pool",
                   kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0])
    else:
        raise ValueError(scale_idx)

    s = g.node("Conv", [s, conv_w("conv_1x1"), bias("conv_1x1")],
               name=name + "_1x1", kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0])
    s = g.node("Conv", [s, conv_w("conv_3x3"), bias("conv_3x3")],
               name=name + "_3x3", kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
    # NCHW -> NHWC, matching the ggml state tensors' [C,W,H,B] layout
    # (reversed = [B,H,W,C] = NHWC) that the C++ runtime wrapper writes into.
    s = transpose(g, s, [0, 2, 3, 1], name=name + "_nhwc")
    return s


# Output NHWC shapes, in the order sam3_build_neck_graph documents them.
NECK_OUT_HW = {0: GRID * 4, 1: GRID * 2, 2: GRID, 3: GRID // 2}


def build_graph(mf):
    g = GraphBuilder()

    patch_w = g.const("patch_embed_w", mf.conv_w("vit.patch_embed.proj.weight"))
    x = g.node("Conv", ["image", patch_w], name="patch_embed",
               kernel_shape=[PATCH_SIZE, PATCH_SIZE], strides=[PATCH_SIZE, PATCH_SIZE], pads=[0, 0, 0, 0])
    # (1,E,72,72) NCHW -> (1,72,72,E) NHWC -> (1,5184,E) sequence, token n = h*72+w
    x = transpose(g, x, [0, 2, 3, 1], name="patch_nhwc")
    x = reshape(g, x, [1, N_TOK, E], name="patch_seq")

    # pos_embed: ggml [E,24,24,1] -> reversed (1,24,24,E) -> squeeze -> (24,24,E);
    # tile 3x3 (Hiera pretrained resolution -> native 72x72) and flatten to a
    # sequence in Python (avoids replicating ggml_repeat's broadcast in ONNX).
    pos = mf.raw("vit.pos_embed").reshape(24, 24, E)
    pos_tiled = np.tile(pos, (3, 3, 1)).reshape(1, N_TOK, E)
    pos_const = g.const("pos_embed_tiled", pos_tiled)
    x = add(g, x, pos_const, name="add_pos")

    ln_pre_w = g.const("ln_pre_w", mf.bias("vit.ln_pre.weight"))
    ln_pre_b = g.const("ln_pre_b", mf.bias("vit.ln_pre.bias"))
    x = layer_norm(g, x, ln_pre_w, ln_pre_b, LN_EPS, name="ln_pre")

    for i in range(DEPTH):
        x = vit_block(g, x, mf, i, name=f"block{i}")

    vit_spatial = reshape(g, x, [1, GRID, GRID, E], name="vit_spatial")  # NHWC, matches ggml vit_output
    g.alias(vit_spatial, "vit_output")

    x_nchw = transpose(g, vit_spatial, [0, 3, 1, 2], name="neck_input_nchw")

    has_det = mf.has("neck.det.0.conv_1x1.weight")
    necks = [("neck.trk.", "neck_trk")]
    if has_det:
        necks.append(("neck.det.", "neck_det"))
    output_names = ["vit_output"]
    for prefix, tag in necks:
        for scale in range(4):
            out = neck_scale(g, x_nchw, mf, f"{prefix}{scale}.", scale, name=f"{tag}{scale}")
            out_name = f"{tag}_{scale}"
            g.alias(out, out_name)
            output_names.append(out_name)

    graph_inputs = [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, IMG_SIZE, IMG_SIZE])]
    graph_outputs = [helper.make_tensor_value_info("vit_output", TensorProto.FLOAT, [1, GRID, GRID, E])]
    for prefix, tag in necks:
        for scale in range(4):
            hw = NECK_OUT_HW[scale]
            graph_outputs.append(helper.make_tensor_value_info(f"{tag}_{scale}", TensorProto.FLOAT,
                                                                 [1, hw, hw, NECK_DIM]))

    model_def = build_model(g.nodes, g.initializers, graph_inputs, graph_outputs, "sam3_encoder")
    return model_def, output_names


def run_onnxruntime_check(onnx_path, output_names, seed=0):
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    image = (rng.standard_normal((1, 3, IMG_SIZE, IMG_SIZE)) * 0.25).astype(np.float32)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    outputs = sess.run(output_names, {"image": image})

    print(f"onnxruntime CPU check ({len(output_names)} outputs):")
    for name, arr in zip(output_names, outputs):
        finite = np.isfinite(arr).all()
        print(f"  {name:16s} shape={arr.shape} finite={finite} "
              f"mean={arr.mean():.5f} std={arr.std():.5f} min={arr.min():.5f} max={arr.max():.5f}")
        if not finite:
            raise SystemExit(f"error: non-finite values in output '{name}'")
    return image, dict(zip(output_names, outputs))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-dir", required=True, help="dir written by sam3_dump_encoder_weights")
    ap.add_argument("--out", required=True, help="output .onnx path")
    ap.add_argument("--check", action="store_true", help="run an onnxruntime CPU sanity check after export")
    ap.add_argument("--dump-ref", help="if set with --check, write the random input + outputs as "
                                        "<name>.bin/.shape (load_ref_f32-compatible) into this directory")
    args = ap.parse_args()

    mf = Manifest(args.export_dir)
    model_def, output_names = build_graph(mf)

    import onnx
    onnx.save(model_def, args.out)
    print(f"saved {args.out} ({len(model_def.graph.node)} nodes, "
          f"{len(model_def.graph.initializer)} initializers)")

    if args.check:
        image, outputs = run_onnxruntime_check(args.out, output_names)
        if args.dump_ref:
            os.makedirs(args.dump_ref, exist_ok=True)

            def dump(name, arr):
                arr = np.ascontiguousarray(arr, dtype=np.float32)
                arr.tofile(os.path.join(args.dump_ref, name + ".bin"))
                with open(os.path.join(args.dump_ref, name + ".shape"), "w") as f:
                    f.write(",".join(str(d) for d in arr.shape))

            dump("preprocessed", image[0])  # (3,1008,1008), matches sam3_encode_image_from_preprocessed's chw_data
            for name, arr in outputs.items():
                dump(name, arr[0])  # drop batch dim -> matches ggml's dumped (H,W,C)/(W,H,C) rank
            print(f"wrote reference dumps to {args.dump_ref}")


if __name__ == "__main__":
    main()
