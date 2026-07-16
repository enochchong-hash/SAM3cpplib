#!/usr/bin/env python3
"""Author an ONNX graph for SAM3's PVS (Point-Visual-prompted Segmentation)
pipeline: SAM two-way transformer mask decoder -> hypernetwork mask
prediction + upscaling -> IoU / object-score heads.

Offline, one-time conversion tool -- part of the TensorRT migration (see
docs/sam3/PLAN.md). Never runs at serving time. Direct transcription of
sam3_build_sam_dec_graph / sam3_twoway_block_forward / sam3_sam_attention in
sam3.cpp.

The sparse-token dimension is DYNAMIC (`sparse` is (1,N,256) with symbolic
N), so one engine serves every prompt shape sam3_collect_pvs_prompt_tokens
can produce: any number of positive/negative points, a box (2 corner tokens,
labels 2/3), combinations, plus the always-appended padding token. The
special tokens (obj_score, iou, 4 mask tokens) sit at positions 0-5, sparse
tokens append AFTER them, so every output-token slice is N-independent; only
the token-side attention reshapes carry the dynamic dim (Reshape -1). The
TensorRT builder needs an optimization profile for N (handled by
sam3_trt_build_or_load_engine; serving uses N in [1,16]).

The prompt encoder's per-request sparse embedding (`sam3_pe_encode_coord` +
learned label embeddings) is genuinely per-request (real point coordinates)
and stays exactly as-is on the C++ host side, unchanged -- it's fed into this
graph as the `sparse` input, opaque to this script. Everything that's a pure
function of fixed weights (the dense positional-encoding grid, the
"no mask prompt" tiled embedding, no_mem_embed) is baked as a constant here.

Runtime inputs: `sparse` (1,N,256, built by unchanged host C++ code) and the
image encoder's neck_trk[0]/[1]/[2] outputs (`feat_s0`/`feat_s1`/`image_feats`,
NHWC -- exactly what Phase 1's TensorRT image encoder produces).
"""
import argparse

import numpy as np
from onnx import TensorProto, helper

from sam3_onnx_common import (
    Manifest, GraphBuilder, add, build_model, concat, gelu_erf, layer_norm,
    linear, matmul, merge_heads, mha, mul, reshape, slice_step, split_heads,
    sub, transpose,
)

D = 256
H = 72
N_IMG = H * H
FPN_HW = {0: H * 4, 1: H * 2}  # 288, 144
DEC_DEPTH = 2
N_SPECIAL = 6                  # obj_score, iou, 4 mask tokens
# Token count is DYNAMIC: N_TOK = N_SPECIAL + n_sparse, with n_sparse a
# symbolic input dim. `None` is the in-script marker for "dynamic token dim"
# understood by split_heads/merge_heads (Reshape -1).
N_TOK = None

LN_EPS = 1e-5
LN2D_EPS = 1e-6

SA_HEADS, SA_HEAD_DIM = 8, D // 8          # self-attn: internal_dim=256
CA_HEADS, CA_INTERNAL, CA_HEAD_DIM = 8, 128, 128 // 8  # cross-attns: internal_dim=128


def mlp3(g, x, mf, prefix, name, sigmoid_last=False):
    w0 = g.const(name + "_w0", mf.linear_w(prefix + "layers.0.weight"))
    b0 = g.const(name + "_b0", mf.bias(prefix + "layers.0.bias"))
    w1 = g.const(name + "_w1", mf.linear_w(prefix + "layers.1.weight"))
    b1 = g.const(name + "_b1", mf.bias(prefix + "layers.1.bias"))
    w2 = g.const(name + "_w2", mf.linear_w(prefix + "layers.2.weight"))
    b2 = g.const(name + "_b2", mf.bias(prefix + "layers.2.bias"))
    x = g.node("Relu", [linear(g, x, w0, b0, name=name + "_l0")], name=name + "_r0")
    x = g.node("Relu", [linear(g, x, w1, b1, name=name + "_l1")], name=name + "_r1")
    x = linear(g, x, w2, b2, name=name + "_l2")
    if sigmoid_last:
        x = g.node("Sigmoid", [x], name=name + "_sig")
    return x


def sam_attention(g, q_src, k_src, v_src, mf, prefix, internal_dim, heads, head_dim,
                  batch, n_q, n_kv, name):
    """Separate (non-fused) Q/K/V/Out weight matrices -- sam3_sam_attention's
    convention, unlike PCS's fused in_proj_weight."""
    qw = g.const(name + "_qw", mf.linear_w(prefix + "q_proj.weight"))
    qb = g.const(name + "_qb", mf.bias(prefix + "q_proj.bias"))
    kw = g.const(name + "_kw", mf.linear_w(prefix + "k_proj.weight"))
    kb = g.const(name + "_kb", mf.bias(prefix + "k_proj.bias"))
    vw = g.const(name + "_vw", mf.linear_w(prefix + "v_proj.weight"))
    vb = g.const(name + "_vb", mf.bias(prefix + "v_proj.bias"))
    ow = g.const(name + "_ow", mf.linear_w(prefix + "out_proj.weight"))
    ob = g.const(name + "_ob", mf.bias(prefix + "out_proj.bias"))

    q = linear(g, q_src, qw, qb, name=name + "_q")
    k = linear(g, k_src, kw, kb, name=name + "_k")
    v = linear(g, v_src, vw, vb, name=name + "_v")
    q = split_heads(g, q, batch, n_q, heads, head_dim, name=name + "_qh")
    k = split_heads(g, k, batch, n_kv, heads, head_dim, name=name + "_kh")
    v = split_heads(g, v, batch, n_kv, heads, head_dim, name=name + "_vh")
    attn = mha(g, q, k, v, heads, head_dim, name=name + "_mha")
    merged = merge_heads(g, attn, batch, n_q, internal_dim, name=name + "_merge")
    return linear(g, merged, ow, ob, name=name + "_out")


def twoway_block(g, queries, keys, query_pos, key_pe, mf, i, name):
    p = f"sam_dec.twoway.{i}."
    n1w, n1b = g.const(name + "_n1w", mf.bias(p + "norm1.weight")), g.const(name + "_n1b", mf.bias(p + "norm1.bias"))
    n2w, n2b = g.const(name + "_n2w", mf.bias(p + "norm2.weight")), g.const(name + "_n2b", mf.bias(p + "norm2.bias"))
    n3w, n3b = g.const(name + "_n3w", mf.bias(p + "norm3.weight")), g.const(name + "_n3b", mf.bias(p + "norm3.bias"))
    n4w, n4b = g.const(name + "_n4w", mf.bias(p + "norm4.weight")), g.const(name + "_n4b", mf.bias(p + "norm4.bias"))
    l1w, l1b = g.const(name + "_l1w", mf.linear_w(p + "mlp.lin1.weight")), g.const(name + "_l1b", mf.bias(p + "mlp.lin1.bias"))
    l2w, l2b = g.const(name + "_l2w", mf.linear_w(p + "mlp.lin2.weight")), g.const(name + "_l2b", mf.bias(p + "mlp.lin2.bias"))

    # 1. Self-attention. Layer 0 (skip_first_layer_pe): no pos, no residual.
    if i == 0:
        queries = sam_attention(g, queries, queries, queries, mf, p + "sa.", D, SA_HEADS, SA_HEAD_DIM,
                                1, N_TOK, N_TOK, name=name + "_sa")
    else:
        q_in = add(g, queries, query_pos, name=name + "_saqin")
        sa_out = sam_attention(g, q_in, q_in, queries, mf, p + "sa.", D, SA_HEADS, SA_HEAD_DIM,
                               1, N_TOK, N_TOK, name=name + "_sa")
        queries = add(g, queries, sa_out, name=name + "_sares")
    queries = layer_norm(g, queries, n1w, n1b, LN_EPS, name=name + "_ln1")

    # 2. Cross-attn: tokens attend to image. Q=queries+pos, K=keys+pe, V=keys (no pe).
    q_in = add(g, queries, query_pos, name=name + "_ca1qin")
    k_in = add(g, keys, key_pe, name=name + "_ca1kin")
    ca_out = sam_attention(g, q_in, k_in, keys, mf, p + "cross_attn_token_to_image.",
                           CA_INTERNAL, CA_HEADS, CA_HEAD_DIM, 1, N_TOK, N_IMG, name=name + "_ca1")
    queries = add(g, queries, ca_out, name=name + "_ca1res")
    queries = layer_norm(g, queries, n2w, n2b, LN_EPS, name=name + "_ln2")

    # 3. MLP (ReLU).
    mlp = linear(g, queries, l1w, l1b, name=name + "_mlp1")
    mlp = g.node("Relu", [mlp], name=name + "_mlprelu")
    mlp = linear(g, mlp, l2w, l2b, name=name + "_mlp2")
    queries = add(g, queries, mlp, name=name + "_mlpres")
    queries = layer_norm(g, queries, n3w, n3b, LN_EPS, name=name + "_ln3")

    # 4. Cross-attn: image attends to tokens (Q/K roles swapped vs step 2).
    q_in2 = add(g, queries, query_pos, name=name + "_ca2qin")
    k_in2 = add(g, keys, key_pe, name=name + "_ca2kin")
    ca_out2 = sam_attention(g, k_in2, q_in2, queries, mf, p + "cross_attn_image_to_token.",
                            CA_INTERNAL, CA_HEADS, CA_HEAD_DIM, 1, N_IMG, N_TOK, name=name + "_ca2")
    keys = add(g, keys, ca_out2, name=name + "_ca2res")
    keys = layer_norm(g, keys, n4w, n4b, LN_EPS, name=name + "_ln4")

    return queries, keys


def group_norm2d(g, x_nchw, weight, bias, num_channels, h, w, eps, name):
    """LayerNorm2d (per-sample, all-channels-and-spatial norm per SAM's
    LayerNorm2d -- actually normalizes over the channel axis ONLY, per
    location, matching SAM's `u = x.mean(1); s=(x-u).pow(2).mean(1)` -- i.e.
    channel-wise LayerNorm at each spatial location, NOT GroupNorm)."""
    mean = g.node("ReduceMean", [x_nchw], name=name + "_mean", axes=[1], keepdims=1)
    centered = sub(g, x_nchw, mean, name=name + "_centered")
    sq = mul(g, centered, centered, name=name + "_sq")
    var = g.node("ReduceMean", [sq], name=name + "_var", axes=[1], keepdims=1)
    eps_c = g.const(name + "_eps", np.float32(eps))
    std = g.node("Sqrt", [add(g, var, eps_c, name=name + "_vareps")], name=name + "_std")
    normed = g.node("Div", [centered, std], name=name + "_normed")
    w_r = reshape(g, weight, [1, num_channels, 1, 1], name=name + "_wr")
    b_r = reshape(g, bias, [1, num_channels, 1, 1], name=name + "_br")
    return add(g, mul(g, normed, w_r, name=name + "_scaled"), b_r, name=name + "_biased")


def build_graph(mf):
    g = GraphBuilder()

    sparse = "sparse"                 # (1,8,256), host-built, per-request
    image_feats_raw = "image_feats"   # (1,72,72,256) NHWC = neck_trk[2], no no_mem_embed yet
    feat_s0_nhwc = "feat_s0"          # (1,288,288,256) NHWC = neck_trk[0]
    feat_s1_nhwc = "feat_s1"          # (1,144,144,256) NHWC = neck_trk[1]

    # ── Baked constants (pure functions of fixed weights/hyperparameters) ──
    no_mem_embed = g.const("no_mem_embed_const", mf.raw("no_mem_embed").reshape(1, 1, D))

    pe_gauss = mf.raw("sam_pe.pe_gaussian").reshape(2, 128)  # NOT reversed -- see Manifest.freqs-style exception
    yy, xx = np.meshgrid(np.arange(H), np.arange(H), indexing="ij")  # yy=row, xx=col
    x_norm = (xx + 0.5) / H
    y_norm = (yy + 0.5) / H
    cx = 2.0 * x_norm - 1.0
    cy = 2.0 * y_norm - 1.0
    dot = (cx[..., None] * pe_gauss[0] + cy[..., None] * pe_gauss[1]) * (2.0 * np.pi)  # (H,H,128)
    dense_pe_np = np.concatenate([np.sin(dot), np.cos(dot)], axis=-1).astype(np.float32)  # (H,H,256)
    image_pe = g.const("image_pe_const", dense_pe_np.reshape(1, N_IMG, D))

    no_mask_embed = mf.bias("sam_pe.no_mask_embed.weight")  # (256,)
    dense_nomask_np = np.tile(no_mask_embed, (H, H, 1)).astype(np.float32)
    dense_emb = g.const("dense_emb_const", dense_nomask_np.reshape(1, N_IMG, D))

    obj_score_tok = g.const("obj_score_tok", mf.raw("sam_dec.obj_score_token.weight").reshape(1, 1, D))
    iou_tok = g.const("iou_tok", mf.raw("sam_dec.iou_token.weight").reshape(1, 1, D))
    mask_toks = g.const("mask_toks", mf.raw("sam_dec.mask_tokens.weight").reshape(1, 4, D))

    # ── Assemble initial tokens/image features ──────────────────────────
    output_tokens = concat(g, [obj_score_tok, iou_tok, mask_toks], axis=1, name="output_tokens")  # (1,6,256)
    tokens = concat(g, [output_tokens, sparse], axis=1, name="init_tokens")  # (1,8,256)
    query_pos = tokens  # reused unchanged at every layer (== initial concatenation)

    img_seq = reshape(g, image_feats_raw, [1, N_IMG, D], name="img_seq")
    img_with_mem = add(g, img_seq, no_mem_embed, name="img_with_mem")
    keys = add(g, img_with_mem, dense_emb, name="keys_init")  # (1,5184,256)
    key_pe = image_pe

    queries = tokens
    for i in range(DEC_DEPTH):
        queries, keys = twoway_block(g, queries, keys, query_pos, key_pe, mf, i, name=f"twoway{i}")

    # Final attention: tokens attend to image only (no reverse, no further norm on keys).
    q_in = add(g, queries, query_pos, name="final_qin")
    k_in = add(g, keys, key_pe, name="final_kin")
    final_out = sam_attention(g, q_in, k_in, keys, mf, "sam_dec.final_attn.",
                              CA_INTERNAL, CA_HEADS, CA_HEAD_DIM, 1, N_TOK, N_IMG, name="final_attn")
    queries = add(g, queries, final_out, name="final_res")
    fnw, fnb = g.const("final_norm_w", mf.bias("sam_dec.final_norm.weight")), g.const("final_norm_b", mf.bias("sam_dec.final_norm.bias"))
    queries = layer_norm(g, queries, fnw, fnb, LN_EPS, name="final_norm")

    # ── Output-token extraction (s=1, has_obj_score always true for SAM3) ──
    obj_in = slice_step(g, queries, axis=1, start=0, step=1, end=1, name="obj_in")        # (1,1,256)
    iou_tok_out = slice_step(g, queries, axis=1, start=1, step=1, end=2, name="iou_tok_out")
    mask_toks_out = slice_step(g, queries, axis=1, start=2, step=1, end=6, name="mask_toks_out")  # (1,4,256)
    sam_token = slice_step(g, queries, axis=1, start=2, step=1, end=3, name="sam_token_slice")    # (1,1,256)

    # ── Upscaling (verbatim SAM2 high-res-features branch) ──────────────
    src_img = reshape(g, keys, [1, H, H, D], name="src_img_nhwc")
    src_img_nchw = transpose(g, src_img, [0, 3, 1, 2], name="src_img_nchw")  # (1,256,72,72)

    up0_w = g.const("up0_w", mf.deconv_w("sam_dec.upscale.0.weight"))
    up0_b = g.const("up0_b", mf.bias("sam_dec.upscale.0.bias"))
    up1 = g.node("ConvTranspose", [src_img_nchw, up0_w, up0_b], name="up0",
                kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0])  # (1,64,144,144)

    hs1_w = g.const("hs1_w", mf.conv_w("sam_dec.conv_s1.weight"))
    hs1_b = g.const("hs1_b", mf.bias("sam_dec.conv_s1.bias"))
    feat_s1_nchw = transpose(g, feat_s1_nhwc, [0, 3, 1, 2], name="feat_s1_nchw")
    hs1 = g.node("Conv", [feat_s1_nchw, hs1_w, hs1_b], name="hs1", kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0])

    up1 = add(g, up1, hs1, name="up1_merged")
    ln1_w = g.const("upscale_ln1_w", mf.bias("sam_dec.upscale.1.weight"))
    ln1_b = g.const("upscale_ln1_b", mf.bias("sam_dec.upscale.1.bias"))
    up1 = group_norm2d(g, up1, ln1_w, ln1_b, 64, FPN_HW[1], FPN_HW[1], LN2D_EPS, name="upscale_ln1")
    up1 = gelu_erf(g, up1, name="upscale_gelu1")

    up2_w = g.const("up2_w", mf.deconv_w("sam_dec.upscale.3.weight"))
    up2_b = g.const("up2_b", mf.bias("sam_dec.upscale.3.bias"))
    up2 = g.node("ConvTranspose", [up1, up2_w, up2_b], name="up2",
                kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0])  # (1,32,288,288)

    hs0_w = g.const("hs0_w", mf.conv_w("sam_dec.conv_s0.weight"))
    hs0_b = g.const("hs0_b", mf.bias("sam_dec.conv_s0.bias"))
    feat_s0_nchw = transpose(g, feat_s0_nhwc, [0, 3, 1, 2], name="feat_s0_nchw")
    hs0 = g.node("Conv", [feat_s0_nchw, hs0_w, hs0_b], name="hs0", kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0])

    up2 = add(g, up2, hs0, name="up2_merged")  # NOTE: no LayerNorm2d here (matches sam3.cpp exactly)
    upscaled = gelu_erf(g, up2, name="upscale_gelu2")  # (1,32,288,288)

    upscaled_flat = reshape(g, upscaled, [1, 32, FPN_HW[0] * FPN_HW[0]], name="upscaled_flat")

    # ── Hypernetwork dot product (4 mask candidates) ─────────────────────
    mask_list = []
    for m in range(4):
        tok = slice_step(g, mask_toks_out, axis=1, start=m, step=1, end=m + 1, name=f"masktok{m}")  # (1,1,256)
        hyper = mlp3(g, tok, mf, f"sam_dec.hyper.{m}.", name=f"hyper{m}")  # (1,1,32)
        mlogit = matmul(g, hyper, upscaled_flat, name=f"mlogit{m}")  # (1,1,82944)
        mask_list.append(mlogit)
    masks = concat(g, mask_list, axis=1, name="masks_concat")  # (1,4,82944)
    masks = reshape(g, masks, [1, 4, FPN_HW[0], FPN_HW[0]], name="masks_final")

    iou_pred = mlp3(g, iou_tok_out, mf, "sam_dec.iou_prediction_head.", name="iou_head", sigmoid_last=True)  # (1,1,4)
    iou_pred = reshape(g, iou_pred, [1, 4], name="iou_pred_final")

    obj_score_logit = mlp3(g, obj_in, mf, "sam_dec.pred_obj_score_head.", name="obj_head")  # (1,1,1)
    obj_score_logit = reshape(g, obj_score_logit, [1, 1], name="obj_score_final")

    sam_token_out = reshape(g, sam_token, [1, D], name="sam_token_final")

    g.alias(masks, "masks")
    g.alias(iou_pred, "iou_pred")
    g.alias(obj_score_logit, "obj_score_logit")
    g.alias(sam_token_out, "sam_token")

    graph_inputs = [
        # Symbolic token dim: any number of sparse prompt tokens (points,
        # box corners, pad) -- see the module docstring.
        helper.make_tensor_value_info("sparse", TensorProto.FLOAT, [1, "n_sparse", D]),
        helper.make_tensor_value_info("image_feats", TensorProto.FLOAT, [1, H, H, D]),
        helper.make_tensor_value_info("feat_s0", TensorProto.FLOAT, [1, FPN_HW[0], FPN_HW[0], D]),
        helper.make_tensor_value_info("feat_s1", TensorProto.FLOAT, [1, FPN_HW[1], FPN_HW[1], D]),
    ]
    graph_outputs = [
        helper.make_tensor_value_info("masks", TensorProto.FLOAT, [1, 4, FPN_HW[0], FPN_HW[0]]),
        helper.make_tensor_value_info("iou_pred", TensorProto.FLOAT, [1, 4]),
        helper.make_tensor_value_info("obj_score_logit", TensorProto.FLOAT, [1, 1]),
        helper.make_tensor_value_info("sam_token", TensorProto.FLOAT, [1, D]),
    ]

    model_def = build_model(g.nodes, g.initializers, graph_inputs, graph_outputs, "sam3_pvs")
    return model_def, ["masks", "iou_pred", "obj_score_logit", "sam_token"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-dir", required=True, help="dir written by sam3_dump_pcs_pvs_weights")
    ap.add_argument("--out", required=True, help="output .onnx path")
    ap.add_argument("--check", action="store_true", help="run an onnxruntime CPU sanity check after export")
    args = ap.parse_args()

    mf = Manifest(args.export_dir)
    model_def, output_names = build_graph(mf)

    import onnx
    onnx.save(model_def, args.out)
    print(f"saved {args.out} ({len(model_def.graph.node)} nodes, {len(model_def.graph.initializer)} initializers)")

    if args.check:
        import onnxruntime as ort
        rng = np.random.default_rng(0)
        image_feats = (rng.standard_normal((1, H, H, D)) * 0.5).astype(np.float32)
        feat_s0 = (rng.standard_normal((1, FPN_HW[0], FPN_HW[0], D)) * 0.1).astype(np.float32)
        feat_s1 = (rng.standard_normal((1, FPN_HW[1], FPN_HW[1], D)) * 0.2).astype(np.float32)

        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        # Exercise the dynamic token dim: 2 = 1 point + pad (the old static
        # scope), 4 = box + point + pad, 9 = many points.
        for n_sparse in (2, 4, 9):
            sparse = (rng.standard_normal((1, n_sparse, D)) * 0.3).astype(np.float32)
            outputs = sess.run(output_names, {"sparse": sparse, "image_feats": image_feats,
                                              "feat_s0": feat_s0, "feat_s1": feat_s1})
            print(f"onnxruntime CPU check, n_sparse={n_sparse} ({len(output_names)} outputs):")
            for name, arr in zip(output_names, outputs):
                finite = np.isfinite(arr).all()
                print(f"  {name:16s} shape={arr.shape} finite={finite} mean={arr.mean():.5f} "
                      f"std={arr.std():.5f} min={arr.min():.5f} max={arr.max():.5f}")
                if not finite:
                    raise SystemExit(f"error: non-finite values in output '{name}'")


if __name__ == "__main__":
    main()
