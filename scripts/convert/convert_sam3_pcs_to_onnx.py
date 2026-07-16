#!/usr/bin/env python3
"""Author a static ONNX graph for SAM3's PCS (text-Prompted Concept
Segmentation) pipeline: text encoder -> geometry encoder -> fusion encoder
-> DETR decoder + scoring -> MaskFormer segmentation head.

Offline, one-time conversion tool -- part of the TensorRT migration (see
docs/sam3/PLAN.md). Never runs at serving time. Direct transcription of
sam3_build_text_encoder_graph / sam3_build_geom_enc_graph /
sam3_build_fenc_graph / sam3_build_ddec_graph / sam3_ddec_layer_forward /
sam3_build_sine_pos_embed_4d / sam3_build_query_pos / sam3_compute_box_rpb /
sam3_dot_product_scoring / sam3_build_seg_head_graph / sam3_pixel_decoder in
sam3.cpp (all read directly, not from a summary, given how much this stage
depends on exact tensor-name-to-forward-pass-role mapping -- see the
"norm1/norm2/norm3/norm_ca_text" comment on the DETR decoder below).

Exemplar boxes are SUPPORTED via the dynamic `geom_in` input (1, n_geo, 256):
the geometry-encoder input tokens (one per exemplar box + the CLS token,
CLS last) are computed on the CPU by sam3_precompute_geom_input -- box coord
projection + ROI-align/pooling over the backbone features + positional
projection + label embedding, then final_proj + norm. ROI-Align therefore
never needs to be a graph op, and each row is also exactly the persistable
"concept embedding" unit (1KB/exemplar) for precomputed concept libraries.
With no boxes, geom_in is just the CLS row (the value the old GEOM_CLS_CONST
bake held). n_geo is dynamic via a TensorRT optimization profile (1..16).

Runtime inputs: `token_ids` (int32[32], BPE tokens + zero padding) and the
image encoder's neck_det[0]/[1]/[2] outputs (`fpn0`/`fpn1`/`img_feats`,
NHWC -- exactly what Phase 1's TensorRT image encoder already produces).
Everything else that's a pure function of fixed hyperparameters (causal
mask, image positional encoding, geometry CLS embedding, DETR reference-box
sigmoid, RPB coordinate grid, sine dim_t table) is baked as a constant.
"""
import argparse
import os

import numpy as np
from onnx import TensorProto, helper

from sam3_onnx_common import (
    Manifest, GraphBuilder, add, build_model, concat, fused_mha, gelu_erf,
    layer_norm, linear, matmul, merge_heads, mha, mul, reduce_sum, reshape,
    slice_step, split_heads, split_n, sub, transpose, unsqueeze,
)

D = 256                 # neck / prompt / decoder embed dim
H = 72                   # image feature grid (72x72 = 5184 tokens)
N_IMG = H * H
FFN = 2048

TEXT_E = 1024
TEXT_HEADS = 16
TEXT_HEAD_DIM = TEXT_E // TEXT_HEADS
TEXT_L = 32
TEXT_LAYERS = 24
TEXT_VOCAB = 49408

GEOM_LAYERS = 3
GEOM_HEADS = 8
GEOM_HEAD_DIM = D // GEOM_HEADS  # 32
# The geometry-token count is DYNAMIC: N_geo = 1 CLS + one token per exemplar
# box, fed as the `geom_in` graph input (1, n_geo, 256) built on the CPU by
# sam3_precompute_geom_input (box coords projection + ROI-pooled image
# features + positional projection + label embedding + final_proj + norm --
# genuinely per-request data, unlike the pure-weights CLS-only case the old
# baked constant covered). `None` marks the dynamic token dim for
# split_heads/merge_heads, exactly like the PVS converter.
N_GEO = None

FENC_LAYERS = 6
FENC_HEADS = 8

DDEC_LAYERS = 6
DDEC_HEADS = 8
DDEC_HEAD_DIM = D // DDEC_HEADS  # 32
NQ = 200                 # object queries
NQ1 = NQ + 1              # + presence token

T = None                 # combined prompt length = 32 + n_geo, dynamic

LN_EPS = 1e-5
GN_EPS = 1e-5
GN_GROUPS = 8

FPN_HW = {0: H * 4, 1: H * 2, 2: H}  # 288, 144, 72

# Set by main() from --fenc-chunks: number of exact query-row chunks for the
# fusion encoder's 5184-token self-attention. 0/None = unchunked (required
# for FP16/mixed builds to keep TensorRT's fused attention); 8 = the
# validated whole-FP32 configuration (ctx memory 1938MB -> 466MB).
FENC_Q_CHUNKS = None


def mlp3(g, x, mf, prefix, name, relu_last=False):
    """3-layer MLP (Linear,ReLU,Linear,ReLU,Linear[,ReLU]) -- the shared
    pattern for bbox_embed / ref_point_head(2-layer, handled separately) /
    hyper / iou_prediction_head / pred_obj_score_head / presence_token_head /
    mask_embed."""
    w0 = g.const(name + "_w0", mf.linear_w(prefix + "layers.0.weight"))
    b0 = g.const(name + "_b0", mf.bias(prefix + "layers.0.bias"))
    w1 = g.const(name + "_w1", mf.linear_w(prefix + "layers.1.weight"))
    b1 = g.const(name + "_b1", mf.bias(prefix + "layers.1.bias"))
    w2 = g.const(name + "_w2", mf.linear_w(prefix + "layers.2.weight"))
    b2 = g.const(name + "_b2", mf.bias(prefix + "layers.2.bias"))
    x = g.node("Relu", [linear(g, x, w0, b0, name=name + "_l0")], name=name + "_r0")
    x = g.node("Relu", [linear(g, x, w1, b1, name=name + "_l1")], name=name + "_r1")
    x = linear(g, x, w2, b2, name=name + "_l2")
    if relu_last:
        x = g.node("Relu", [x], name=name + "_r2")
    return x


# ── Text encoder (24 causal-masked pre-norm blocks, exact-erf GELU) ───────
def build_text_encoder(g, mf, token_ids, causal_mask_const):
    token_embed = g.const("text_token_embed", mf.raw("text.token_embed.weight").reshape(TEXT_VOCAB, TEXT_E))
    pos_embed = g.const("text_pos_embed", mf.raw("text.pos_embed").reshape(1, TEXT_L, TEXT_E))

    x = g.node("Gather", [token_embed, token_ids], name="text_embed_lookup", axis=0)  # (32,1024)
    x = reshape(g, x, [1, TEXT_L, TEXT_E], name="text_embed_r")
    x = add(g, x, pos_embed, name="text_add_pos")

    for i in range(TEXT_LAYERS):
        name = f"text_block{i}"
        p = f"text.blocks.{i}."
        ln1_w = g.const(name + "_ln1w", mf.bias(p + "ln_1.weight"))
        ln1_b = g.const(name + "_ln1b", mf.bias(p + "ln_1.bias"))
        ln2_w = g.const(name + "_ln2w", mf.bias(p + "ln_2.weight"))
        ln2_b = g.const(name + "_ln2b", mf.bias(p + "ln_2.bias"))
        in_w = g.const(name + "_inw", mf.linear_w(p + "attn.in_proj.weight"))
        in_b = g.const(name + "_inb", mf.bias(p + "attn.in_proj.bias"))
        out_w = g.const(name + "_outw", mf.linear_w(p + "attn.out_proj.weight"))
        out_b = g.const(name + "_outb", mf.bias(p + "attn.out_proj.bias"))
        fc1_w = g.const(name + "_fc1w", mf.linear_w(p + "mlp.fc1.weight"))
        fc1_b = g.const(name + "_fc1b", mf.bias(p + "mlp.fc1.bias"))
        fc2_w = g.const(name + "_fc2w", mf.linear_w(p + "mlp.fc2.weight"))
        fc2_b = g.const(name + "_fc2b", mf.bias(p + "mlp.fc2.bias"))

        shortcut = x
        xn = layer_norm(g, x, ln1_w, ln1_b, LN_EPS, name=name + "_ln1")
        qkv = linear(g, xn, in_w, in_b, name=name + "_qkv")
        q, k, v = split_n(g, qkv, [TEXT_E, TEXT_E, TEXT_E], axis=2, name=name + "_split")
        q = split_heads(g, q, 1, TEXT_L, TEXT_HEADS, TEXT_HEAD_DIM, name=name + "_qh")
        k = split_heads(g, k, 1, TEXT_L, TEXT_HEADS, TEXT_HEAD_DIM, name=name + "_kh")
        v = split_heads(g, v, 1, TEXT_L, TEXT_HEADS, TEXT_HEAD_DIM, name=name + "_vh")
        attn = mha(g, q, k, v, TEXT_HEADS, TEXT_HEAD_DIM, name=name + "_mha", mask=causal_mask_const)
        attn = merge_heads(g, attn, 1, TEXT_L, TEXT_E, name=name + "_merge")
        attn = linear(g, attn, out_w, out_b, name=name + "_outproj")
        x = add(g, shortcut, attn, name=name + "_res1")

        shortcut = x
        xn = layer_norm(g, x, ln2_w, ln2_b, LN_EPS, name=name + "_ln2")
        hdd = linear(g, xn, fc1_w, fc1_b, name=name + "_fc1")
        hdd = gelu_erf(g, hdd, name=name + "_gelu")
        hdd = linear(g, hdd, fc2_w, fc2_b, name=name + "_fc2")
        x = add(g, shortcut, hdd, name=name + "_res2")

    ln_final_w = g.const("text_lnf_w", mf.bias("text.ln_final.weight"))
    ln_final_b = g.const("text_lnf_b", mf.bias("text.ln_final.bias"))
    x = layer_norm(g, x, ln_final_w, ln_final_b, LN_EPS, name="text_ln_final")
    resizer_w = g.const("text_resizer_w", mf.linear_w("text.resizer.weight"))
    resizer_b = g.const("text_resizer_b", mf.bias("text.resizer.bias"))
    x = linear(g, x, resizer_w, resizer_b, name="text_resizer")  # (1,32,256)
    return x


# ── Geometry encoder (3 pre-norm layers over n_geo dynamic tokens) ──
def geom_layer(g, x, img_feats, img_pe, mf, i, name):
    p = f"geom.layers.{i}."
    n1w, n1b = g.const(name + "_n1w", mf.bias(p + "norm1.weight")), g.const(name + "_n1b", mf.bias(p + "norm1.bias"))
    n2w, n2b = g.const(name + "_n2w", mf.bias(p + "norm2.weight")), g.const(name + "_n2b", mf.bias(p + "norm2.bias"))
    n3w, n3b = g.const(name + "_n3w", mf.bias(p + "norm3.weight")), g.const(name + "_n3b", mf.bias(p + "norm3.bias"))
    sa_qkv = [(g.const(f"{name}_sa{j}w", w), g.const(f"{name}_sa{j}b", b))
              for j, (w, b) in enumerate(mf.fused_qkv(p + "sa.in_proj_weight", p + "sa.in_proj_bias", D))]
    sa_out_w = g.const(name + "_saoutw", mf.linear_w(p + "sa.out_proj.weight"))
    sa_out_b = g.const(name + "_saoutb", mf.bias(p + "sa.out_proj.bias"))
    ca_qkv = [(g.const(f"{name}_ca{j}w", w), g.const(f"{name}_ca{j}b", b))
              for j, (w, b) in enumerate(mf.fused_qkv(p + "ca.in_proj_weight", p + "ca.in_proj_bias", D))]
    ca_out_w = g.const(name + "_caoutw", mf.linear_w(p + "ca.out_proj.weight"))
    ca_out_b = g.const(name + "_caoutb", mf.bias(p + "ca.out_proj.bias"))
    l1_w, l1_b = g.const(name + "_l1w", mf.linear_w(p + "linear1.weight")), g.const(name + "_l1b", mf.bias(p + "linear1.bias"))
    l2_w, l2_b = g.const(name + "_l2w", mf.linear_w(p + "linear2.weight")), g.const(name + "_l2b", mf.bias(p + "linear2.bias"))

    # 1. Self-attn: Q=K=V=LN(x), no positional encoding.
    shortcut = x
    xn = layer_norm(g, x, n1w, n1b, LN_EPS, name=name + "_ln1")
    attn = fused_mha(g, xn, xn, sa_qkv, sa_out_w, sa_out_b, 1, N_GEO, N_GEO,
                     GEOM_HEADS, GEOM_HEAD_DIM, name=name + "_sa")
    x = add(g, shortcut, attn, name=name + "_res1")

    # 2. Cross-attn: Q=LN(x) (no pos), K=img_feats+img_pe, V=img_feats (no pos).
    shortcut = x
    xn = layer_norm(g, x, n2w, n2b, LN_EPS, name=name + "_ln2")
    k_in = add(g, img_feats, img_pe, name=name + "_kpe")
    attn = fused_mha(g, xn, img_feats, ca_qkv, ca_out_w, ca_out_b, 1, N_GEO, N_IMG,
                     GEOM_HEADS, GEOM_HEAD_DIM, name=name + "_ca", k_src=k_in, v_src=img_feats)
    x = add(g, shortcut, attn, name=name + "_res2")

    # 3. FFN (ReLU).
    shortcut = x
    xn = layer_norm(g, x, n3w, n3b, LN_EPS, name=name + "_ln3")
    hdd = linear(g, xn, l1_w, l1_b, name=name + "_ffn1")
    hdd = g.node("Relu", [hdd], name=name + "_relu")
    hdd = linear(g, hdd, l2_w, l2_b, name=name + "_ffn2")
    x = add(g, shortcut, hdd, name=name + "_res3")
    return x


def build_geom_encoder(g, mf, geom_in, img_feats, img_pe):
    x = geom_in  # (1,n_geo,256) graph input -- see the N_GEO comment up top
    for i in range(GEOM_LAYERS):
        x = geom_layer(g, x, img_feats, img_pe, mf, i, name=f"geom_layer{i}")
    enw = g.const("geom_encnorm_w", mf.bias("geom.encode_norm.weight"))
    enb = g.const("geom_encnorm_b", mf.bias("geom.encode_norm.bias"))
    x = layer_norm(g, x, enw, enb, LN_EPS, name="geom_encode_norm")
    return x  # (1,n_geo,256)


# ── Fusion encoder (6 pre-norm layers) ────────────────────────────────────
def fenc_layer(g, x, img_pe, prompt, prompt_mask, mf, i, name):
    p = f"fenc.layers.{i}."
    n1w, n1b = g.const(name + "_n1w", mf.bias(p + "norm1.weight")), g.const(name + "_n1b", mf.bias(p + "norm1.bias"))
    n2w, n2b = g.const(name + "_n2w", mf.bias(p + "norm2.weight")), g.const(name + "_n2b", mf.bias(p + "norm2.bias"))
    n3w, n3b = g.const(name + "_n3w", mf.bias(p + "norm3.weight")), g.const(name + "_n3b", mf.bias(p + "norm3.bias"))
    sa_qkv = [(g.const(f"{name}_sa{j}w", w), g.const(f"{name}_sa{j}b", b))
              for j, (w, b) in enumerate(mf.fused_qkv(p + "sa.in_proj_weight", p + "sa.in_proj_bias", D))]
    sa_out_w = g.const(name + "_saoutw", mf.linear_w(p + "sa.out_proj.weight"))
    sa_out_b = g.const(name + "_saoutb", mf.bias(p + "sa.out_proj.bias"))
    ca_qkv = [(g.const(f"{name}_ca{j}w", w), g.const(f"{name}_ca{j}b", b))
              for j, (w, b) in enumerate(mf.fused_qkv(p + "ca.in_proj_weight", p + "ca.in_proj_bias", D))]
    ca_out_w = g.const(name + "_caoutw", mf.linear_w(p + "ca.out_proj.weight"))
    ca_out_b = g.const(name + "_caoutb", mf.bias(p + "ca.out_proj.bias"))
    l1_w, l1_b = g.const(name + "_l1w", mf.linear_w(p + "linear1.weight")), g.const(name + "_l1b", mf.bias(p + "linear1.bias"))
    l2_w, l2_b = g.const(name + "_l2w", mf.linear_w(p + "linear2.weight")), g.const(name + "_l2b", mf.bias(p + "linear2.bias"))

    # 1. Self-attn on image tokens: Q=K=x+pos, V=x (no pos). This is the ONE
    # attention in PCS whose scores tensor is huge -- (8,5184,5184) f32 =
    # 860MB, measured to dominate the FP32 engine's ~1.9GB execution-context
    # memory since TensorRT has no FP32 fused attention. --fenc-chunks splits
    # it into exact query-row chunks (see mha's doc) for whole-FP32 builds;
    # leave it OFF (the default) for FP16/mixed builds, where chunking would
    # defeat TensorRT's fused-MHA kernels (which already avoid materializing
    # the scores tensor). Every other attention here is small (<=33MB scores).
    shortcut = x
    xn = layer_norm(g, x, n1w, n1b, LN_EPS, name=name + "_ln1")
    qk_in = add(g, xn, img_pe, name=name + "_qkpe")
    attn = fused_mha(g, qk_in, xn, sa_qkv, sa_out_w, sa_out_b, 1, N_IMG, N_IMG,
                     FENC_HEADS, D // FENC_HEADS, name=name + "_sa", k_src=qk_in, v_src=xn,
                     q_chunk=(N_IMG // FENC_Q_CHUNKS) if FENC_Q_CHUNKS else None)
    x = add(g, shortcut, attn, name=name + "_res1")

    # 2. Cross-attn: image tokens attend to prompt (no pos anywhere here).
    shortcut = x
    xn = layer_norm(g, x, n2w, n2b, LN_EPS, name=name + "_ln2")
    attn = fused_mha(g, xn, prompt, ca_qkv, ca_out_w, ca_out_b, 1, N_IMG, T,
                     FENC_HEADS, D // FENC_HEADS, name=name + "_ca", mask=prompt_mask)
    x = add(g, shortcut, attn, name=name + "_res2")

    # 3. FFN (ReLU).
    shortcut = x
    xn = layer_norm(g, x, n3w, n3b, LN_EPS, name=name + "_ln3")
    hdd = linear(g, xn, l1_w, l1_b, name=name + "_ffn1")
    hdd = g.node("Relu", [hdd], name=name + "_relu")
    hdd = linear(g, hdd, l2_w, l2_b, name=name + "_ffn2")
    x = add(g, shortcut, hdd, name=name + "_res3")
    return x


def build_fenc(g, mf, img_feats, img_pe, prompt, prompt_mask):
    x = img_feats
    for i in range(FENC_LAYERS):
        x = fenc_layer(g, x, img_pe, prompt, prompt_mask, mf, i, name=f"fenc_layer{i}")
    return x  # (1,5184,256), no final norm


# ── DETR decoder + scoring ─────────────────────────────────────────────────
def sine_pos_embed_4d(g, ref_boxes, dim_t_const, name):
    """ref_boxes: (1,200,4) cxcywh. Returns (1,200,512), coord order [cy,cx,w,h]."""
    coord_order = [1, 0, 2, 3]
    parts = []
    for c in coord_order:
        coord = slice_step(g, ref_boxes, axis=2, start=c, step=1, end=c + 1, name=f"{name}_c{c}")  # (1,200,1)
        angles = mul(g, coord, dim_t_const, name=f"{name}_ang{c}")  # (1,200,1)*(1,1,64) bcast -> (1,200,64)
        sin_v = g.node("Sin", [angles], name=f"{name}_sin{c}")
        cos_v = g.node("Cos", [angles], name=f"{name}_cos{c}")
        sin_u = unsqueeze(g, sin_v, [3], name=f"{name}_sinu{c}")  # (1,200,64,1)
        cos_u = unsqueeze(g, cos_v, [3], name=f"{name}_cosu{c}")
        inter = concat(g, [sin_u, cos_u], axis=3, name=f"{name}_inter{c}")  # (1,200,64,2)
        inter = reshape(g, inter, [1, NQ, 128], name=f"{name}_flat{c}")     # interleaved sin/cos
        parts.append(inter)
    return concat(g, parts, axis=2, name=name + "_concat")  # (1,200,512)


def build_query_pos(g, ref_boxes, dim_t_const, mf, name):
    sine = sine_pos_embed_4d(g, ref_boxes, dim_t_const, name=name + "_sine")  # (1,200,512)
    w0 = g.const(name + "_w0", mf.linear_w("ddec.ref_point_head.layers.0.weight"))
    b0 = g.const(name + "_b0", mf.bias("ddec.ref_point_head.layers.0.bias"))
    w1 = g.const(name + "_w1", mf.linear_w("ddec.ref_point_head.layers.1.weight"))
    b1 = g.const(name + "_b1", mf.bias("ddec.ref_point_head.layers.1.bias"))
    h = g.node("Relu", [linear(g, sine, w0, b0, name=name + "_l0")], name=name + "_r0")
    qpos_obj = linear(g, h, w1, b1, name=name + "_l1")  # (1,200,256)
    qpos_pres = g.const(name + "_pres_zero", np.zeros((1, 1, D), dtype=np.float32))
    return concat(g, [qpos_pres, qpos_obj], axis=1, name=name + "_full")  # (1,201,256)


def box_rpb(g, ref_boxes, rpb_coords_const, mf, i, name):
    """Returns (1,8,201,5184) additive mask for the image cross-attention."""
    cx = slice_step(g, ref_boxes, axis=2, start=0, step=1, end=1, name=f"{name}_cx")  # (1,200,1)
    cy = slice_step(g, ref_boxes, axis=2, start=1, step=1, end=2, name=f"{name}_cy")
    bw = slice_step(g, ref_boxes, axis=2, start=2, step=1, end=3, name=f"{name}_bw")
    bh = slice_step(g, ref_boxes, axis=2, start=3, step=1, end=4, name=f"{name}_bh")
    half_w = g.const(f"{name}_half", np.float32(0.5))
    hw2 = mul(g, bw, half_w, name=f"{name}_hw2")
    hh2 = mul(g, bh, half_w, name=f"{name}_hh2")
    x0 = sub(g, cx, hw2, name=f"{name}_x0")  # (1,200,1)
    x1 = add(g, cx, hw2, name=f"{name}_x1")
    y0 = sub(g, cy, hh2, name=f"{name}_y0")
    y1 = add(g, cy, hh2, name=f"{name}_y1")

    # rpb_coords_const: (1,72,1,1) broadcastable against edges reshaped (1,1,200,1)
    def outer_sub(edge, tag):
        e = reshape(g, edge, [1, 1, NQ, 1], name=f"{name}_{tag}_r")
        return sub(g, rpb_coords_const, e, name=f"{name}_{tag}_delta")  # (1,72,200,1)

    dx0, dx1 = outer_sub(x0, "dx0"), outer_sub(x1, "dx1")
    dy0, dy1 = outer_sub(y0, "dy0"), outer_sub(y1, "dy1")

    def log_transform(d, tag):
        d8 = mul(g, d, g.const(f"{name}_{tag}_8", np.float32(8.0)), name=f"{name}_{tag}_d8")
        sign_d = g.node("Sign", [d8], name=f"{name}_{tag}_sign")
        abs_d = g.node("Abs", [d8], name=f"{name}_{tag}_abs")
        abs_d1 = add(g, abs_d, g.const(f"{name}_{tag}_one", np.float32(1.0)), name=f"{name}_{tag}_abs1")
        lg = g.node("Log", [abs_d1], name=f"{name}_{tag}_log")
        lg = mul(g, lg, g.const(f"{name}_{tag}_invln2", np.float32(1.0 / np.log(2.0))), name=f"{name}_{tag}_log2")
        lg = mul(g, lg, g.const(f"{name}_{tag}_invlog8", np.float32(1.0 / np.log2(8.0))), name=f"{name}_{tag}_scaled")
        return mul(g, sign_d, lg, name=f"{name}_{tag}_signed")

    dx0, dx1 = log_transform(dx0, "lx0"), log_transform(dx1, "lx1")
    dy0, dy1 = log_transform(dy0, "ly0"), log_transform(dy1, "ly1")

    deltas_x = concat(g, [dx0, dx1], axis=3, name=f"{name}_deltasx")  # (1,72,200,2)
    deltas_y = concat(g, [dy0, dy1], axis=3, name=f"{name}_deltasy")

    def rpb_mlp(deltas, axis, tag):
        w0 = g.const(f"{name}_{tag}_w0", mf.linear_w(f"ddec.boxRPB_embed_{axis}.layers.0.weight"))
        b0 = g.const(f"{name}_{tag}_b0", mf.bias(f"ddec.boxRPB_embed_{axis}.layers.0.bias"))
        w1 = g.const(f"{name}_{tag}_w1", mf.linear_w(f"ddec.boxRPB_embed_{axis}.layers.1.weight"))
        b1 = g.const(f"{name}_{tag}_b1", mf.bias(f"ddec.boxRPB_embed_{axis}.layers.1.bias"))
        h = g.node("Relu", [linear(g, deltas, w0, b0, name=f"{name}_{tag}_l0")], name=f"{name}_{tag}_r0")
        return linear(g, h, w1, b1, name=f"{name}_{tag}_l1")  # (1,72,200,8)

    rpb_x = rpb_mlp(deltas_x, "x", "rx")  # (1,72,200,8) axis1=W
    rpb_y = rpb_mlp(deltas_y, "y", "ry")  # (1,72,200,8) axis1=H

    rpb_y_5d = reshape(g, rpb_y, [1, H, 1, NQ, DDEC_HEADS], name=f"{name}_y5d")
    rpb_x_5d = reshape(g, rpb_x, [1, 1, H, NQ, DDEC_HEADS], name=f"{name}_x5d")
    rpb_hw = add(g, rpb_y_5d, rpb_x_5d, name=f"{name}_outer")  # (1,H,W,NQ,heads)
    rpb_flat = reshape(g, rpb_hw, [1, N_IMG, NQ, DDEC_HEADS], name=f"{name}_flat")  # s=h*72+w
    rpb_t = transpose(g, rpb_flat, [0, 3, 2, 1], name=f"{name}_t")  # (1,heads,NQ,spatial)

    zeros_pres = g.const(f"{name}_zeros_pres", np.zeros((1, DDEC_HEADS, 1, N_IMG), dtype=np.float32))
    return concat(g, [zeros_pres, rpb_t], axis=2, name=f"{name}_full")  # (1,heads,NQ+1,spatial)


def ddec_layer(g, queries, ref_boxes, fenc_out, img_pe, prompt, prompt_mask,
              dim_t_const, rpb_coords_const, mf, i, name):
    p = f"ddec.layers.{i}."
    query_pos = build_query_pos(g, ref_boxes, dim_t_const, mf, name=name + "_qpos")
    rpb_mask = box_rpb(g, ref_boxes, rpb_coords_const, mf, i, name=name + "_rpb")

    sa_qkv = [(g.const(f"{name}_sa{j}w", w), g.const(f"{name}_sa{j}b", b))
              for j, (w, b) in enumerate(mf.fused_qkv(p + "sa.in_proj_weight", p + "sa.in_proj_bias", D))]
    sa_out_w = g.const(name + "_saoutw", mf.linear_w(p + "sa.out_proj.weight"))
    sa_out_b = g.const(name + "_saoutb", mf.bias(p + "sa.out_proj.bias"))
    n2w, n2b = g.const(name + "_n2w", mf.bias(p + "norm2.weight")), g.const(name + "_n2b", mf.bias(p + "norm2.bias"))

    ct_qkv = [(g.const(f"{name}_ct{j}w", w), g.const(f"{name}_ct{j}b", b))
              for j, (w, b) in enumerate(mf.fused_qkv(p + "ca_text.in_proj_weight", p + "ca_text.in_proj_bias", D))]
    ct_out_w = g.const(name + "_ctoutw", mf.linear_w(p + "ca_text.out_proj.weight"))
    ct_out_b = g.const(name + "_ctoutb", mf.bias(p + "ca_text.out_proj.bias"))
    n3w, n3b = g.const(name + "_n3w", mf.bias(p + "norm_ca_text.weight")), g.const(name + "_n3b", mf.bias(p + "norm_ca_text.bias"))

    ci_qkv = [(g.const(f"{name}_ci{j}w", w), g.const(f"{name}_ci{j}b", b))
              for j, (w, b) in enumerate(mf.fused_qkv(p + "ca.in_proj_weight", p + "ca.in_proj_bias", D))]
    ci_out_w = g.const(name + "_cioutw", mf.linear_w(p + "ca.out_proj.weight"))
    ci_out_b = g.const(name + "_cioutb", mf.bias(p + "ca.out_proj.bias"))
    n1w, n1b = g.const(name + "_n1w", mf.bias(p + "norm1.weight")), g.const(name + "_n1b", mf.bias(p + "norm1.bias"))

    l1w, l1b = g.const(name + "_l1w", mf.linear_w(p + "linear1.weight")), g.const(name + "_l1b", mf.bias(p + "linear1.bias"))
    l2w, l2b = g.const(name + "_l2w", mf.linear_w(p + "linear2.weight")), g.const(name + "_l2b", mf.bias(p + "linear2.bias"))
    n4w, n4b = g.const(name + "_n4w", mf.bias(p + "norm3.weight")), g.const(name + "_n4b", mf.bias(p + "norm3.bias"))

    # 1. Self-attn among all 201 queries: Q=K=queries+pos, V=queries (no pos). Post-norm2.
    qpos = add(g, queries, query_pos, name=name + "_qpos1")
    sa = fused_mha(g, qpos, qpos, sa_qkv, sa_out_w, sa_out_b, 1, NQ1, NQ1, DDEC_HEADS, DDEC_HEAD_DIM,
                  name=name + "_sa", k_src=qpos, v_src=queries)
    queries = add(g, queries, sa, name=name + "_res1")
    queries = layer_norm(g, queries, n2w, n2b, LN_EPS, name=name + "_ln_postsa")

    # 2. Cross-attn to combined prompt: Q=queries+pos, K=V=prompt. Mask=padding. Post-norm_ca_text.
    qpos2 = add(g, queries, query_pos, name=name + "_qpos2")
    ct = fused_mha(g, qpos2, prompt, ct_qkv, ct_out_w, ct_out_b, 1, NQ1, T, DDEC_HEADS, DDEC_HEAD_DIM,
                  name=name + "_ct", mask=prompt_mask)
    queries = add(g, queries, ct, name=name + "_res2")
    queries = layer_norm(g, queries, n3w, n3b, LN_EPS, name=name + "_ln_postct")

    # 3. Cross-attn to conditioned image features: Q=queries+pos, K=fenc+pe, V=fenc. Mask=RPB. Post-norm1.
    qpos3 = add(g, queries, query_pos, name=name + "_qpos3")
    k_in = add(g, fenc_out, img_pe, name=name + "_kpe")
    ci = fused_mha(g, qpos3, fenc_out, ci_qkv, ci_out_w, ci_out_b, 1, NQ1, N_IMG, DDEC_HEADS, DDEC_HEAD_DIM,
                  name=name + "_ci", mask=rpb_mask, k_src=k_in, v_src=fenc_out)
    queries = add(g, queries, ci, name=name + "_res3")
    queries = layer_norm(g, queries, n1w, n1b, LN_EPS, name=name + "_ln_postci")

    # 4. FFN (ReLU). Post-norm3(loaded as struct field norm4, tensor name is "norm3").
    ffn = linear(g, queries, l1w, l1b, name=name + "_ffn1")
    ffn = g.node("Relu", [ffn], name=name + "_relu")
    ffn = linear(g, ffn, l2w, l2b, name=name + "_ffn2")
    queries = add(g, queries, ffn, name=name + "_res4")
    queries = layer_norm(g, queries, n4w, n4b, LN_EPS, name=name + "_ln_postffn")

    # Box refinement (object queries only, index 1..200).
    obj_q = slice_step(g, queries, axis=1, start=1, step=1, end=NQ1, name=name + "_objq")
    ddec_norm_w = g.const(name + "_ddecnormw", mf.bias("ddec.norm.weight"))
    ddec_norm_b = g.const(name + "_ddecnormb", mf.bias("ddec.norm.bias"))
    obj_q_normed = layer_norm(g, obj_q, ddec_norm_w, ddec_norm_b, LN_EPS, name=name + "_objn")
    delta = mlp3(g, obj_q_normed, mf, "ddec.bbox_embed.", name=name + "_bbox")  # (1,200,4)

    ref_clamped = g.node("Clip", [ref_boxes,
                                  g.const(name + "_clip_lo", np.float32(1e-3)),
                                  g.const(name + "_clip_hi", np.float32(1.0 - 1e-3))], name=name + "_refclip")
    log_x = g.node("Log", [ref_clamped], name=name + "_logx")
    one_minus = sub(g, g.const(name + "_one", np.float32(1.0)), ref_clamped, name=name + "_1mx")
    log_1mx = g.node("Log", [one_minus], name=name + "_log1mx")
    inv_sig = sub(g, log_x, log_1mx, name=name + "_invsig")
    ref_boxes = g.node("Sigmoid", [add(g, inv_sig, delta, name=name + "_presig")], name=name + "_newref")

    return queries, ref_boxes, obj_q_normed


def dot_product_scoring(g, prompt, valid_indicator, n_valid, obj_queries, mf, name):
    l0w = g.const(name + "_l0w", mf.linear_w("scoring.prompt_mlp.layers.0.weight"))
    l0b = g.const(name + "_l0b", mf.bias("scoring.prompt_mlp.layers.0.bias"))
    l1w = g.const(name + "_l1w", mf.linear_w("scoring.prompt_mlp.layers.1.weight"))
    l1b = g.const(name + "_l1b", mf.bias("scoring.prompt_mlp.layers.1.bias"))
    onw = g.const(name + "_onw", mf.bias("scoring.prompt_mlp.out_norm.weight"))
    onb = g.const(name + "_onb", mf.bias("scoring.prompt_mlp.out_norm.bias"))

    h = linear(g, prompt, l0w, l0b, name=name + "_l0")
    h = g.node("Relu", [h], name=name + "_relu")
    h = linear(g, h, l1w, l1b, name=name + "_l1")
    text_mlp = add(g, h, prompt, name=name + "_res")
    text_mlp = layer_norm(g, text_mlp, onw, onb, LN_EPS, name=name + "_ln")  # (1,33,256)

    weighted = mul(g, text_mlp, valid_indicator, name=name + "_weighted")  # valid_indicator (1,33,1)
    summed = reduce_sum(g, weighted, [1], 0, name=name + "_sum")  # (1,256)
    pooled = g.node("Div", [summed, n_valid], name=name + "_pooled")  # (1,256)

    ppw = g.const(name + "_ppw", mf.linear_w("scoring.prompt_proj.weight"))
    ppb = g.const(name + "_ppb", mf.bias("scoring.prompt_proj.bias"))
    proj_pooled = linear(g, pooled, ppw, ppb, name=name + "_projpooled")  # (1,256)

    hsw = g.const(name + "_hsw", mf.linear_w("scoring.hs_proj.weight"))
    hsb = g.const(name + "_hsb", mf.bias("scoring.hs_proj.bias"))
    proj_hs = linear(g, obj_queries, hsw, hsb, name=name + "_projhs")  # (1,200,256)

    proj_pooled_u = unsqueeze(g, proj_pooled, [1], name=name + "_ppu")  # (1,1,256)
    prod = mul(g, proj_hs, proj_pooled_u, name=name + "_prod")  # (1,200,256)
    scores = reduce_sum(g, prod, [2], 0, name=name + "_scoresum")  # (1,200)
    scale_c = g.const(name + "_scale", np.float32(1.0 / np.sqrt(D)))
    scores = mul(g, scores, scale_c, name=name + "_scaled")
    scores = g.node("Clip", [scores, g.const(name + "_lo", np.float32(-12.0)),
                            g.const(name + "_hi", np.float32(12.0))], name=name + "_clip")
    return scores  # (1,200)


def build_ddec(g, mf, fenc_out, img_pe, prompt, prompt_mask, valid_indicator, n_valid, dim_t_const, rpb_coords_const):
    presence_tok = g.const("ddec_pres_tok", mf.raw("ddec.presence_token.weight").reshape(1, 1, D))
    query_embed = g.const("ddec_query_embed", mf.raw("ddec.query_embed.weight").reshape(1, NQ, D))
    queries = concat(g, [presence_tok, query_embed], axis=1, name="ddec_init_queries")  # (1,201,256)

    ref_pts = mf.raw("ddec.reference_points.weight").reshape(NQ, 4)
    ref_boxes_np = (1.0 / (1.0 + np.exp(-ref_pts))).astype(np.float32).reshape(1, NQ, 4)
    ref_boxes = g.const("ddec_ref_boxes_init", ref_boxes_np)  # baked: reference_points is a fixed weight

    obj_q_normed = None
    for i in range(DDEC_LAYERS):
        queries, ref_boxes, obj_q_normed = ddec_layer(
            g, queries, ref_boxes, fenc_out, img_pe, prompt, prompt_mask,
            dim_t_const, rpb_coords_const, mf, i, name=f"ddec_layer{i}")

    last_presence = slice_step(g, queries, axis=1, start=0, step=1, end=1, name="ddec_last_presence")  # (1,1,256)
    obj_queries = obj_q_normed  # already LN(ddec.norm)'d during the last layer's box refinement

    class_scores = dot_product_scoring(g, prompt, valid_indicator, n_valid, obj_queries, mf, name="scoring")

    pon_w = g.const("ddec_presnormw", mf.bias("ddec.presence_token_out_norm.weight"))
    pon_b = g.const("ddec_presnormb", mf.bias("ddec.presence_token_out_norm.bias"))
    pres = layer_norm(g, last_presence, pon_w, pon_b, LN_EPS, name="ddec_presnorm")
    presence_score = mlp3(g, pres, mf, "ddec.presence_token_head.", name="ddec_presmlp")  # (1,1,1)
    presence_score = reshape(g, presence_score, [1, 1], name="ddec_presence_final")

    return obj_queries, class_scores, ref_boxes, presence_score


# ── MaskFormer segmentation head ──────────────────────────────────────────
def group_norm(g, x_nchw, weight, bias, num_groups, num_channels, h, w, eps, name):
    cpg = num_channels // num_groups
    xr = reshape(g, x_nchw, [1, num_groups, cpg, h, w], name=name + "_r")
    mean = g.node("ReduceMean", [xr], name=name + "_mean", axes=[2, 3, 4], keepdims=1)
    centered = sub(g, xr, mean, name=name + "_centered")
    sq = mul(g, centered, centered, name=name + "_sq")
    var = g.node("ReduceMean", [sq], name=name + "_var", axes=[2, 3, 4], keepdims=1)
    eps_c = g.const(name + "_eps", np.float32(eps))
    std = g.node("Sqrt", [add(g, var, eps_c, name=name + "_vareps")], name=name + "_std")
    normed = g.node("Div", [centered, std], name=name + "_normed")
    xr2 = reshape(g, normed, [1, num_channels, h, w], name=name + "_r2")
    w_r = reshape(g, weight, [1, num_channels, 1, 1], name=name + "_wr")
    b_r = reshape(g, bias, [1, num_channels, 1, 1], name=name + "_br")
    return add(g, mul(g, xr2, w_r, name=name + "_scaled"), b_r, name=name + "_biased")


def nearest_upsample2x(g, x_nchw, name):
    scales_f = g.const(name + "_scales", np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32))
    roi = g.const(name + "_roi", np.array([], dtype=np.float32))
    return g.node("Resize", [x_nchw, roi, scales_f], name=name, mode="nearest",
                 coordinate_transformation_mode="asymmetric", nearest_mode="floor")


def pixel_decoder(g, fpn2_nhwc, fpn0_nhwc, fpn1_nhwc, mf):
    x = transpose(g, fpn2_nhwc, [0, 3, 1, 2], name="pxd_fpn2_nchw")       # (1,256,72,72)
    up = nearest_upsample2x(g, x, name="pxd_up0")                         # (1,256,144,144)
    fpn1 = transpose(g, fpn1_nhwc, [0, 3, 1, 2], name="pxd_fpn1_nchw")
    merged = add(g, up, fpn1, name="pxd_merge0")
    w0 = g.const("pxd_conv0_w", mf.conv_w("seg.pixel_decoder.conv_layers.0.weight"))
    b0 = g.const("pxd_conv0_b", mf.bias("seg.pixel_decoder.conv_layers.0.bias"))
    conv0 = g.node("Conv", [merged, w0, b0], name="pxd_conv0", kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
    gnw0 = g.const("pxd_gn0_w", mf.bias("seg.pixel_decoder.norms.0.weight"))
    gnb0 = g.const("pxd_gn0_b", mf.bias("seg.pixel_decoder.norms.0.bias"))
    gn0 = group_norm(g, conv0, gnw0, gnb0, GN_GROUPS, D, FPN_HW[1], FPN_HW[1], GN_EPS, name="pxd_gn0")
    relu0 = g.node("Relu", [gn0], name="pxd_relu0")

    up2 = nearest_upsample2x(g, relu0, name="pxd_up1")                    # (1,256,288,288)
    fpn0 = transpose(g, fpn0_nhwc, [0, 3, 1, 2], name="pxd_fpn0_nchw")
    merged2 = add(g, up2, fpn0, name="pxd_merge1")
    w1 = g.const("pxd_conv1_w", mf.conv_w("seg.pixel_decoder.conv_layers.1.weight"))
    b1 = g.const("pxd_conv1_b", mf.bias("seg.pixel_decoder.conv_layers.1.bias"))
    conv1 = g.node("Conv", [merged2, w1, b1], name="pxd_conv1", kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
    gnw1 = g.const("pxd_gn1_w", mf.bias("seg.pixel_decoder.norms.1.weight"))
    gnb1 = g.const("pxd_gn1_b", mf.bias("seg.pixel_decoder.norms.1.bias"))
    gn1 = group_norm(g, conv1, gnw1, gnb1, GN_GROUPS, D, FPN_HW[0], FPN_HW[0], GN_EPS, name="pxd_gn1")
    relu1 = g.node("Relu", [gn1], name="pxd_relu1")
    return relu1  # (1,256,288,288) NCHW


def build_seg_head(g, mf, fenc_out, fpn0, fpn1, prompt, prompt_mask, obj_queries):
    canw = g.const("seg_canormw", mf.bias("seg.cross_attn_norm.weight"))
    canb = g.const("seg_canormb", mf.bias("seg.cross_attn_norm.bias"))
    ca_norm = layer_norm(g, fenc_out, canw, canb, LN_EPS, name="seg_ca_norm")

    ca_qkv = [(g.const(f"seg_ca{j}w", w), g.const(f"seg_ca{j}b", b))
              for j, (w, b) in enumerate(mf.fused_qkv("seg.cross_attend_prompt.in_proj_weight",
                                                       "seg.cross_attend_prompt.in_proj_bias", D))]
    ca_out_w = g.const("seg_caoutw", mf.linear_w("seg.cross_attend_prompt.out_proj.weight"))
    ca_out_b = g.const("seg_caoutb", mf.bias("seg.cross_attend_prompt.out_proj.bias"))
    ca_out = fused_mha(g, ca_norm, prompt, ca_qkv, ca_out_w, ca_out_b, 1, N_IMG, T, 8, D // 8,
                       name="seg_ca", mask=prompt_mask)
    enc = add(g, fenc_out, ca_out, name="seg_enc")  # residual, no post-norm

    fpn2_mod = reshape(g, enc, [1, H, H, D], name="seg_fpn2_mod")  # NHWC
    pixel_feats = pixel_decoder(g, fpn2_mod, fpn0, fpn1, mf)       # NCHW (1,256,288,288)

    isw = g.const("seg_instw", mf.conv_w("seg.instance_seg_head.weight"))
    isb = g.const("seg_instb", mf.bias("seg.instance_seg_head.bias"))
    pixel_embed = g.node("Conv", [pixel_feats, isw, isb], name="seg_inst",
                         kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0])  # (1,256,288,288)
    pixel_embed = transpose(g, pixel_embed, [0, 2, 3, 1], name="seg_pe_nhwc")     # (1,288,288,256)
    pixel_embed = reshape(g, pixel_embed, [1, FPN_HW[0] * FPN_HW[0], D], name="seg_pe_flat")

    mask_embed = mlp3(g, obj_queries, mf, "seg.mask_predictor.mask_embed.", name="seg_maskembed")  # (1,200,256)
    mask_embed_t = transpose(g, mask_embed, [0, 2, 1], name="seg_me_t")  # (1,256,200)
    mask_logits = matmul(g, pixel_embed, mask_embed_t, name="seg_masklogits")  # (1,82944,200)
    mask_logits = transpose(g, mask_logits, [0, 2, 1], name="seg_ml_t")       # (1,200,82944)
    mask_logits = reshape(g, mask_logits, [1, NQ, FPN_HW[0], FPN_HW[0]], name="seg_ml_final")
    return mask_logits


# ── Full graph assembly ────────────────────────────────────────────────────
def build_graph(mf):
    g = GraphBuilder()

    token_ids = "token_ids"
    img_feats_nhwc = "img_feats"
    fpn0_nhwc = "fpn0"
    fpn1_nhwc = "fpn1"

    # img_feats: NHWC (1,72,72,256) -> sequence (1,5184,256), matches ggml's
    # [D,W,H,1] state.neck_det[2] reversed layout (see Phase 1 neck_scale()).
    img_feats = reshape(g, img_feats_nhwc, [1, N_IMG, D], name="img_feats_seq")

    # img_pe: sinusoidal image PE, a pure function of fixed H=W=72,D=256 --
    # bake as a constant (same formula as sam3_sinusoidal_pe_2d, and the same
    # "pure function of fixed hyperparameters" fact this project's own
    # neck-PE-cache fix already exploited).
    half = D // 2
    dim_t = 10000.0 ** (2 * (np.arange(half) // 2) / half)
    idx = (np.arange(H) + 1) / H * 2 * np.pi  # sam3_sinusoidal_pe_2d: pos=((idx+1)/size)*2pi
    ang = idx[:, None] / dim_t[None, :]  # (72,128)
    pe_1d = np.empty((H, half), dtype=np.float32)
    pe_1d[:, 0::2] = np.sin(ang[:, 0::2])
    pe_1d[:, 1::2] = np.cos(ang[:, 1::2])
    img_pe_np = np.zeros((H, H, D), dtype=np.float32)
    img_pe_np[:, :, :half] = pe_1d[:, None, :]          # y-component, channels [0:128]
    img_pe_np[:, :, half:] = pe_1d[None, :, :]          # x-component, channels [128:256]
    img_pe = g.const("img_pe_const", img_pe_np.reshape(1, N_IMG, D))

    # Geometry-encoder input: computed per-request on the CPU by
    # sam3_precompute_geom_input (exemplar boxes' coord-projection +
    # ROI-pooled image features + positional projection + label embedding,
    # then final_proj + norm; plus the CLS token) and fed as a dynamic graph
    # input -- one row per exemplar box, CLS last, n_geo = n_boxes + 1.
    geom_in = "geom_in"

    # Causal mask for the text encoder (fixed L=32).
    causal = np.triu(np.full((TEXT_L, TEXT_L), -1e9, dtype=np.float32), k=1)  # [q,kv]; kv<=q valid
    causal_mask_const = g.const("causal_mask_const", causal.reshape(1, 1, TEXT_L, TEXT_L))

    # RPB coordinate grid (72,) = i/72, baked; broadcast shape (1,72,1,1).
    rpb_coords_np = (np.arange(H, dtype=np.float32) / H).reshape(1, H, 1, 1)
    rpb_coords_const = g.const("rpb_coords_const", rpb_coords_np)

    # Sine dim_t table for DETR query-pos (64,) = 2pi/10000^(2i/128), reshape (1,1,64).
    dim_t_np = (2 * np.pi / (10000.0 ** (2 * np.arange(64) / 128))).astype(np.float32)
    dim_t_const = g.const("ddec_dim_t_const", dim_t_np.reshape(1, 1, 64))

    text_features = build_text_encoder(g, mf, token_ids, causal_mask_const)  # (1,32,256)
    geom_output = build_geom_encoder(g, mf, geom_in, img_feats, img_pe)  # (1,n_geo,256)
    prompt = concat(g, [text_features, geom_output], axis=1, name="combined_prompt")  # (1,32+n_geo,256)

    # Padding-derived mask + valid-token count (only the text part can pad;
    # geometry tokens are always valid). The geometry-side ones/zeros vectors
    # must have the DYNAMIC length n_geo -- derive them from geom_in itself:
    # zero out its values and reduce away batch+channel dims.
    zero_i32 = g.const_i32("token_zero", [0])
    is_pad = g.node("Equal", [token_ids, zero_i32], name="is_pad")  # (32,) bool
    is_pad_f = g.node("Cast", [is_pad], name="is_pad_f", to=TensorProto.FLOAT)
    ones32 = g.const("ones32", np.ones((TEXT_L,), dtype=np.float32))
    text_valid = sub(g, ones32, is_pad_f, name="text_valid")  # (32,)
    zero_f = g.const("geom_zero_f", np.float32(0.0))
    one_f = g.const("geom_one_f", np.float32(1.0))
    geom_zeroed = mul(g, geom_in, zero_f, name="geom_zeroed")            # (1,n_geo,256) of 0
    geom_bias = reduce_sum(g, geom_zeroed, [0, 2], 0, name="geom_bias")  # (n_geo,) zeros
    geom_valid = add(g, geom_bias, one_f, name="geom_valid")             # (n_geo,) ones
    valid_indicator = concat(g, [text_valid, geom_valid], axis=0, name="valid_indicator")  # (32+n_geo,)
    valid_indicator_r = reshape(g, valid_indicator, [1, -1, 1], name="valid_indicator_r")
    n_valid = reduce_sum(g, valid_indicator, [0], 0, name="n_valid")  # scalar

    neg_bias = g.const("neg_bias", np.float32(-1e9))
    text_bias = mul(g, is_pad_f, neg_bias, name="text_bias")  # (32,)
    prompt_bias = concat(g, [text_bias, geom_bias], axis=0, name="prompt_bias")  # (32+n_geo,)
    prompt_mask = reshape(g, prompt_bias, [1, 1, 1, -1], name="prompt_mask")  # bcast (1,heads,Nq,T)

    fenc_out = build_fenc(g, mf, img_feats, img_pe, prompt, prompt_mask)  # (1,5184,256)

    obj_queries, class_scores, pred_boxes, presence_score = build_ddec(
        g, mf, fenc_out, img_pe, prompt, prompt_mask, valid_indicator_r, n_valid, dim_t_const, rpb_coords_const)

    mask_logits = build_seg_head(g, mf, fenc_out, fpn0_nhwc, fpn1_nhwc, prompt, prompt_mask, obj_queries)

    g.alias(class_scores, "class_scores")
    g.alias(pred_boxes, "pred_boxes")
    g.alias(presence_score, "presence_score")
    g.alias(mask_logits, "mask_logits")

    graph_inputs = [
        helper.make_tensor_value_info("token_ids", TensorProto.INT32, [TEXT_L]),
        # Geometry tokens (exemplar boxes + CLS, CLS last), built on the CPU
        # by sam3_precompute_geom_input -- dynamic count, n_geo = n_boxes + 1.
        helper.make_tensor_value_info("geom_in", TensorProto.FLOAT, [1, "n_geo", D]),
        helper.make_tensor_value_info("img_feats", TensorProto.FLOAT, [1, H, H, D]),
        helper.make_tensor_value_info("fpn0", TensorProto.FLOAT, [1, FPN_HW[0], FPN_HW[0], D]),
        helper.make_tensor_value_info("fpn1", TensorProto.FLOAT, [1, FPN_HW[1], FPN_HW[1], D]),
    ]
    graph_outputs = [
        helper.make_tensor_value_info("class_scores", TensorProto.FLOAT, [1, NQ]),
        helper.make_tensor_value_info("pred_boxes", TensorProto.FLOAT, [1, NQ, 4]),
        helper.make_tensor_value_info("presence_score", TensorProto.FLOAT, [1, 1]),
        helper.make_tensor_value_info("mask_logits", TensorProto.FLOAT, [1, NQ, FPN_HW[0], FPN_HW[0]]),
    ]

    model_def = build_model(g.nodes, g.initializers, graph_inputs, graph_outputs, "sam3_pcs")
    return model_def, ["class_scores", "pred_boxes", "presence_score", "mask_logits"]


def compute_cls_row(mf):
    """The CLS geometry token after final_proj + norm -- the exact value the
    old GEOM_CLS_CONST bake produced, and the exact math
    sam3_precompute_geom_input runs on the CPU for the CLS row."""
    cls_embed = mf.raw("geom.cls_embed.weight").reshape(D)
    x = cls_embed @ mf.linear_w("geom.final_proj.weight") + mf.bias("geom.final_proj.bias")
    norm_w, norm_b = mf.bias("geom.norm.weight"), mf.bias("geom.norm.bias")
    x = (x - x.mean()) / np.sqrt(x.var() + LN_EPS) * norm_w + norm_b
    return x.astype(np.float32)


def main():
    global FENC_Q_CHUNKS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-dir", required=True, help="dir written by sam3_dump_pcs_pvs_weights")
    ap.add_argument("--out", required=True, help="output .onnx path")
    ap.add_argument("--check", action="store_true", help="run an onnxruntime CPU sanity check after export")
    ap.add_argument("--fenc-chunks", type=int, default=0,
                    help="query-row chunks for the fenc self-attention (0 = unchunked, "
                         "required for FP16/mixed builds; 8 = validated whole-FP32 config)")
    args = ap.parse_args()
    FENC_Q_CHUNKS = args.fenc_chunks if args.fenc_chunks > 0 else None

    mf = Manifest(args.export_dir)
    model_def, output_names = build_graph(mf)

    import onnx
    onnx.save(model_def, args.out)
    print(f"saved {args.out} ({len(model_def.graph.node)} nodes, {len(model_def.graph.initializer)} initializers)")

    if args.check:
        import onnxruntime as ort
        rng = np.random.default_rng(0)
        token_ids = np.zeros(TEXT_L, dtype=np.int32)
        token_ids[:5] = [49406, 320, 1929, 269, 49407]  # a plausible short tokenized prompt + padding
        img_feats = (rng.standard_normal((1, H, H, D)) * 0.5).astype(np.float32)
        fpn0 = (rng.standard_normal((1, FPN_HW[0], FPN_HW[0], D)) * 0.1).astype(np.float32)
        fpn1 = (rng.standard_normal((1, FPN_HW[1], FPN_HW[1], D)) * 0.2).astype(np.float32)
        cls_row = compute_cls_row(mf)

        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        # n_geo=1 (CLS only, the old static scope) and n_geo=3 (2 fake
        # exemplar rows + CLS) both must run and stay finite.
        for n_geo in (1, 3):
            geom_in = np.zeros((1, n_geo, D), dtype=np.float32)
            if n_geo > 1:
                geom_in[0, :n_geo - 1, :] = (rng.standard_normal((n_geo - 1, D)) * 0.5).astype(np.float32)
            geom_in[0, -1, :] = cls_row  # CLS last, matching sam3_precompute_geom_input
            outputs = sess.run(output_names, {"token_ids": token_ids, "geom_in": geom_in,
                                              "img_feats": img_feats, "fpn0": fpn0, "fpn1": fpn1})
            print(f"onnxruntime CPU check, n_geo={n_geo} ({len(output_names)} outputs):")
            for name, arr in zip(output_names, outputs):
                finite = np.isfinite(arr).all()
                print(f"  {name:16s} shape={arr.shape} finite={finite} mean={arr.mean():.5f} "
                      f"std={arr.std():.5f} min={arr.min():.5f} max={arr.max():.5f}")
                if not finite:
                    raise SystemExit(f"error: non-finite values in output '{name}'")


if __name__ == "__main__":
    main()
