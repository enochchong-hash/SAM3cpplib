"""Shared helpers for the SAM3 -> ONNX offline conversion scripts
(convert_sam3_*_to_onnx.py). Part of the TensorRT migration (see
docs/sam3/PLAN.md) -- offline, one-time tooling, never imported by the
deployed server.

Targets opset 13 for broad TensorRT-parser compatibility. LayerNorm and GELU
are manually decomposed (Reduce/Sqrt/Div and Erf respectively) rather than
using newer native ops (LayerNormalization needs opset17+, Gelu needs
opset20+) -- this avoids depending on parser support beyond what TensorRT
10.13's onnx-tensorrt (pinned release/10.7-GA) generation guarantees.
"""
import json
import os

import numpy as np
from onnx import helper, numpy_helper

OPSET = 13


class Manifest:
    """Reads a manifest.json + raw .bin weight dump (from
    sam3_dump_encoder_weights or a sibling dump tool) written in ggml's own
    tensor-name/shape convention."""

    def __init__(self, export_dir):
        with open(os.path.join(export_dir, "manifest.json")) as f:
            m = json.load(f)
        self.dir = export_dir
        self.entries = {t["name"]: t for t in m["tensors"]}

    def has(self, name):
        return name in self.entries

    def raw(self, name):
        """Raw dequantized f32 buffer reshaped to the reversed-ggml_ne (numpy
        row-major) shape -- always the full 4-tuple, never squeezed (ggml's
        own trailing-1 padding can land at either end after reversal, so
        squeezing generically is ambiguous -- see dump tool's comment)."""
        e = self.entries[name]
        arr = np.fromfile(os.path.join(self.dir, e["file"]), dtype=np.float32)
        ne = e["ggml_ne"]
        return arr.reshape(tuple(reversed(ne)))

    def bias(self, name):
        """1D tensors (LayerNorm weight/bias, conv/linear biases, learned
        embedding vectors)."""
        return np.ascontiguousarray(self.raw(name).reshape(-1))

    def linear_w(self, name):
        """nn.Linear-style weight, ggml ne=[in,out,1,1] -> returns [in,out]
        (pre-transposed so the graph can use plain MatMul(x, w), no Gemm
        transB / no Transpose node needed at every call site).

        raw() gives the full reversed-ne 4-tuple with ggml's trailing-1
        padding now at the *front* (e.g. (1,1,out,in)) -- take the last two
        (real) axes, not the first two."""
        arr = self.raw(name)
        out_dim, in_dim = arr.shape[-2], arr.shape[-1]
        return np.ascontiguousarray(arr.reshape(out_dim, in_dim).T)

    def conv_w(self, name):
        """Conv weight, ggml ne=[kW,kH,Cin,Cout] -> reversed [Cout,Cin,kH,kW],
        already the exact ONNX Conv weight convention, no transpose needed."""
        arr = self.raw(name)
        return np.ascontiguousarray(arr.reshape(arr.shape[:4]))

    def deconv_w(self, name):
        """ConvTranspose weight, ggml ne=[kW,kH,Cout,Cin] -> reversed
        [Cin,Cout,kH,kW], already the exact ONNX ConvTranspose convention."""
        return self.conv_w(name)  # same 4D reversal, different semantic role

    def fused_qkv(self, weight_name, bias_name, d):
        """Fused [D, 3D] in_proj weight/bias (the ggml view-offset pattern
        used for every self/cross-attention outside the ViT/text encoder --
        see sam3_multihead_attn_fused / sam3_fenc_layer_forward in sam3.cpp)
        split into 3 separate (in,out)=(D,D) weight arrays + (D,) biases,
        in (Q,K,V) order -- matches the ggml view offsets 0, D*nb1, 2D*nb1."""
        w = self.linear_w(weight_name)  # (D, 3D), already (in,out)-transposed
        b = self.bias(bias_name)        # (3D,)
        return [(np.ascontiguousarray(w[:, i * d:(i + 1) * d]),
                 np.ascontiguousarray(b[i * d:(i + 1) * d])) for i in range(3)]

    def freqs(self, name):
        """RoPE table. ggml ne=[2,half,rope_n,1] -> reversed (1,rope_n,half,2)
        -> squeeze leading batch -> (rope_n,half,2); split into cos/sin (each
        (rope_n,half)), matching sam3_apply_rope's dim0={0:cos,1:sin}."""
        arr = self.raw(name)
        arr = arr.reshape(arr.shape[-3], arr.shape[-2], arr.shape[-1])
        cos = np.ascontiguousarray(arr[:, :, 0])
        sin = np.ascontiguousarray(arr[:, :, 1])
        return cos, sin


class GraphBuilder:
    def __init__(self):
        self.nodes = []
        self.initializers = []
        self._n = 0

    def _uniq(self, prefix):
        self._n += 1
        return f"{prefix}_{self._n}"

    def const(self, name, arr):
        arr = np.asarray(arr, dtype=np.float32)
        self.initializers.append(numpy_helper.from_array(arr, name=name))
        return name

    def const_i32(self, name, values):
        arr = np.array(values, dtype=np.int32)
        self.initializers.append(numpy_helper.from_array(arr, name=name))
        return name

    def const_i64(self, name, values):
        arr = np.array(values, dtype=np.int64)
        self.initializers.append(numpy_helper.from_array(arr, name=name))
        return name

    def node(self, op_type, inputs, n_outputs=1, name=None, **attrs):
        name = name or self._uniq(op_type)
        outputs = [f"{name}_out"] if n_outputs == 1 else [f"{name}_out{i}" for i in range(n_outputs)]
        self.nodes.append(helper.make_node(op_type, inputs, outputs, name=name, **attrs))
        return outputs[0] if n_outputs == 1 else outputs

    def alias(self, x, out_name):
        """Force a specific graph-output value name via an Identity node."""
        self.nodes.append(helper.make_node("Identity", [x], [out_name], name=out_name + "_id"))
        return out_name


# ── Generic op helpers (all static shapes -- every shape passed in is a
#    concrete Python int known at authoring time, so Reshape targets are
#    always fully explicit constants, never relying on 0/-1 copy tricks). ──
def reshape(g, x, shape, name):
    shp = g.const_i64(name + "_shape", shape)
    return g.node("Reshape", [x, shp], name=name)


def transpose(g, x, perm, name):
    return g.node("Transpose", [x], name=name, perm=perm)


def add(g, a, b, name):
    return g.node("Add", [a, b], name=name)


def sub(g, a, b, name):
    return g.node("Sub", [a, b], name=name)


def mul(g, a, b, name):
    return g.node("Mul", [a, b], name=name)


def matmul(g, a, b, name):
    return g.node("MatMul", [a, b], name=name)


def reduce_sum(g, x, axes, keepdims, name):
    """ReduceSum moved `axes` from attribute to an optional input at opset13
    (unlike ReduceMean, which only changed at opset18) -- pass it as a const."""
    ax = g.const_i64(name + "_axes", axes)
    return g.node("ReduceSum", [x, ax], name=name, keepdims=keepdims)


def concat(g, tensors, axis, name):
    return g.node("Concat", tensors, name=name, axis=axis)


def unsqueeze(g, x, axes, name):
    ax = g.const_i64(name + "_axes", axes)
    return g.node("Unsqueeze", [x, ax], name=name)


def squeeze(g, x, axes, name):
    ax = g.const_i64(name + "_axes", axes)
    return g.node("Squeeze", [x, ax], name=name)


def slice_step(g, x, axis, start, step, name, end=None):
    starts = g.const_i64(name + "_starts", [start])
    ends = g.const_i64(name + "_ends", [np.iinfo(np.int64).max if end is None else end])
    axes = g.const_i64(name + "_axes", [axis])
    steps = g.const_i64(name + "_steps", [step])
    return g.node("Slice", [x, starts, ends, axes, steps], name=name)


def split_n(g, x, sizes, axis, name):
    sz = g.const_i64(name + "_sizes", sizes)
    return g.node("Split", [x, sz], n_outputs=len(sizes), name=name, axis=axis)


def linear(g, x, w_init, b_init, name):
    y = matmul(g, x, w_init, name=name + "_mm")
    if b_init is None:
        return y
    return add(g, y, b_init, name=name + "_bias")


def layer_norm(g, x, w_init, b_init, eps, name):
    # Manual decomposition (not native LayerNormalization, opset17+) -- exact
    # transcription of sam3_layer_norm: mean/var over the last axis, affine.
    mean = g.node("ReduceMean", [x], name=name + "_mean", axes=[-1], keepdims=1)
    centered = sub(g, x, mean, name=name + "_centered")
    sq = mul(g, centered, centered, name=name + "_sq")
    var = g.node("ReduceMean", [sq], name=name + "_var", axes=[-1], keepdims=1)
    eps_c = g.const(name + "_eps", np.float32(eps))
    var_eps = add(g, var, eps_c, name=name + "_var_eps")
    std = g.node("Sqrt", [var_eps], name=name + "_std")
    normed = g.node("Div", [centered, std], name=name + "_normed")
    scaled = mul(g, normed, w_init, name=name + "_scaled")
    return add(g, scaled, b_init, name=name + "_affine")


def gelu_erf(g, x, name):
    # Exact-erf GELU (matches ggml_gelu_erf): 0.5*x*(1+erf(x/sqrt(2)))
    inv_sqrt2 = g.const(name + "_invsqrt2", np.float32(0.7071067811865476))
    half_c = g.const(name + "_half", np.float32(0.5))
    one_c = g.const(name + "_one", np.float32(1.0))
    scaled = mul(g, x, inv_sqrt2, name=name + "_scale")
    erf = g.node("Erf", [scaled], name=name + "_erf")
    erf1 = add(g, erf, one_c, name=name + "_erf1")
    halfx = mul(g, x, half_c, name=name + "_halfx")
    return mul(g, halfx, erf1, name=name + "_out")


def mha(g, q, k, v, num_heads, head_dim, name, mask=None, q_chunk=None, n_q=None):
    """Generic scaled-dot-product multi-head attention on already-split
    per-head tensors q/k/v shaped (batch,heads,Nq_or_Nkv,head_dim). No RoPE,
    no windowing -- callers apply those beforehand. `mask` (if given) is an
    additive bias broadcastable to (batch,heads,Nq,Nkv) (e.g. a causal mask).

    `q_chunk` (with `n_q`, the static query count): compute attention in
    query-row chunks instead of materializing the full (batch,heads,Nq,Nkv)
    scores tensor at once. Softmax normalizes each query row independently,
    so chunking along the query axis is numerically EXACT -- same math, same
    per-row reduction order, just a bounded working set. This exists because
    TensorRT has no fused-attention kernels for FP32 (FP16-only), so an FP32
    engine otherwise materializes the full scores tensor: for the PCS fusion
    encoder's 5184-token self-attention that is 8*5184*5184*4 = 860MB per
    live copy, measured as ~1.9GB of execution-context memory across the
    whole graph. Chunking bounds it to ~(Nq/chunks) of that. Use only where
    that matters -- small attentions gain nothing and it defeats potential
    attention fusion (which is why the default stays unchunked)."""
    kt = transpose(g, k, [0, 1, 3, 2], name=name + "_kt")  # (batch,heads,head_dim,Nkv)
    scale_c = g.const(name + "_scale", np.float32(1.0 / np.sqrt(head_dim)))
    if q_chunk is None:
        scores = matmul(g, q, kt, name=name + "_scores")   # (batch,heads,Nq,Nkv)
        scores = mul(g, scores, scale_c, name=name + "_scaled")
        if mask is not None:
            scores = add(g, scores, mask, name=name + "_masked")
        probs = g.node("Softmax", [scores], name=name + "_softmax", axis=-1)
        return matmul(g, probs, v, name=name + "_attnv")   # (batch,heads,Nq,head_dim)
    assert mask is None, "q_chunk + mask not supported (no call site needs it)"
    assert n_q is not None and n_q % q_chunk == 0, "n_q must be a multiple of q_chunk"
    zero_c = g.const(name + "_chainzero", np.float32(0.0))
    parts = []
    for ci, start in enumerate(range(0, n_q, q_chunk)):
        cn = f"{name}_c{ci}"
        qc = slice_step(g, q, axis=2, start=start, step=1, name=cn + "_q", end=start + q_chunk)
        if parts:
            # Serialization chain: qc += 0 * previous chunk's output. The
            # chunks are otherwise independent, and TensorRT's memory planner
            # was measured to keep several chunks' scores buffers live
            # concurrently (only ~40% ctx-memory saving instead of ~85%).
            # This fake data dependency forces sequential scheduling so one
            # chunk's buffers can be reused for the next. Numerically exact:
            # finite*0 = ±0 and x + ±0 = x in IEEE754 (attention outputs are
            # finite; a -0 query would become +0, which cannot change any
            # downstream product or sum).
            chained = mul(g, parts[-1], zero_c, name=cn + "_chain0")
            qc = add(g, qc, chained, name=cn + "_chained")
        scores = matmul(g, qc, kt, name=cn + "_scores")    # (batch,heads,q_chunk,Nkv)
        scores = mul(g, scores, scale_c, name=cn + "_scaled")
        probs = g.node("Softmax", [scores], name=cn + "_softmax", axis=-1)
        parts.append(matmul(g, probs, v, name=cn + "_attnv"))
    return concat(g, parts, axis=2, name=name + "_attnv")  # (batch,heads,Nq,head_dim)


def split_heads(g, x, batch, n_tok, num_heads, head_dim, name):
    """(batch,N,E) -> (batch,heads,N,head_dim). n_tok=None marks a dynamic
    token dimension (Reshape -1 -- at most one per Reshape, which holds here
    since batch/heads/head_dim are always static)."""
    x = reshape(g, x, [batch, -1 if n_tok is None else n_tok, num_heads, head_dim], name=name + "_r")
    return transpose(g, x, [0, 2, 1, 3], name=name + "_t")


def merge_heads(g, x, batch, n_tok, embed_dim, name):
    """(batch,heads,N,head_dim) -> (batch,N,E). n_tok=None -> dynamic (-1)."""
    x = transpose(g, x, [0, 2, 1, 3], name=name + "_t")
    return reshape(g, x, [batch, -1 if n_tok is None else n_tok, embed_dim], name=name + "_merge")


def fused_mha(g, q_src, kv_src, qkv_consts, out_w, out_b, batch, n_q, n_kv,
              num_heads, head_dim, name, mask=None, k_src=None, v_src=None,
              q_chunk=None):
    """Q from q_src; K from k_src (or kv_src if not given); V from v_src (or
    kv_src). Uses 3 separately-sliced (Q,K,V) weight/bias initializer-name
    pairs (see Manifest.fused_qkv) -- the pattern used by every attention in
    the geometry/fusion/DETR/seg-head stages (sam3_fenc_layer_forward et al).
    K and V often need DIFFERENT effective inputs even in "cross-attention to
    one source" blocks (e.g. geometry encoder: K = img_feats+pe, V = img_feats
    with no pe) -- pass k_src/v_src explicitly whenever that's the case;
    kv_src is only a same-source convenience default for plain self-attn."""
    (wq, bq), (wk, bk), (wv, bv) = qkv_consts
    k_src = kv_src if k_src is None else k_src
    v_src = kv_src if v_src is None else v_src
    q = linear(g, q_src, wq, bq, name=name + "_q")
    k = linear(g, k_src, wk, bk, name=name + "_k")
    v = linear(g, v_src, wv, bv, name=name + "_v")
    q = split_heads(g, q, batch, n_q, num_heads, head_dim, name=name + "_qh")
    k = split_heads(g, k, batch, n_kv, num_heads, head_dim, name=name + "_kh")
    v = split_heads(g, v, batch, n_kv, num_heads, head_dim, name=name + "_vh")
    attn = mha(g, q, k, v, num_heads, head_dim, name=name + "_attn", mask=mask,
               q_chunk=q_chunk, n_q=n_q)
    merged = merge_heads(g, attn, batch, n_q, num_heads * head_dim, name=name + "_merge")
    return linear(g, merged, out_w, out_b, name=name + "_out")


def build_model(graph_nodes, initializers, graph_inputs, graph_outputs, graph_name):
    """Assembles + validates a ModelProto. Does NOT write to disk -- callers
    decide when/where to onnx.save() (main() typically does, after this)."""
    import onnx

    graph_def = helper.make_graph(graph_nodes, graph_name, graph_inputs, graph_outputs,
                                   initializer=initializers)
    model_def = helper.make_model(graph_def, producer_name="sam3.cpp-trt-export",
                                   opset_imports=[helper.make_opsetid("", OPSET)])
    # Pin a conservative IR version -- the installed onnx package may default
    # to a newer IR version than onnxruntime/TensorRT's parser understands.
    model_def.ir_version = 9
    onnx.checker.check_model(model_def)
    return model_def
