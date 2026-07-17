"""Amax calibration for FP8 quantization of the SAM3 PCS head.

Unlike the encoder (fp8_amax_calib.py), only the LINEAR GEMMs of the fusion
encoder and DETR decoder are FP8 targets -- attention BMMs stay FP16 so
TensorRT's fused-MHA pattern (the source of the 97->35ms PCS win) keeps
matching, and the text encoder / geometry / scoring / seg-head / bbox+RPB
MLPs are untouched (see docs/tensorrt.md for the rationale per subsystem).

Inputs are REAL tensors captured by examples/dump_pcs_calib_inputs.cpp
(backbone features + geometry rows from several images, several token sets).
The PCS graph is small enough to calibrate un-split with ORT CPU.

Usage:
  python3 fp8_pcs_amax_calib.py sam3_pcs.onnx <calib_dir> <out.json>
"""
import glob, json, os, re, sys
import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort

src, calib_dir, out_json = sys.argv[1], sys.argv[2], sys.argv[3]

# Linear GEMMs of fenc + ddec only (attention BMMs deliberately excluded).
FP8_GEMM = re.compile(r"^(fenc|ddec)_layer\d+_((sa|ca|ci|ct)_(q|k|v|out)_mm|ffn[01]_mm)$")

# ── Weight amax straight from the initializers ─────────────────────────────
m = onnx.load(src, load_external_data=False)
inits = {i.name: i for i in m.graph.initializer}
weight_amax = {}
targets = []
for n in m.graph.node:
    if n.op_type == "MatMul" and FP8_GEMM.match(n.name):
        targets.append(n)
        if n.input[1] in inits:
            arr = onnx.numpy_helper.to_array(inits[n.input[1]])
            weight_amax[n.input[1]] = float(np.abs(arr).max())
print(f"{len(targets)} target GEMMs, {len(weight_amax)} weights")

# ── Interleave Abs->ReduceMax taps after each activation producer ──────────
g = m.graph
taps = sorted({n.input[0] for n in targets})
tap_idx = {t: i for i, t in enumerate(taps)}
graph_inputs = {i.name for i in g.input} | {i.name for i in g.initializer}
new_nodes = []
for t in taps:
    if t in graph_inputs:
        i = tap_idx[t]
        new_nodes.append(helper.make_node("Abs", [t], [f"ax_{i}"], name=f"ax_{i}"))
        new_nodes.append(helper.make_node("ReduceMax", [f"ax_{i}"], [f"ar_{i}"], name=f"ar_{i}", keepdims=0))
for n in list(g.node):
    new_nodes.append(n)
    for o in n.output:
        if o in tap_idx:
            i = tap_idx[o]
            new_nodes.append(helper.make_node("Abs", [o], [f"ax_{i}"], name=f"ax_{i}"))
            new_nodes.append(helper.make_node("ReduceMax", [f"ax_{i}"], [f"ar_{i}"], name=f"ar_{i}", keepdims=0))
del g.node[:]
g.node.extend(new_nodes)
for i in range(len(taps)):
    g.output.append(helper.make_tensor_value_info(f"ar_{i}", TensorProto.FLOAT, []))

aug = os.path.join(os.path.dirname(out_json) or ".", "pcs_amax_aug.onnx")
onnx.save(m, aug)
del m, g
import gc; gc.collect()

# ── Assemble feeds from the captured inputs ────────────────────────────────
def raw(p, dt=np.float32):
    return np.fromfile(p, dtype=dt)

tok_dtype = np.int32
mm = onnx.load(src, load_external_data=False)
for gi in mm.graph.input:
    if gi.name == "token_ids" and gi.type.tensor_type.elem_type == TensorProto.INT64:
        tok_dtype = np.int64
del mm; gc.collect()

tokens = [raw(p, np.int32).astype(tok_dtype)
          for p in sorted(glob.glob(os.path.join(calib_dir, "tokens_*.bin")))]
n_imgs = len(glob.glob(os.path.join(calib_dir, "img*_geom1.bin")))
assert tokens and n_imgs, "run examples/dump_pcs_calib_inputs first"

feeds = []
for k in range(n_imgs):
    pre = os.path.join(calib_dir, f"img{k}_")
    base = {
        "img_feats": raw(pre + "img_feats.bin.bin").reshape(1, 72, 72, 256),
        "fpn0":      raw(pre + "fpn0.bin.bin").reshape(1, 288, 288, 256),
        "fpn1":      raw(pre + "fpn1.bin.bin").reshape(1, 144, 144, 256),
        "token_ids": tokens[k % len(tokens)],
    }
    feeds.append({**base, "geom_in": raw(pre + "geom1.bin").reshape(1, -1, 256)})
    if k < 2:  # a couple of exemplar-token feeds for the dynamic-geom range
        feeds.append({**base, "geom_in": raw(pre + "geom3.bin").reshape(1, -1, 256)})

# ── Run ─────────────────────────────────────────────────────────────────────
so = ort.SessionOptions()
so.log_severity_level = 3
so.enable_cpu_mem_arena = False
so.enable_mem_pattern = False
sess = ort.InferenceSession(aug, so, providers=["CPUExecutionProvider"])
out_names = [f"ar_{i}" for i in range(len(taps))]
amax = np.zeros(len(taps), dtype=np.float64)
for fi, feed in enumerate(feeds):
    vals = sess.run(out_names, feed)
    amax = np.maximum(amax, np.array([float(v) for v in vals]))
    print(f"  feed {fi + 1}/{len(feeds)} done")
del sess; gc.collect()
os.unlink(aug)

acts = {t: float(a) for t, a in zip(taps, amax)}
with open(out_json, "w") as f:
    json.dump({"activations": acts, "weights": weight_amax}, f, indent=1)
print(f"saved {out_json}: {len(acts)} activation + {len(weight_amax)} weight amax values")
print("largest:", sorted(acts.items(), key=lambda kv: -kv[1])[:3])
