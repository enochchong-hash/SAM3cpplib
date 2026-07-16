"""Inject FP8 (E4M3) QuantizeLinear/DequantizeLinear pairs into the SAM3
image-encoder ONNX for TensorRT explicit FP8 quantization, following the
fused-MHA pattern TensorRT documents (Q/DQ on both BMM1 inputs, and on the
softmax output + V feeding BMM2) plus FP8 GEMMs for qkv/proj/fc1/fc2.
Scales are per-tensor amax/448 from fp8_amax_calib.py. The model is first
version-converted to opset 19 (FP8 types require it)."""
import json, sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, version_converter

src, amax_json, dst = sys.argv[1], sys.argv[2], sys.argv[3]
FP8_MAX = 448.0

m = onnx.load(src)
print("converting opset 13 -> 19 ...")
# The version converter copies the whole ModelProto several times; on a ~1.9GB
# weights-embedded model that transiently needs >10GB and gets OOM-killed on a
# 32GB workstation with servers resident. It only rewrites node structure
# (e.g. ReduceMean's axes attribute -> input) and never reads weight payloads,
# so swap each initializer for a zero-payload external-data stub (same name/
# dtype/dims, so symbol resolution still passes) around the conversion, then
# restore the real payloads afterwards.
detached = {i.name: i for i in m.graph.initializer}
stubs = []
for i in m.graph.initializer:
    s = TensorProto()
    s.name = i.name
    s.data_type = i.data_type
    s.dims.extend(i.dims)
    s.data_location = TensorProto.EXTERNAL
    e = s.external_data.add(); e.key = "location"; e.value = "detached.stub"
    stubs.append(s)
del m.graph.initializer[:]
m.graph.initializer.extend(stubs)
m = version_converter.convert_version(m, 19)
restored = [detached.get(i.name, i) for i in m.graph.initializer]
del m.graph.initializer[:]
m.graph.initializer.extend(restored)
del detached, stubs, restored
g = m.graph

with open(amax_json) as f:
    amax = json.load(f)
act_amax = amax["activations"]
w_amax = amax["weights"]

counter = [0]
new_nodes = []
qdq_cache = {}

def qdq_for(tensor, scale_val, tag):
    """Q->DQ for `tensor` (per-tensor fp8 scale), inserted in-place into
    new_nodes so topological order holds; cached per (tensor, scale)."""
    key = (tensor, round(scale_val, 12))
    if key in qdq_cache:
        return qdq_cache[key]
    i = counter[0]; counter[0] += 1
    sc = helper.make_tensor(f"fp8_sc_{i}", TensorProto.FLOAT, [], [max(scale_val, 1e-6)])
    zp = helper.make_tensor(f"fp8_zp_{i}", TensorProto.FLOAT8E4M3FN, [], [0.0])
    g.initializer.extend([sc, zp])
    qn, dn = f"fp8_q_{i}_{tag}", f"fp8_dq_{i}_{tag}"
    new_nodes.append(helper.make_node("QuantizeLinear", [tensor, f"fp8_sc_{i}", f"fp8_zp_{i}"], [qn], name=qn))
    new_nodes.append(helper.make_node("DequantizeLinear", [qn, f"fp8_sc_{i}", f"fp8_zp_{i}"], [dn], name=dn))
    qdq_cache[key] = dn
    return dn

n_gemm = n_bmm = 0
for n in list(g.node):
    # Insert this node's Q/DQ producers BEFORE the node itself.
    if n.op_type == "MatMul" and "block" in n.name:
        if any(t in n.name for t in ("_qkv_mm", "_proj_mm", "_fc1_mm", "_fc2_mm")):
            a, w = n.input[0], n.input[1]
            if a in act_amax and w in w_amax:
                n.input[0] = qdq_for(a, act_amax[a] / FP8_MAX, "act")
                n.input[1] = qdq_for(w, w_amax[w] / FP8_MAX, "wgt")
                n_gemm += 1
        elif "_attn_scores" in n.name or "_attn_attnv" in n.name:
            for idx in (0, 1):
                t = n.input[idx]
                if t in act_amax:
                    n.input[idx] = qdq_for(t, act_amax[t] / FP8_MAX, "bmm")
            n_bmm += 1
    new_nodes.append(n)

del g.node[:]
g.node.extend(new_nodes)

print(f"FP8-quantized {n_gemm} GEMMs and {n_bmm} attention BMMs "
      f"({counter[0]} Q/DQ pairs)")
onnx.checker.check_model(m)
onnx.save(m, dst)
import os
print(f"saved {dst} ({os.path.getsize(dst)/1e6:.0f} MB)")
