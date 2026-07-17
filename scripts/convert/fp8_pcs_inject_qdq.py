"""Inject FP8 (E4M3) QuantizeLinear/DequantizeLinear pairs into the SAM3 PCS
ONNX -- LINEAR GEMMs of the fusion encoder and DETR decoder only.

Deliberate differences from the encoder's fp8_inject_qdq.py:
  * attention BMMs are NOT quantized: PCS's speed comes from TensorRT's
    fused FP16 MHA (97->35 ms), and Q/DQ on the BMM inputs would break the
    fusion pattern match;
  * text encoder, geometry encoder, scoring, seg head, and the bbox/RPB/qpos
    MLPs are untouched (FP32-pinned text via mixed:text_, tiny or
    numerically delicate elsewhere -- docs/tensorrt.md has the full map).

Scales are per-tensor amax/448 from fp8_pcs_amax_calib.py. The model is
version-converted 13->19 first (FP8 types need opset 19); weight payloads
are detached around the conversion to keep peak memory low.

Usage: python3 fp8_pcs_inject_qdq.py sam3_pcs.onnx amax.json sam3_pcs_fp8.onnx
"""
import json, re, sys
import onnx
from onnx import TensorProto, helper, version_converter

src, amax_json, dst = sys.argv[1], sys.argv[2], sys.argv[3]
FP8_MAX = 448.0
FP8_GEMM = re.compile(r"^(fenc|ddec)_layer\d+_((sa|ca|ci|ct)_(q|k|v|out)_mm|ffn[01]_mm)$")

m = onnx.load(src)
print("converting opset 13 -> 19 ...")
# Same memory trick as the encoder script: the version converter round-trips
# the whole ModelProto through several serialized copies but never reads
# weight payloads -- swap initializers for zero-payload external-data stubs
# (name/dtype/dims kept so symbol resolution passes), restore afterwards.
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

n_gemm = 0
for n in list(g.node):
    if n.op_type == "MatMul" and FP8_GEMM.match(n.name):
        a, w = n.input[0], n.input[1]
        if a in act_amax and w in w_amax:
            n.input[0] = qdq_for(a, act_amax[a] / FP8_MAX, "act")
            n.input[1] = qdq_for(w, w_amax[w] / FP8_MAX, "wgt")
            n_gemm += 1
    new_nodes.append(n)

del g.node[:]
g.node.extend(new_nodes)

print(f"FP8-quantized {n_gemm} linear GEMMs ({counter[0]} Q/DQ pairs); "
      f"attention BMMs left FP16 (fused MHA preserved)")
onnx.checker.check_model(m)
onnx.save(m, dst)
import os
print(f"saved {dst} ({os.path.getsize(dst)/1e6:.0f} MB)")
