"""Split-graph amax calibration for FP8 quantization of the SAM3 encoder.
Full-model ORT inference peaks ~14.7GB RAM (OOM on this box) and >8GB VRAM,
so the model is cut at the block15/block16 boundary and calibrated in two
sequential passes, chaining the 21MB boundary tensor. Taps (Abs->ReduceMax,
interleaved after producers) cover the FP8-target tensors: GEMM activation
inputs and attention BMM inputs; softmax outputs are analytic (amax=1)."""
import glob, json, os, sys
import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.utils import extract_model
import onnxruntime as ort

src, calib_dir, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
BOUNDARY = "block15_res2_out"
scratch = os.path.dirname(out_json)

# ── Weight amax from the original initializers (no inference needed) ─────
m = onnx.load(src, load_external_data=False)
inits = {i.name: i for i in m.graph.initializer}
weight_amax = {}
gemm_weight_of = {}
for n in m.graph.node:
    if n.op_type == "MatMul" and any(t in n.name for t in ("_qkv_mm", "_proj_mm", "_fc1_mm", "_fc2_mm")):
        if n.input[1] in inits:
            arr = onnx.numpy_helper.to_array(inits[n.input[1]])
            weight_amax[n.input[1]] = float(np.abs(arr).max())
del m, inits
import gc; gc.collect()

# ── Split ──────────────────────────────────────────────────────────────────
halfA, halfB = os.path.join(scratch, "enc_halfA.onnx"), os.path.join(scratch, "enc_halfB.onnx")
if not (os.path.exists(halfA) and os.path.exists(halfB)):
    print("splitting model at", BOUNDARY, "...")
    full_outputs = [o.name for o in onnx.load(src, load_external_data=False).graph.output]
    extract_model(src, halfA, ["image"], [BOUNDARY])
    extract_model(src, halfB, [BOUNDARY], full_outputs)
    print("split done")

def augment_and_run(model_path, feed_name, feeds_iter, out_extra=None):
    """Add interleaved amax taps to `model_path`, run all feeds, return
    (amax dict, analytic dict, list of extra-output arrays per feed)."""
    m = onnx.load(model_path)
    g = m.graph
    producer = {o: n for n in g.node for o in n.output}
    graph_inits = {i.name for i in g.initializer}
    taps, analytic = set(), {}

    def add_act(t):
        p = producer.get(t)
        if p is not None and p.op_type == "Softmax":
            analytic[t] = 1.0
        else:
            taps.add(t)

    for n in g.node:
        if n.op_type != "MatMul" or "block" not in n.name:
            continue
        if any(t in n.name for t in ("_qkv_mm", "_proj_mm", "_fc1_mm", "_fc2_mm")):
            add_act(n.input[0])
        elif "_attn_scores" in n.name or "_attn_attnv" in n.name:
            add_act(n.input[0]); add_act(n.input[1])

    tap_list = sorted(taps)
    tap_idx = {t: i for i, t in enumerate(tap_list)}
    new_nodes = []
    gi = {i.name for i in g.input}
    for t in tap_list:  # taps on graph inputs first
        if t in gi or t in graph_inits:
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
    for i in range(len(tap_list)):
        g.output.append(helper.make_tensor_value_info(f"ar_{i}", TensorProto.FLOAT, []))

    aug = model_path + ".aug"
    onnx.save(m, aug)
    del m, g, producer
    gc.collect()

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    so.add_session_config_entry("session.disable_prepacking", "1")
    sess = ort.InferenceSession(aug, so, providers=["CPUExecutionProvider"])
    out_names = [f"ar_{i}" for i in range(len(tap_list))]
    extras = []
    amax = np.zeros(len(tap_list), dtype=np.float64)
    for fi, feed in enumerate(feeds_iter):
        wanted = out_names + ([out_extra] if out_extra else [])
        vals = sess.run(wanted, {feed_name: feed})
        amax = np.maximum(amax, np.array([float(v) for v in vals[:len(tap_list)]]))
        if out_extra:
            extras.append(vals[-1])
        print(f"    feed {fi+1} done")
    del sess
    gc.collect()
    os.unlink(aug)
    d = {t: float(a) for t, a in zip(tap_list, amax)}
    d.update(analytic)
    return d, extras

files = sorted(glob.glob(os.path.join(calib_dir, "calib_*.bin")))
images = [np.fromfile(f, dtype=np.float32).reshape(1, 3, 1008, 1008) for f in files]
print(f"pass A ({len(images)} images)...")
amaxA, boundaries = augment_and_run(halfA, "image", images, out_extra=BOUNDARY)
del images; gc.collect()
print("pass B...")
amaxB, _ = augment_and_run(halfB, BOUNDARY, boundaries)

acts = {}
acts.update(amaxA); acts.update(amaxB)
with open(out_json, "w") as f:
    json.dump({"activations": acts, "weights": weight_amax}, f, indent=1)
print(f"saved {out_json}: {len(acts)} activation + {len(weight_amax)} weight amax values")
big = sorted(acts.items(), key=lambda kv: -kv[1])[:5]
print("largest activation amax:", [(k, round(v, 1)) for k, v in big])
