#!/usr/bin/env python3
"""Extract the small runtime-data sidecar needed by TensorRT deployments.

The TensorRT engines already own the inference weights.  The C++ runtime still
needs model hyperparameters, the embedded BPE tokenizer, sparse-PVS prompt
embeddings, and the CPU geometry-input projections.  This tool copies exactly
those records from a full SAM3 checkpoint into a versioned .sam3rt file.

The output deliberately uses a different magic from a full checkpoint so the
runtime cannot accidentally treat an arbitrary/partial file as a fallback
model.
"""

import argparse
import os
import struct
from pathlib import Path


SAM3_MAGIC = 0x73616D33       # "sam3"
SAM3_TRT_MAGIC = 0x74723373   # "s3rt"
SAM3_VERSION = 3
N_HPARAMS = 39

# ggml type id -> (block size, bytes per block).  These are the only types the
# SAM3 v3 converter emits and the C++ loader accepts.
TYPE_LAYOUT = {
    0: (1, 4),    # F32
    1: (1, 2),    # F16
    2: (32, 18),  # Q4_0
    3: (32, 20),  # Q4_1
    8: (32, 34),  # Q8_0
}

GEOM_HELPERS = {
    "geom.boxes_direct_project.weight",
    "geom.boxes_direct_project.bias",
    "geom.label_embed.weight",
    "geom.cls_embed.weight",
    "geom.boxes_pos_enc_project.weight",
    "geom.boxes_pos_enc_project.bias",
    "geom.boxes_pool_project.weight",
    "geom.boxes_pool_project.bias",
    "geom.img_pre_norm.weight",
    "geom.img_pre_norm.bias",
    "geom.final_proj.weight",
    "geom.final_proj.bias",
    "geom.norm.weight",
    "geom.norm.bias",
}


def read_exact(fin, size):
    data = fin.read(size)
    if len(data) != size:
        raise ValueError("unexpected end of checkpoint")
    return data


def tensor_nbytes(shape, dtype):
    try:
        block, block_bytes = TYPE_LAYOUT[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported ggml tensor dtype {dtype}") from exc
    elements = 1
    for dim in shape:
        elements *= dim
    if shape[0] % block:
        raise ValueError(f"tensor row width {shape[0]} is not divisible by block {block}")
    return (shape[0] // block) * block_bytes * (elements // shape[0])


def write_tensor(fout, record):
    name, shape, dtype, data = record
    encoded = name.encode("utf-8")
    fout.write(struct.pack("<iii", len(shape), len(encoded), dtype))
    fout.write(struct.pack("<" + "i" * len(shape), *shape))
    fout.write(encoded)
    fout.write(b"\0" * ((-fout.tell()) % 32))
    fout.write(data)


def extract(source: Path, output: Path):
    with source.open("rb") as fin:
        magic, version, ftype, n_tensors = struct.unpack("<Iiii", read_exact(fin, 16))
        if magic != SAM3_MAGIC:
            raise ValueError(f"not a full SAM3 checkpoint (magic=0x{magic:08x})")
        if version != SAM3_VERSION:
            raise ValueError(f"unsupported SAM3 version {version}")
        hparams = read_exact(fin, N_HPARAMS * 4)

        retained = []
        retained_payload = 0
        for _ in range(n_tensors):
            n_dims, name_len, dtype = struct.unpack("<iii", read_exact(fin, 12))
            if n_dims <= 0 or n_dims > 4 or name_len <= 0:
                raise ValueError("invalid tensor record")
            shape = struct.unpack("<" + "i" * n_dims, read_exact(fin, n_dims * 4))
            name = read_exact(fin, name_len).decode("utf-8")
            fin.seek((-fin.tell()) % 32, os.SEEK_CUR)
            size = tensor_nbytes(shape, dtype)
            keep = name.startswith("sam_pe.") or name in GEOM_HELPERS
            if keep:
                data = read_exact(fin, size)
                retained.append((name, shape, dtype, data))
                retained_payload += size
            else:
                fin.seek(size, os.SEEK_CUR)

        # The tokenizer section is already independently versioned by its own
        # magic and extends to EOF. Visual-only models legitimately have none.
        tokenizer = fin.read()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as fout:
        fout.write(struct.pack("<Iiii", SAM3_TRT_MAGIC, version, ftype, len(retained)))
        fout.write(hparams)
        for record in retained:
            write_tensor(fout, record)
        fout.write(tokenizer)
    os.replace(temporary, output)

    print(
        f"wrote {output}: {len(retained)} helper tensors, "
        f"{retained_payload / (1024 * 1024):.2f} MiB tensor payload, "
        f"{len(tokenizer) / (1024 * 1024):.2f} MiB tokenizer, "
        f"{output.stat().st_size / (1024 * 1024):.2f} MiB total"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="full SAM3 v3 .ggml checkpoint")
    parser.add_argument("output", type=Path, help="output .sam3rt runtime-data sidecar")
    args = parser.parse_args()
    extract(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
