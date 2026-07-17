Model weights (`.ggml`) go here and are gitignored because each checkpoint is
roughly 1–4 GB.

Acquire the same ready-to-use checkpoint as `release/sam3`:

```bash
./scripts/download_models.sh q8_0
```

The downloader uses
`https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam3-VARIANT.ggml`,
downloads through an atomic `.part` file, and validates the SAM3 file magic.
Run `./scripts/download_models.sh --help` for mirrors, custom destinations,
all variants, and offline import from another deployment.

To create weights yourself, use
`scripts/convert/convert_sam3_to_ggml.py` and the quantization example.

The library source is MIT-licensed, but SAM3 model weights remain subject to
Meta's SAM model license. Review that license before redistributing a
downloaded checkpoint; the setup script intentionally downloads weights at
deployment time instead of bundling them with this repository.
