# R10 EXL3/MCG numeric closure

This directory publishes the complete Python source closure required by
`quant_pipeline.codecs.exl3_mcg.Exl3MCGCodec`. The files are byte-identical
to the corrected R10 bundle published in
[`brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78`](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78/tree/7c73450f05a151439d0f184f216b1eefcc394a31/reproducibility/r10)
at immutable revision `7c73450f05a151439d0f184f216b1eefcc394a31`.

## Contents and identity

- `r7_encoder/`: the complete deployed Python encoder package, including
  `r10_codec.R10TrellisCodec` and all supporting modules.
- `lineage/encode_tr3_v31.py`: the pinned v31 numerical core.
- `SOURCE_SHA256SUMS`: SHA-256 for every source file above.
- `verify_bundle.py`: an offline hash, syntax, and import-closure verifier.

The two originally missing identities are:

```text
8b31fb8d1214df63fa1557175a926f6d2d680d69d2cb3689d1df4b5c62a1eded  r7_encoder/r10_codec.py
e9a85a47e165c8d8644354cef611efbb81dfd9ba88544ca59f0c80ee6bc75032  lineage/encode_tr3_v31.py
```

Verify the complete closure:

```bash
python3 reproducibility/r10/verify_bundle.py
```

## Adapter use

The repository does not select machine-specific files implicitly. Point the
adapter at this directory and separately provide the exact compiled
`exllamav3_ext` binary for the target PyTorch, CUDA, and SM architecture:

```python
from pathlib import Path
from quant_pipeline.codecs.exl3_mcg import Exl3MCGCodec

bundle = Path("reproducibility/r10")
codec = Exl3MCGCodec(
    source_root=bundle,
    numeric_core=bundle / "lineage/encode_tr3_v31.py",
    extension=Path("/absolute/path/to/exllamav3_ext.so"),
    device="cuda:0",
)
```

The adapter hashes the Python closure, numeric core, and compiled extension and
fails if any file changes after construction. The extension is intentionally
not bundled because it is ABI- and GPU-architecture-specific.

`encode_tr3_v31.py` preserves the GLM-5.2 production lineage from which the
numeric primitives came. It is not presented as a one-command GLM-5.3 or Qwen
campaign driver; current campaigns consume it through the model-neutral
adapter above.

## Licensing

Repository-authored changes remain under the repository license. Portions of
the pinned numerical core are derived from ExLlamaV3 v0.0.43 and retain its MIT
notice in `THIRD_PARTY_LICENSES/EXLLAMAV3-MIT.txt`.
