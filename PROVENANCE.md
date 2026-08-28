# Provenance and attribution

This release uses transparent, content-addressed provenance. It does not use a
hidden watermark, phone-home request, telemetry, or inference-output marker.

The current v84 image embeds `/opt/glm53/PROVENANCE.json` and carries standard
OCI source, revision, author, documentation, version, checkpoint, and release
labels. The manifest binds the corrected Triton DFlash mask, INT4 attention
call path, vision RoPE fallback, multimodal template, and validation receipts
with SHA-256 hashes. The registry digest binds the complete published image.

Inspect and verify a pulled image with:

```bash
runtime/verify-provenance.sh \
  verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-dcp2-v84-dflash2@sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692
```

The canonical public history is the combination of:

- the immutable Docker/OCI digest and attached build attestations;
- the Git commit history and release receipt in this repository;
- the Hugging Face model revision and copied release receipt;
- the hashes inside the embedded provenance manifest.

These records make copied or modified artifacts correlatable; they do not make
removal impossible and do not, by themselves, prove legal infringement. The
applicable attribution and provenance-retention requirements are in `LICENSE`.
