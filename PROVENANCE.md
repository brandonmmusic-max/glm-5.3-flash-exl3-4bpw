# Provenance and attribution

This release uses transparent, content-addressed provenance. It does not use a
hidden watermark, phone-home request, telemetry, or inference-output marker.

The v75 image embeds `/usr/share/glm53/provenance.json` and carries standard OCI
source, revision, author, documentation, version, and license labels plus the
namespaced label `io.github.brandonmmusic-max.glm53.provenance-fingerprint`.
The fingerprint binds the runtime Dockerfile, EXL3 loader, route-128 kernel,
NVFP4 attention implementation, and 46-layer scale bank used to prepare the
release. The registry digest binds the complete published image.

Inspect and verify a pulled image with:

```bash
runtime/verify-provenance.sh \
  verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-v75@sha256:RELEASE_DIGEST
```

The canonical public history is the combination of:

- the immutable Docker/OCI digest and attached build attestations;
- the Git commit history and release receipt in this repository;
- the Hugging Face model revision and copied release receipt;
- the hashes inside the embedded provenance manifest.

These records make copied or modified artifacts correlatable; they do not make
removal impossible and do not, by themselves, prove legal infringement. The
applicable attribution and provenance-retention requirements are in `LICENSE`.

