#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:?usage: verify-provenance.sh IMAGE@sha256:DIGEST}"
EXPECTED_FINGERPRINT="sha256:508ca365c13c06ff79a61ae921d108ada77efc9cadf54e1d780c975c341bff2a"

case "${IMAGE}" in
  *@sha256:*) ;;
  *)
    echo "Use an immutable IMAGE@sha256:DIGEST reference." >&2
    exit 2
    ;;
esac

docker pull "${IMAGE}" >/dev/null
ACTUAL_FINGERPRINT="$({
  docker image inspect "${IMAGE}" \
    --format '{{index .Config.Labels "io.github.brandonmmusic-max.glm53.provenance-fingerprint"}}'
} 2>/dev/null)"

if [[ "${ACTUAL_FINGERPRINT}" != "${EXPECTED_FINGERPRINT}" ]]; then
  echo "fingerprint mismatch: expected ${EXPECTED_FINGERPRINT}, got ${ACTUAL_FINGERPRINT}" >&2
  exit 1
fi

MANIFEST="$({
  docker run --rm --entrypoint /bin/sh "${IMAGE}" \
    -c 'exec cat /usr/share/glm53/provenance.json'
})"
MANIFEST_FINGERPRINT="$(printf '%s' "${MANIFEST}" | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["runtime_bundle_fingerprint"])')"

if [[ "${MANIFEST_FINGERPRINT}" != "${EXPECTED_FINGERPRINT}" ]]; then
  echo "embedded manifest mismatch: expected ${EXPECTED_FINGERPRINT}, got ${MANIFEST_FINGERPRINT}" >&2
  exit 1
fi

docker image inspect "${IMAGE}" --format '{{json .Config.Labels}}' | python3 -m json.tool
printf '%s\n' "${MANIFEST}" | python3 -m json.tool
echo "provenance verified: ${EXPECTED_FINGERPRINT}"
