#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-dcp2-v84-dflash2@sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692}"

case "${IMAGE}" in
  *@sha256:*) ;;
  *)
    echo "Use an immutable IMAGE@sha256:DIGEST reference." >&2
    exit 2
    ;;
esac

docker pull "${IMAGE}" >/dev/null
docker image inspect "${IMAGE}" --format '{{json .Config.Labels}}' | python3 -m json.tool
docker run --rm --entrypoint /bin/sh "${IMAGE}" \
  -c 'exec cat /opt/glm53/PROVENANCE.json' | python3 -m json.tool
