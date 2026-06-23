#!/usr/bin/env bash
set -euo pipefail

MODE="${AYL_RUN_MODE:-${1:-idle}}"
STATUS="started"

echo "[AYL_WAN27_ENTRYPOINT] start"
echo "[AYL_WAN27_ENTRYPOINT] mode=${MODE}"

case "$MODE" in
  idle)
    STATUS="idle"
    if [[ "${AYL_IDLE_SECONDS:-}" == "" || "${AYL_IDLE_SECONDS:-}" == "infinity" ]]; then
      sleep infinity
    else
      sleep "$AYL_IDLE_SECONDS"
    fi
    ;;
  wan27_probe)
    python /opt/ayl/runtime_probe.py --mode wan27_probe
    STATUS="ok"
    ;;
  *)
    echo "[AYL_WAN27_ENTRYPOINT] unknown mode" >&2
    STATUS="unknown_mode"
    exit 64
    ;;
esac

echo "[AYL_WAN27_ENTRYPOINT] done status=${STATUS}"
