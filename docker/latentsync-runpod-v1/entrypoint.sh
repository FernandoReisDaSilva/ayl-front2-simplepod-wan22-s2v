#!/usr/bin/env bash
set -euo pipefail

MODE="${AYL_RUN_MODE:-${1:-idle}}"
STATUS="started"

echo "[AYL_ENTRYPOINT] start"
echo "[AYL_ENTRYPOINT] mode=${MODE}"

case "$MODE" in
  idle)
    STATUS="idle"
    if [[ "${AYL_IDLE_SECONDS:-}" == "" || "${AYL_IDLE_SECONDS:-}" == "infinity" ]]; then
      sleep infinity
    else
      sleep "$AYL_IDLE_SECONDS"
    fi
    ;;
  r2_probe)
    python /opt/ayl/runtime_probe.py --mode r2_probe
    STATUS="ok"
    ;;
  latentsync_probe)
    python /opt/ayl/runtime_probe.py --mode latentsync_probe
    STATUS="ok"
    ;;
  latentsync_run)
    echo "[AYL_ENTRYPOINT] latentsync_run reserved" >&2
    STATUS="reserved"
    exit 2
    ;;
  *)
    echo "[AYL_ENTRYPOINT] unknown mode" >&2
    STATUS="unknown_mode"
    exit 64
    ;;
esac

echo "[AYL_ENTRYPOINT] done status=${STATUS}"
