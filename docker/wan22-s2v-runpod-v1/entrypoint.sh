#!/usr/bin/env bash
set -euo pipefail

MODE="${AYL_RUN_MODE:-${1:-idle}}"

echo "[AYL_WAN22_S2V_ENTRYPOINT] start"
echo "[AYL_WAN22_S2V_ENTRYPOINT] mode=${MODE}"

STATUS="ok"
case "${MODE}" in
  idle)
    sleep infinity
    ;;
  wan22_s2v_probe)
    python /opt/ayl/runtime_probe.py --mode wan22_s2v_probe
    ;;
  *)
    echo "[AYL_WAN22_S2V_ENTRYPOINT] unknown mode" >&2
    STATUS="unknown_mode"
    exit 2
    ;;
esac

echo "[AYL_WAN22_S2V_ENTRYPOINT] done status=${STATUS}"
