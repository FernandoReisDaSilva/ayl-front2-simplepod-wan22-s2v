#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

source .venv/bin/activate

RUNNER_PREFIX="scripts/simplepod/temp_simplepod_run_"
RUNNER_CHARACTER="mae"
RUNNER_SUFFIX="_wan22_s2v_14_8s_1080_blackwell_natural_v5.py"

AYL_SCRIPT_ID="TEMP_SIMPLEPOD_RUN_ALEX_WAN22_S2V_1080_BLACKWELL_V1" \
python3 "${RUNNER_PREFIX}${RUNNER_CHARACTER}${RUNNER_SUFFIX}" \
  --execute \
  --confirm-start \
  --confirm-inference \
  --confirm-delete \
  --character-id alex \
  --taught-language EN \
  --test-id alex_en_wan22_s2v_1080_v1 \
  --output-stem alex_en_wan22_s2v_14s_1080_blackwell_v1 \
  --input-image-key "tests/simplepod_wan22_s2v/inputs/alex_en_wan22_s2v_720_v1/reference/alex_en_reference.png" \
  --input-audio-key "tests/simplepod_wan22_s2v/inputs/alex_en_wan22_s2v_720_v1/audio/alex_en_test_14s.wav" \
  --width 1080 \
  --height 1080 \
  --ready-timeout-seconds 900 \
  --job-timeout-seconds 3600 \
  --job-poll-interval-seconds 30
