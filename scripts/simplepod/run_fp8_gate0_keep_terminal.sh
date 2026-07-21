#!/usr/bin/env bash
set -u

NO_PAUSE=0
for arg in "$@"; do
  case "$arg" in
    --no-pause)
      NO_PAUSE=1
      ;;
    *)
      echo "Argumento desconhecido: $arg" >&2
      echo "Uso: $0 [--no-pause]" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1

mkdir -p logs
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/fp8_gate0_keep_terminal_${TIMESTAMP}.log"

echo "[FP8_GATE0] repo=$REPO_ROOT"
echo "[FP8_GATE0] log=$LOG_PATH"
echo "[FP8_GATE0] image=ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.3.06-blackwell-fp8-wan-gate0-path-resolution-v1"
echo "[FP8_GATE0] template_id=26108"
echo "[FP8_GATE0] starting paid SimplePod probe"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
  echo "[FP8_GATE0] virtualenv=.venv"
else
  echo "[FP8_GATE0] virtualenv=not_found_using_system_python"
fi

set +e
python3 scripts/simplepod/temp_simplepod_fp8_runtime_probe_v1.py \
  --template-id 26108 \
  --execute \
  --confirm-start \
  --confirm-delete \
  --startup-timeout-seconds 1200 \
  --probe-timeout-seconds 900 \
  --poll-interval-seconds 10 \
  --debug-startup-classification \
  --debug-probe-monitor 2>&1 | tee "$LOG_PATH"
PYTHON_EXIT=${PIPESTATUS[0]}
set -e

echo
echo "[FP8_GATE0] EXECUCAO CONCLUIDA"
echo "[FP8_GATE0] exit_code=$PYTHON_EXIT"
echo "[FP8_GATE0] log=$LOG_PATH"

if [ "$NO_PAUSE" -eq 1 ] || [ ! -t 0 ]; then
  exit "$PYTHON_EXIT"
fi

echo
echo "Pressione ENTER para abrir um shell neste terminal."
read -r _
exec "${SHELL:-/bin/zsh}" -l
