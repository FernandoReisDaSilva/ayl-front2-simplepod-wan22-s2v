#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

OUT_DIR="data/simplepod_wan22_s2v/inputs/alex_en_wan22_s2v_720_v1/audio"
OUT_WAV="$OUT_DIR/alex_en_test_14s.wav"
TMP_JSON="$OUT_DIR/alex_en_test_14s_google_tts_response.json"

TEXT='You know these words. You know “going to.” You know “want to.”'

mkdir -p "$OUT_DIR"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "ERROR: gcloud not found. Install Google Cloud CLI or use an API key flow."
  exit 1
fi

ACCESS_TOKEN="$(gcloud auth print-access-token)"

cat > "$OUT_DIR/request.json" <<JSON
{
  "input": {
    "text": "$TEXT"
  },
  "voice": {
    "languageCode": "en-US",
    "name": "en-US-Neural2-D",
    "ssmlGender": "MALE"
  },
  "audioConfig": {
    "audioEncoding": "LINEAR16",
    "speakingRate": 0.92,
    "pitch": -1.0,
    "sampleRateHertz": 24000
  }
}
JSON

curl -sS \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "x-goog-user-project: gen-lang-client-0143729042" \
  -H "Content-Type: application/json; charset=utf-8" \
  --data @"$OUT_DIR/request.json" \
  "https://texttospeech.googleapis.com/v1/text:synthesize" \
  > "$TMP_JSON"

python3 - <<PY
import base64, json
from pathlib import Path

tmp = Path("$TMP_JSON")
out = Path("$OUT_WAV")

data = json.loads(tmp.read_text())
if "error" in data:
    raise SystemExit("Google TTS error: " + json.dumps(data["error"], indent=2))

audio = base64.b64decode(data["audioContent"])
out.write_bytes(audio)
print("WAV written:", out)
print("bytes:", out.stat().st_size)
PY
