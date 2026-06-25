import argparse
import json
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path


TEST_ID = "TTS_ALEX_GOOGLE_TTS_AYL_0001"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "tts_generation_input.json"
OUTPUT_DIR = REPO_ROOT / "data" / "audio" / "AYL_0001" / "alex"
MANIFEST_PATH = OUTPUT_DIR / "audio_manifest.json"
LOG_PATH = REPO_ROOT / "logs" / "tts_alex_google_tts_AYL_0001_log.json"
LANGUAGE_CODE = "en-US"
PREFERRED_VOICE = "en-US-Chirp3-HD-Alnilam"
SAMPLE_RATE_HERTZ = 24000
PITCH_SEMITONES = -1.0
PITCH_CONTRACT_CANDIDATE = "-1st"
VOICE_MAPPING_CANDIDATE = "Alex / EN-US / Alnilam / pitch=-1st"
DEPENDENCY_NOTE = "python3 -m pip install google-cloud-texttospeech"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_input(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def segment_list(data) -> list[dict]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        raise RuntimeError("Input JSON must be an object or a list of segments.")
    for key in ("segments", "items", "lines", "tts_segments"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    if "ssml" in data and "output_file" in data:
        return [data]
    raise RuntimeError("Input JSON must contain a segment list key such as 'segments' or 'items'.")


def normalize_segments(data) -> tuple[list[dict], list[str]]:
    problems = []
    normalized = []
    for index, item in enumerate(segment_list(data)):
        if not isinstance(item, dict):
            problems.append(f"segment {index} is not an object")
            continue
        ssml = str(item.get("ssml", "")).strip()
        output_file = str(item.get("output_file", "")).strip()
        if not ssml:
            problems.append(f"segment {index} missing ssml")
        if not output_file:
            problems.append(f"segment {index} missing output_file")
        if output_file and Path(output_file).name != output_file:
            problems.append(f"segment {index} output_file must be a file name only: {output_file}")
        normalized.append(
            {
                "index": index,
                "id": item.get("id", item.get("segment_id", f"segment_{index:04d}")),
                "ssml": ssml,
                "output_file": output_file,
                "text_reference_present": "text_reference" in item,
            }
        )
    return normalized, problems


def import_texttospeech():
    try:
        from google.cloud import texttospeech
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency 'google-cloud-texttospeech'. Install it with: {DEPENDENCY_NOTE}") from exc
    return texttospeech


def gender_name(texttospeech, gender: object) -> str:
    try:
        return texttospeech.SsmlVoiceGender(gender).name
    except Exception:
        return str(gender)


def choose_voice(client, texttospeech) -> tuple[object, dict]:
    voices = list(client.list_voices(language_code=LANGUAGE_CODE).voices)
    preferred = [voice for voice in voices if getattr(voice, "name", "") == PREFERRED_VOICE]
    if preferred:
        return preferred[0], {"voice_fallback_used": False, "voice_fallback_reason": ""}

    male_voices = [
        voice
        for voice in voices
        if LANGUAGE_CODE in list(getattr(voice, "language_codes", []))
        and gender_name(texttospeech, getattr(voice, "ssml_gender", "")) == "MALE"
    ]
    if not male_voices:
        raise RuntimeError(f"No native {LANGUAGE_CODE} male Google TTS voices returned by ADC project.")

    def score(voice) -> tuple[int, str]:
        name = getattr(voice, "name", "")
        lower_name = name.lower()
        value = 0
        if "chirp3-hd" in lower_name or "chirp3_hd" in lower_name:
            value += 100
        if "chirp" in lower_name:
            value += 60
        if "studio" in lower_name:
            value += 40
        if "neural" in lower_name:
            value += 20
        if "wavenet" in lower_name:
            value += 10
        if "standard" in lower_name:
            value -= 20
        return value, name

    voice = sorted(male_voices, key=score, reverse=True)[0]
    return voice, {
        "voice_fallback_used": True,
        "voice_fallback_reason": f"{PREFERRED_VOICE} not available; selected consistent native {LANGUAGE_CODE} male voice.",
    }


def wav_duration_seconds(path: Path) -> dict:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        sample_width = wav_file.getsampwidth()
    return {
        "duration_seconds": frames / sample_rate if sample_rate else 0.0,
        "sample_rate_hertz": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frame_count": frames,
    }


def build_common_log(args: argparse.Namespace, input_path: Path, segments: list[dict], problems: list[str], status: str, error: str = "") -> dict:
    return {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": status,
        "error": error,
        "problems": problems,
        "dry_run": not args.execute,
        "execute": args.execute,
        "input_path": str(input_path),
        "output_dir": str(OUTPUT_DIR),
        "manifest_path": str(MANIFEST_PATH),
        "language_code": LANGUAGE_CODE,
        "preferred_voice": PREFERRED_VOICE,
        "sample_rate_hertz": SAMPLE_RATE_HERTZ,
        "audio_encoding": "LINEAR16",
        "channels": 1,
        "pitch_semitones": PITCH_SEMITONES,
        "pitch_requested": PITCH_CONTRACT_CANDIDATE,
        "voice_mapping_candidate": VOICE_MAPPING_CANDIDATE,
        "uses_adc": True,
        "uses_api_key": False,
        "no_runpod": True,
        "not_latentsync": True,
        "not_wan": True,
        "not_wan22": True,
        "segments": segments,
    }


def voice_omits_pitch_by_default(voice_name: str) -> bool:
    lower_name = voice_name.lower()
    return "chirp3-hd" in lower_name or "chirp3_hd" in lower_name or "alnilam" in lower_name


def build_audio_config(texttospeech, *, include_pitch: bool):
    kwargs = {
        "audio_encoding": texttospeech.AudioEncoding.LINEAR16,
        "sample_rate_hertz": SAMPLE_RATE_HERTZ,
    }
    if include_pitch:
        kwargs["pitch"] = PITCH_SEMITONES
    return texttospeech.AudioConfig(**kwargs)


def supports_pitch_error(exc: Exception) -> bool:
    return "does not support pitch" in str(exc).lower()


def synthesize_segments(args: argparse.Namespace, input_path: Path, segments: list[dict]) -> tuple[list[dict], dict]:
    texttospeech = import_texttospeech()
    client = texttospeech.TextToSpeechClient()
    voice, voice_info = choose_voice(client, texttospeech)
    voice_name = getattr(voice, "name", "")
    voice_gender = gender_name(texttospeech, getattr(voice, "ssml_gender", ""))

    voice_params = texttospeech.VoiceSelectionParams(
        language_code=LANGUAGE_CODE,
        name=voice_name,
        ssml_gender=getattr(voice, "ssml_gender", texttospeech.SsmlVoiceGender.MALE),
    )
    include_pitch = not voice_omits_pitch_by_default(voice_name)
    pitch_omitted_reason = "voice_does_not_support_pitch" if not include_pitch else ""
    pitch_retry_without_pitch = False

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rendered = []
    for segment in segments:
        output_path = OUTPUT_DIR / segment["output_file"]
        synthesis_input = texttospeech.SynthesisInput(ssml=segment["ssml"])
        try:
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=build_audio_config(texttospeech, include_pitch=include_pitch),
            )
        except Exception as exc:
            if not include_pitch or not supports_pitch_error(exc):
                raise
            include_pitch = False
            pitch_retry_without_pitch = True
            pitch_omitted_reason = "voice_does_not_support_pitch"
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=build_audio_config(texttospeech, include_pitch=False),
            )
        output_path.write_bytes(response.audio_content)
        facts = wav_duration_seconds(output_path)
        rendered.append(
            {
                **segment,
                "output_path": str(output_path),
                "size_bytes": output_path.stat().st_size,
                "pitch_applied": include_pitch,
                **facts,
            }
        )

    manifest = {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "input_path": str(input_path),
        "output_dir": str(OUTPUT_DIR),
        "voice_name": voice_name,
        "actual_voice_name": voice_name,
        "voice_gender": voice_gender,
        "voice_mapping_candidate": VOICE_MAPPING_CANDIDATE,
        "language_code": LANGUAGE_CODE,
        "audio_encoding": "LINEAR16",
        "sample_rate_hertz": SAMPLE_RATE_HERTZ,
        "channels": 1,
        "pitch_semitones": PITCH_SEMITONES,
        "pitch_contract_candidate": PITCH_CONTRACT_CANDIDATE,
        "pitch_requested": PITCH_CONTRACT_CANDIDATE,
        "pitch_applied": include_pitch,
        "pitch_omitted_reason": pitch_omitted_reason,
        "pitch_retry_without_pitch": pitch_retry_without_pitch,
        **voice_info,
        "segments": rendered,
    }
    write_json(MANIFEST_PATH, manifest)
    return rendered, manifest


def run(args: argparse.Namespace) -> int:
    input_path = resolve_path(args.input)
    segments: list[dict] = []
    problems: list[str] = []
    try:
        if not input_path.is_file():
            status = "input_missing"
            problems.append(f"input JSON not found: {input_path}")
            write_json(LOG_PATH, build_common_log(args, input_path, segments, problems, status))
            print(f"[{TEST_ID}] DONE status={status} input={input_path} log={LOG_PATH}")
            return 0 if not args.execute else 1

        data = load_input(input_path)
        segments, problems = normalize_segments(data)
        for segment in segments:
            segment["planned_output_path"] = str(OUTPUT_DIR / segment["output_file"]) if segment["output_file"] else ""
        if problems:
            status = "invalid_input"
            write_json(LOG_PATH, build_common_log(args, input_path, segments, problems, status))
            for problem in problems:
                print(f"[{TEST_ID}] {problem}")
            print(f"[{TEST_ID}] DONE status={status} log={LOG_PATH}")
            return 0 if not args.execute else 1

        if not args.execute:
            status = "dry_run_ready"
            write_json(LOG_PATH, build_common_log(args, input_path, segments, [], status))
            print(f"[{TEST_ID}] START dry_run=true segments={len(segments)}")
            for segment in segments:
                print(f"[{TEST_ID}] PLAN output={segment['output_file']} ssml_chars={len(segment['ssml'])}")
            print(f"[{TEST_ID}] DONE status={status} log={LOG_PATH}")
            return 0

        rendered, manifest = synthesize_segments(args, input_path, segments)
        status = "succeeded"
        log = build_common_log(args, input_path, rendered, [], status)
        log.update(
            {
                "manifest": manifest,
                "voice_name": manifest["voice_name"],
                "voice_gender": manifest["voice_gender"],
                "voice_fallback_used": manifest["voice_fallback_used"],
                "voice_fallback_reason": manifest["voice_fallback_reason"],
                "voice_mapping_candidate": manifest["voice_mapping_candidate"],
                "actual_voice_name": manifest["actual_voice_name"],
                "pitch_requested": manifest["pitch_requested"],
                "pitch_applied": manifest["pitch_applied"],
                "pitch_omitted_reason": manifest["pitch_omitted_reason"],
                "pitch_retry_without_pitch": manifest["pitch_retry_without_pitch"],
            }
        )
        write_json(LOG_PATH, log)
        print(f"[{TEST_ID}] DONE status={status} files={len(rendered)} manifest={MANIFEST_PATH} log={LOG_PATH}")
        return 0
    except Exception as exc:
        message = str(exc)
        write_json(LOG_PATH, build_common_log(args, input_path, segments, problems, "failed", message))
        print(f"[{TEST_ID}] ERROR {message[:500]}", file=sys.stderr)
        print(f"[{TEST_ID}] DONE status=failed log={LOG_PATH}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Alex Google Cloud TTS WAV files from SSML JSON using ADC.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Input JSON with ssml/output_file segments.")
    parser.add_argument("--execute", action="store_true", help="Perform real Google Cloud TTS synthesis using ADC.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
