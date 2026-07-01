#!/usr/bin/env python3
"""Audit native Wan2.2 CLI args against the Blackwell runner command.

This is an offline/code-only check. It may clone the Wan2.2 code into /tmp,
but it does not download model weights, start SimplePod, or run inference.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TEST_ID = "TEMP_AUDIT_WAN22_BLACKWELL_CLI_ARGS_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "logs" / "wan22_blackwell_cli_args_audit_v1.json"
WAN_REPO_URL = "https://github.com/Wan-Video/Wan2.2.git"
TMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp")) / "ayl-wan22-dependency-audit"
TMP_WAN_REPO_DIR = TMP_ROOT / "Wan2.2"

MODEL_DIR = Path("/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B")
WORK_DIR = Path("/tmp/ayl_wan22_s2v_jobs/mae_fr_wan22_s2v_14_8s_1080_blackwell_natural_v5_native_partial")
INPUT_IMAGE = WORK_DIR / "reference.png"
INPUT_AUDIO = WORK_DIR / "audio.wav"
OUTPUT_VIDEO = WORK_DIR / "mae_fr_wan22_s2v_14_8s_1080_blackwell_natural_v5_native_partial_1080x1080.mp4"

FORWARDED_PARAMETERS = {
    "positive_prompt": (
        "stable square close-up talking head portrait of the same woman, natural French speech articulation, "
        "stronger and more active accurate lip sync, clear mouth openings closures rounded vowels and labial consonants, "
        "slower more natural head motion, subtle controlled head turns and nods, gentle eye neck shoulder and torso micro movements, "
        "preserved identity, high quality face, natural conversational delivery in French"
    ),
    "negative_prompt": (
        "fast head movement, head bobbing, jerky motion, excessive body swaying, exaggerated motion, overacting, "
        "distorted mouth, weak lip sync, blurry face, identity drift, singing performance, subtitles"
    ),
    "seed": 42,
    "steps": 5,
    "cfg": 1.0,
    "shift": 4.0,
    "offload_model": True,
    "convert_model_dtype": True,
    "task": "s2v-14B",
}

NATURAL_V5_REFERENCE_UNSUPPORTED_PARAMETERS = {
    "denoise_strength": 0.80,
    "audio_scale": 1.55,
    "pose_start_percent": 0.0,
    "pose_end_percent": 0.45,
    "num_frames": 237,
}

CHECKED_ARGS = (
    "--negative_prompt",
    "--prompt",
    "--base_seed",
    "--sample_steps",
    "--sample_shift",
    "--sample_guide_scale",
    "--offload_model",
    "--convert_model_dtype",
    "--task",
    "--size",
    "--ckpt_dir",
    "--image",
    "--audio",
    "--save_file",
    "--frame_num",
    "--infer_frames",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_command(command: list[str], cwd: Path | None = None, timeout: int = 120) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "status": "succeeded" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "stdout_truncated": (completed.stdout or "")[-4000:],
            "stderr_truncated": (completed.stderr or "")[-4000:],
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_truncated": str(exc)[:1000],
        }


def looks_like_wan_repo(path: Path) -> bool:
    return path.exists() and (path / "generate.py").exists() and (path / "wan").is_dir()


def local_wan_candidates() -> list[Path]:
    candidates = []
    env_dir = os.environ.get("WAN22_REPO_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.extend(
        [
            Path("/opt/Wan2.2"),
            REPO_ROOT / "Wan2.2",
            REPO_ROOT.parent / "Wan2.2",
            TMP_WAN_REPO_DIR,
        ]
    )
    return candidates


def resolve_wan_repo() -> dict[str, Any]:
    for candidate in local_wan_candidates():
        if looks_like_wan_repo(candidate):
            return {
                "status": "found_local",
                "path": str(candidate),
                "source": "existing_path",
                "clone_attempted": False,
            }

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    if TMP_WAN_REPO_DIR.exists() and not looks_like_wan_repo(TMP_WAN_REPO_DIR):
        return {
            "status": "failed_existing_tmp_path_invalid",
            "path": str(TMP_WAN_REPO_DIR),
            "clone_attempted": False,
            "error_truncated": "Temporary Wan2.2 path exists but does not look like a Wan2.2 repository.",
        }
    if not TMP_WAN_REPO_DIR.exists():
        clone_result = run_command(
            ["git", "clone", "--depth", "1", WAN_REPO_URL, str(TMP_WAN_REPO_DIR)],
            cwd=TMP_ROOT,
            timeout=180,
        )
        if clone_result["status"] != "succeeded" or not looks_like_wan_repo(TMP_WAN_REPO_DIR):
            return {
                "status": "failed_clone",
                "path": str(TMP_WAN_REPO_DIR),
                "clone_attempted": True,
                "clone_result": clone_result,
            }
    return {
        "status": "cloned_or_cached",
        "path": str(TMP_WAN_REPO_DIR),
        "source": "tmp_cache",
        "clone_attempted": True,
    }


def literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def extract_argparse_flags(generate_py: Path) -> dict[str, Any]:
    try:
        tree = ast.parse(generate_py.read_text(encoding="utf-8"), filename=str(generate_py))
    except Exception as exc:
        return {
            "status": "failed_parse",
            "error_type": type(exc).__name__,
            "error_truncated": str(exc)[:1000],
            "accepted_args": [],
            "arg_records": [],
        }

    accepted_args: set[str] = set()
    arg_records: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        option_strings = [
            value
            for value in (literal_string(arg) for arg in node.args)
            if value and value.startswith("-")
        ]
        if not option_strings:
            continue
        accepted_args.update(option_strings)
        arg_records.append(
            {
                "line_number": getattr(node, "lineno", None),
                "option_strings": option_strings,
            }
        )
    return {
        "status": "succeeded",
        "generate_py": str(generate_py),
        "accepted_args": sorted(accepted_args),
        "arg_records": sorted(arg_records, key=lambda item: item["line_number"] or 0),
    }


def run_help_probe(wan_repo_path: Path) -> dict[str, Any]:
    return run_command([sys.executable, "generate.py", "--help"], cwd=wan_repo_path, timeout=30)


def build_runner_command() -> list[str]:
    command = [
        "python",
        "-m",
        "app.wan22_s2v_generate_wrapper",
        "--task",
        str(FORWARDED_PARAMETERS["task"]),
        "--size",
        "1080*1080",
        "--ckpt_dir",
        str(MODEL_DIR),
        "--offload_model",
        "True",
        "--convert_model_dtype",
        "--prompt",
        str(FORWARDED_PARAMETERS["positive_prompt"]),
        "--image",
        str(INPUT_IMAGE),
        "--audio",
        str(INPUT_AUDIO),
        "--save_file",
        str(OUTPUT_VIDEO),
    ]
    command.extend(["--sample_steps", str(FORWARDED_PARAMETERS["steps"])])
    command.extend(["--sample_shift", str(FORWARDED_PARAMETERS["shift"])])
    command.extend(["--sample_guide_scale", str(FORWARDED_PARAMETERS["cfg"])])
    command.extend(["--base_seed", str(FORWARDED_PARAMETERS["seed"])])
    return command


def forwarded_flags(command: list[str]) -> list[str]:
    flags = []
    for item in command:
        if item.startswith("--"):
            flags.append(item)
    return flags


def recommended_changes(rejected_args: list[str], accepted_args: set[str]) -> list[dict[str, str]]:
    changes = []
    if "--negative_prompt" in rejected_args:
        changes.append(
            {
                "arg": "--negative_prompt",
                "recommendation": "Do not forward negative_prompt to native Wan2.2 generate.py; keep it in the job/report only unless native support is confirmed.",
            }
        )
    if "--frame_num" in accepted_args and "--infer_frames" in accepted_args:
        changes.append(
            {
                "arg": "--infer_frames",
                "recommendation": "For S2V duration control, prefer explicit --infer_frames only after deriving a native-safe value; current runner does not forward it.",
            }
        )
    if rejected_args:
        for arg in rejected_args:
            if arg == "--negative_prompt":
                continue
            changes.append(
                {
                    "arg": arg,
                    "recommendation": "Remove or map this forwarded arg before invoking native Wan2.2 generate.py.",
                }
            )
    if not rejected_args:
        changes.append(
            {
                "arg": "",
                "recommendation": "No rejected forwarded CLI args detected for the current runner command.",
            }
        )
    return changes


def main() -> int:
    print(f"[{TEST_ID}] start offline CLI args audit")
    wan_repo = resolve_wan_repo()
    report: dict[str, Any] = {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": "started",
        "wan_repo_url": WAN_REPO_URL,
        "wan_repo": wan_repo,
        "safety_guards": {
            "calls_simplepod": False,
            "starts_instance": False,
            "downloads_model_weights": False,
            "runs_inference": False,
            "generates_video": False,
            "runs_generate_help_only": True,
        },
    }

    if not str(wan_repo.get("status", "")).startswith(("found", "cloned")):
        report["status"] = "failed_wan_repo_unavailable"
        write_json(REPORT_PATH, report)
        print(f"[{TEST_ID}] status={report['status']}")
        print(f"[{TEST_ID}] report={REPORT_PATH}")
        return 0

    wan_repo_path = Path(str(wan_repo["path"]))
    argparse_result = extract_argparse_flags(wan_repo_path / "generate.py")
    help_probe = run_help_probe(wan_repo_path)
    command = build_runner_command()
    flags = forwarded_flags(command)
    accepted_args = set(argparse_result.get("accepted_args", []))
    accepted_forwarded = [flag for flag in flags if flag in accepted_args]
    rejected_forwarded = [flag for flag in flags if flag not in accepted_args]
    checked_arg_status = {
        arg: {
            "accepted_by_native_cli": arg in accepted_args,
            "forwarded_by_runner": arg in flags,
        }
        for arg in CHECKED_ARGS
    }
    status = "rejected_forwarded_args_found" if rejected_forwarded else "all_forwarded_args_accepted"
    report.update(
        {
            "status": status,
            "argparse_extraction": argparse_result,
            "help_probe": help_probe,
            "runner_command": command,
            "runner_forwarded_args": flags,
            "accepted_forwarded_args": accepted_forwarded,
            "rejected_forwarded_args": rejected_forwarded,
            "checked_arg_status": checked_arg_status,
            "parameters_kept_in_report_only": NATURAL_V5_REFERENCE_UNSUPPORTED_PARAMETERS,
            "not_forwarded_parameters": {
                "negative_prompt": {
                    "value": FORWARDED_PARAMETERS["negative_prompt"],
                    "reason": "Native Wan2.2 generate.py does not accept --negative_prompt.",
                }
            },
            "recommended_runner_changes": recommended_changes(rejected_forwarded, accepted_args),
        }
    )
    write_json(REPORT_PATH, report)
    print(f"[{TEST_ID}] status={status}")
    print(f"[{TEST_ID}] accepted_forwarded_args={accepted_forwarded}")
    print(f"[{TEST_ID}] rejected_forwarded_args={rejected_forwarded}")
    print(f"[{TEST_ID}] report={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
