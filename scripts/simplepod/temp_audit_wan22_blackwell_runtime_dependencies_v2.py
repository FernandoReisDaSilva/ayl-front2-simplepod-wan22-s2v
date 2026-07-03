#!/usr/bin/env python3
"""Offline runtime dependency audit for Wan2.2 S2V Blackwell.

This audit does not start SimplePod, download weights, or run inference. It
uses a local Wan2.2 checkout if one exists and otherwise reports a partial
build-context audit with explicit source-unavailable risks.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import re
import sys
import sysconfig
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TEST_ID = "TEMP_AUDIT_WAN22_BLACKWELL_RUNTIME_DEPENDENCIES_V2"
REPO_ROOT = Path(__file__).resolve().parents[2]
BLACKWELL_DIR = REPO_ROOT / "docker" / "simplepod-wan22-s2v-fastapi-v2-blackwell"
DOCKERFILE_PATH = BLACKWELL_DIR / "Dockerfile"
REQUIREMENTS_PATH = BLACKWELL_DIR / "requirements.txt"
REPORT_PATH = REPO_ROOT / "logs" / "wan22_blackwell_runtime_dependency_audit_v2.json"

TARGET_WAN_FILES = (
    "generate.py",
    "wan/speech2video.py",
    "wan/modules/attention.py",
    "wan/modules/s2v/model_s2v.py",
)
APP_FILES = (
    BLACKWELL_DIR / "app" / "main.py",
    BLACKWELL_DIR / "app" / "wan22_s2v_generate_wrapper.py",
    BLACKWELL_DIR / "app" / "wan22_s2v_runner.py",
)

FOCUS_IMPORTS = {
    "accelerate": "accelerate",
    "dashscope": "dashscope",
    "decord": "decord",
    "diffusers": "diffusers",
    "einops": "einops",
    "flash_attn": "flash-attn",
    "flash_attn_interface": "flash-attn-interface",
    "librosa": "librosa",
    "moviepy": "moviepy",
    "numpy": "numpy",
    "omegaconf": "omegaconf",
    "peft": "peft",
    "PIL": "Pillow",
    "safetensors": "safetensors",
    "sageattention": "sageattention",
    "scipy": "scipy",
    "soundfile": "soundfile",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
    "tqdm": "tqdm",
    "transformers": "transformers",
    "triton": "triton",
    "xformers": "xformers",
}
PACKAGE_IMPORT_ALIASES = {
    "imageio-ffmpeg": "imageio_ffmpeg",
    "opencv-python-headless": "cv2",
    "pillow": "PIL",
    "python-dotenv": "dotenv",
}
IMPORT_PACKAGE_ALIASES = {
    "cv2": "opencv-python-headless",
    "dotenv": "python-dotenv",
    "imageio_ffmpeg": "imageio-ffmpeg",
    "PIL": "Pillow",
}
KNOWN_S2V_REQUIRED_IMPORTS = {
    "decord",
    "diffusers",
    "einops",
    "numpy",
    "PIL",
    "safetensors",
    "torch",
    "torchvision",
    "tqdm",
    "transformers",
}
KNOWN_RUNTIME_IMPORTS = {
    "accelerate",
    "dashscope",
    "librosa",
    "moviepy",
    "omegaconf",
    "peft",
    "scipy",
    "soundfile",
    "torchaudio",
}
KNOWN_OPTIONAL_ATTENTION_IMPORTS = {"flash_attn", "flash_attn_interface", "sageattention", "xformers", "triton"}
CONDITIONAL_OPTIONAL_IMPORTS = {"cosyvoice"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_package(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def import_to_package(module: str) -> str:
    return normalize_package(IMPORT_PACKAGE_ALIASES.get(module, FOCUS_IMPORTS.get(module, module)))


def package_to_import(package: str) -> str:
    normalized = normalize_package(package)
    return PACKAGE_IMPORT_ALIASES.get(normalized, normalized.replace("-", "_"))


def parse_requirements(path: Path) -> dict[str, Any]:
    packages: dict[str, dict[str, Any]] = {}
    raw_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for line_number, raw in enumerate(raw_lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        package = re.split(r"\s*(?:==|>=|<=|~=|!=|>|<)\s*", line, maxsplit=1)[0].split("[", 1)[0].strip()
        if package:
            packages[normalize_package(package)] = {
                "line_number": line_number,
                "raw": raw,
                "import_name": package_to_import(package),
                "source": "requirements.txt",
            }
    return {"path": str(path), "exists": path.exists(), "packages": packages, "raw_lines": raw_lines}


def parse_dockerfile(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    implicit = {}
    if "torch torchvision torchaudio" in text:
        for package in ("torch", "torchvision", "torchaudio"):
            implicit[package] = {
                "source": "Dockerfile",
                "reason": "Installed from PyTorch cu128 index.",
                "import_name": package,
            }
    clone_url = ""
    match = re.search(r"git clone\s+--depth\s+1\s+(\S+)\s+/opt/Wan2\.2", text)
    if match:
        clone_url = match.group(1)
    return {
        "path": str(path),
        "exists": path.exists(),
        "clone_url": clone_url,
        "implicit_packages": implicit,
        "uses_torch_cu128": "cu128" in text,
    }


def stdlib_names() -> set[str]:
    names = set(getattr(sys, "stdlib_module_names", set()))
    names.update(
        {
            "__future__",
            "argparse",
            "contextlib",
            "copy",
            "datetime",
            "functools",
            "gc",
            "importlib",
            "json",
            "logging",
            "math",
            "os",
            "pathlib",
            "random",
            "re",
            "subprocess",
            "sys",
            "threading",
            "time",
            "traceback",
            "types",
            "typing",
            "warnings",
        }
    )
    stdlib_path = sysconfig.get_paths().get("stdlib")
    if stdlib_path:
        names.add(Path(stdlib_path).name)
    return names


def local_wan_candidates() -> list[Path]:
    candidates = []
    if os.environ.get("WAN22_REPO_DIR"):
        candidates.append(Path(os.environ["WAN22_REPO_DIR"]).expanduser())
    candidates.extend(
        [
            Path("/opt/Wan2.2"),
            REPO_ROOT / "Wan2.2",
            REPO_ROOT.parent / "Wan2.2",
            Path(os.environ.get("TMPDIR", "/tmp")) / "ayl-wan22-dependency-audit" / "Wan2.2",
        ]
    )
    return candidates


def looks_like_wan_repo(path: Path) -> bool:
    return path.exists() and (path / "generate.py").exists() and (path / "wan").is_dir()


def resolve_wan_repo() -> dict[str, Any]:
    checked = []
    for candidate in local_wan_candidates():
        checked.append(str(candidate))
        if looks_like_wan_repo(candidate):
            return {"status": "found_local", "path": str(candidate), "checked_paths": checked}
    return {
        "status": "not_found_local",
        "path": "",
        "checked_paths": checked,
        "note": "Wan2.2 is cloned during Docker build; no local checkout is available for exact static source-line audit.",
    }


def extract_imports(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    imports = []
    errors = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception as exc:
        return [], [{"path": str(path), "error_type": type(exc).__name__, "error_truncated": str(exc)[:500]}]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({"module": alias.name.split(".", 1)[0], "raw": alias.name, "line_number": node.lineno})
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            imports.append({"module": node.module.split(".", 1)[0], "raw": node.module, "line_number": node.lineno})
    return imports, errors


def scan_files(wan_repo_path: Path | None) -> dict[str, Any]:
    stdlib = stdlib_names()
    files = []
    for rel in TARGET_WAN_FILES:
        if wan_repo_path is not None:
            files.append((wan_repo_path / rel, rel, "wan_repo"))
    for path in APP_FILES:
        files.append((path, str(path.relative_to(REPO_ROOT)), "blackwell_app"))

    external: dict[str, dict[str, Any]] = {}
    missing_files = []
    parse_errors = []
    for path, rel, source in files:
        if not path.exists():
            missing_files.append({"path": rel, "source": source})
            continue
        imports, errors = extract_imports(path)
        parse_errors.extend(errors)
        for item in imports:
            module = item["module"]
            if module in stdlib or module in {"wan", "generate", "app"}:
                continue
            record = external.setdefault(
                module,
                {"module": module, "package": import_to_package(module), "sources": [], "sample_locations": [], "count": 0},
            )
            record["count"] += 1
            if source not in record["sources"]:
                record["sources"].append(source)
            if len(record["sample_locations"]) < 12:
                record["sample_locations"].append({"path": rel, "line_number": item["line_number"], "raw": item["raw"]})
    return {"external_imports": external, "missing_files": missing_files, "parse_errors": parse_errors}


def scan_attention_flags(wan_repo_path: Path | None) -> dict[str, Any]:
    wrapper_text = (BLACKWELL_DIR / "app" / "wan22_s2v_generate_wrapper.py").read_text(
        encoding="utf-8",
        errors="replace",
    )
    wrapper_sdpa_patch_detected = (
        "install_sdpa_attention_fallback_patch" in wrapper_text
        and "model_s2v_module.flash_attention" in wrapper_text
        and "attention_module.attention" in wrapper_text
    )
    if wan_repo_path is None:
        return {
            "status": "source_unavailable",
            "has_sdpa_fallback_confirmed": None,
            "wrapper_sdpa_patch_detected": wrapper_sdpa_patch_detected,
            "assert_flash_attn_locations": [],
            "risky_runtime_assertions": [
                {
                    "type": "source_unavailable_attention_backend",
                    "message": "Cannot confirm whether Wan2.2 attention.py has native SDPA fallback or assert FLASH_ATTN_2_AVAILABLE without a local Wan2.2 checkout.",
                    "recommendation": "Before next build or in container import smoke, inspect /opt/Wan2.2/wan/modules/attention.py; if assert FLASH_ATTN_2_AVAILABLE gates execution without fallback, patch to torch.nn.functional.scaled_dot_product_attention or add compatible flash-attn separately.",
                }
            ],
        }
    attention_path = wan_repo_path / "wan" / "modules" / "attention.py"
    if not attention_path.exists():
        return {"status": "missing_attention_py", "has_sdpa_fallback_confirmed": None, "assert_flash_attn_locations": [], "risky_runtime_assertions": []}
    text = attention_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    direct_callers = []
    for rel in ("wan/modules/s2v/model_s2v.py", "wan/modules/model.py"):
        caller_path = wan_repo_path / rel
        if caller_path.exists():
            caller_text = caller_path.read_text(encoding="utf-8", errors="replace")
            if "flash_attention(" in caller_text or "flash_attention," in caller_text:
                direct_callers.append({"path": rel, "uses_flash_attention_directly": True})
    assert_locations = [
        {"path": "wan/modules/attention.py", "line_number": idx, "line": line.strip()}
        for idx, line in enumerate(lines, start=1)
        if "assert" in line and "FLASH_ATTN" in line
    ]
    fallback_terms = ("scaled_dot_product_attention", "torch.nn.functional", "F.scaled_dot_product_attention")
    has_sdpa = any(term in text for term in fallback_terms)
    risky = []
    if assert_locations and direct_callers and not wrapper_sdpa_patch_detected:
        risky.append(
            {
                "type": "direct_flash_attention_call_can_bypass_sdpa_fallback",
                "assert_locations": assert_locations,
                "direct_callers": direct_callers,
                "recommendation": "Patch S2V/model attention call sites to use attention() SDPA fallback, or add a compatible Blackwell/cu128 flash-attn or flash-attn-interface backend.",
            }
        )
    elif assert_locations and not has_sdpa:
        risky.append(
            {
                "type": "flash_attn_assert_without_confirmed_sdpa_fallback",
                "locations": assert_locations,
                "recommendation": "Patch attention.py to use PyTorch SDPA fallback or add flash-attn only after confirming Blackwell/cu128 wheel compatibility.",
            }
        )
    return {
        "status": "scanned",
        "has_sdpa_fallback_confirmed": has_sdpa,
        "wrapper_sdpa_patch_detected": wrapper_sdpa_patch_detected,
        "direct_flash_attention_callers": direct_callers,
        "flash_attention_symbol_seen": "flash_attention" in text,
        "FLASH_ATTN_2_AVAILABLE_seen": "FLASH_ATTN_2_AVAILABLE" in text,
        "sageattention_seen": "sageattention" in text.lower(),
        "xformers_seen": "xformers" in text.lower(),
        "triton_seen": "triton" in text.lower(),
        "assert_flash_attn_locations": assert_locations,
        "risky_runtime_assertions": risky,
    }


def local_import_status(module: str) -> dict[str, Any]:
    try:
        imported = importlib.import_module(module)
        return {"status": "importable_locally", "version": str(getattr(imported, "__version__", ""))}
    except Exception as exc:
        return {"status": "not_importable_locally", "error_type": type(exc).__name__, "error_truncated": str(exc)[:300]}


def classify_dependencies(requirements: dict[str, Any], dockerfile: dict[str, Any], scan: dict[str, Any], attention: dict[str, Any]) -> dict[str, Any]:
    declared = dict(requirements["packages"])
    declared.update(dockerfile["implicit_packages"])
    declared_names = set(declared)
    observed = set(scan["external_imports"])
    if attention.get("FLASH_ATTN_2_AVAILABLE_seen"):
        observed.add("flash_attn")
    if attention.get("sageattention_seen"):
        observed.add("sageattention")
    if attention.get("xformers_seen"):
        observed.add("xformers")
    if attention.get("triton_seen"):
        observed.add("triton")
    observed.update(KNOWN_S2V_REQUIRED_IMPORTS)
    observed.update(KNOWN_RUNTIME_IMPORTS)
    observed.update(KNOWN_OPTIONAL_ATTENTION_IMPORTS)
    observed.update(CONDITIONAL_OPTIONAL_IMPORTS)
    flash_backend_declared = bool({"flash-attn", "flash-attn-interface"} & declared_names)
    direct_flash_attention_risk = any(
        item.get("type") == "direct_flash_attention_call_can_bypass_sdpa_fallback"
        for item in attention.get("risky_runtime_assertions", [])
    )
    wrapper_sdpa_patch_detected = attention.get("wrapper_sdpa_patch_detected") is True

    rows = []
    buckets = {
        "required_present": [],
        "required_missing": [],
        "optional_present": [],
        "optional_missing_with_fallback": [],
        "optional_missing_without_fallback": [],
        "risky_runtime_assertions": attention.get("risky_runtime_assertions", []),
    }
    for module in sorted(observed):
        package = import_to_package(module)
        present = package in declared_names
        required = module in KNOWN_S2V_REQUIRED_IMPORTS or module in KNOWN_RUNTIME_IMPORTS or module in scan["external_imports"]
        optional_attention = module in KNOWN_OPTIONAL_ATTENTION_IMPORTS
        conditional_optional = module in CONDITIONAL_OPTIONAL_IMPORTS
        fallback_confirmed = attention.get("has_sdpa_fallback_confirmed") is True
        row = {
            "module": module,
            "package": package,
            "declared_in_image": present,
            "observed_in_scanned_source": module in scan["external_imports"],
            "known_s2v_required_or_runtime": module in KNOWN_S2V_REQUIRED_IMPORTS or module in KNOWN_RUNTIME_IMPORTS,
            "optional_attention_backend": optional_attention,
            "conditional_optional": conditional_optional,
            "classification": "",
            "sample_locations": scan["external_imports"].get(module, {}).get("sample_locations", []),
            "local_import_status": local_import_status(module),
        }
        if conditional_optional and present:
            row["classification"] = "optional_present"
            row["note"] = "Conditional path; for S2V current gate enable_tts=False."
            buckets["optional_present"].append(row)
        elif conditional_optional:
            row["classification"] = "optional_missing_with_fallback"
            row["note"] = "Conditional TTS path only; current Maé S2V gate uses existing audio and enable_tts=False."
            buckets["optional_missing_with_fallback"].append(row)
        elif module in {"flash_attn", "flash_attn_interface"} and wrapper_sdpa_patch_detected and not flash_backend_declared:
            row["classification"] = "optional_missing_with_fallback"
            row["note"] = "Local Blackwell wrapper patches S2V direct flash_attention calls to attention() SDPA fallback."
            buckets["optional_missing_with_fallback"].append(row)
        elif module in {"flash_attn", "flash_attn_interface"} and direct_flash_attention_risk and not flash_backend_declared:
            row["classification"] = "required_missing"
            row["note"] = "S2V calls flash_attention() directly; one compatible backend is required unless code is patched to use attention() SDPA fallback."
            buckets["required_missing"].append(row)
        elif required and present:
            row["classification"] = "required_present"
            buckets["required_present"].append(row)
        elif required and not present:
            if module == "soundfile" and "librosa" in declared_names:
                row["classification"] = "required_present"
                row["note"] = "Expected as librosa transitive dependency, but not pinned directly."
                buckets["required_present"].append(row)
            else:
                row["classification"] = "required_missing"
                buckets["required_missing"].append(row)
        elif optional_attention and present:
            row["classification"] = "optional_present"
            buckets["optional_present"].append(row)
        elif optional_attention and fallback_confirmed:
            row["classification"] = "optional_missing_with_fallback"
            buckets["optional_missing_with_fallback"].append(row)
        else:
            row["classification"] = "optional_missing_without_fallback"
            buckets["optional_missing_without_fallback"].append(row)
        rows.append(row)
    status = "missing_required_dependencies" if buckets["required_missing"] else "attention_backend_risk" if buckets["optional_missing_without_fallback"] or buckets["risky_runtime_assertions"] else "ok"
    return {"status": status, "rows": rows, **buckets}


def main() -> int:
    requirements = parse_requirements(REQUIREMENTS_PATH)
    dockerfile = parse_dockerfile(DOCKERFILE_PATH)
    wan_repo = resolve_wan_repo()
    wan_path = Path(wan_repo["path"]) if wan_repo.get("path") else None
    scan = scan_files(wan_path)
    attention = scan_attention_flags(wan_path)
    classification = classify_dependencies(requirements, dockerfile, scan, attention)
    report = {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": classification["status"],
        "requirements": requirements,
        "dockerfile": dockerfile,
        "wan_repo": wan_repo,
        "target_wan_files": list(TARGET_WAN_FILES),
        "scan": scan,
        "attention_backend_audit": attention,
        "classification": classification,
        "safety_guards": {
            "calls_simplepod": False,
            "starts_instance": False,
            "downloads_model_weights": False,
            "runs_inference": False,
            "generates_video": False,
            "network_access_attempted": False,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[{TEST_ID}] status={report['status']}")
    print(f"[{TEST_ID}] wan_repo_status={wan_repo['status']}")
    print(f"[{TEST_ID}] required_missing={[row['package'] for row in classification['required_missing']]}")
    print(f"[{TEST_ID}] optional_missing_without_fallback={[row['package'] for row in classification['optional_missing_without_fallback']]}")
    print(f"[{TEST_ID}] risky_runtime_assertions={len(classification['risky_runtime_assertions'])}")
    print(f"[{TEST_ID}] report={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
