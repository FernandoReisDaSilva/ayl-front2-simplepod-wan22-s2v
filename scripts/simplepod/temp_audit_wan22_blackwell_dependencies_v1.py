#!/usr/bin/env python3
"""Offline dependency audit for the Blackwell Wan2.2 image.

This script scans Wan2.2 Python imports and compares them with the
Blackwell image requirements without starting SimplePod, downloading model
weights, or running inference.
"""

from __future__ import annotations

import ast
import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import sysconfig
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TEST_ID = "TEMP_AUDIT_WAN22_BLACKWELL_DEPENDENCIES_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
BLACKWELL_DIR = REPO_ROOT / "docker" / "simplepod-wan22-s2v-fastapi-v2-blackwell"
DOCKERFILE_PATH = BLACKWELL_DIR / "Dockerfile"
REQUIREMENTS_PATH = BLACKWELL_DIR / "requirements.txt"
APP_FILES = [
    BLACKWELL_DIR / "app" / "wan22_s2v_generate_wrapper.py",
    BLACKWELL_DIR / "app" / "wan22_s2v_runner.py",
]
REPORT_PATH = REPO_ROOT / "logs" / "wan22_blackwell_dependency_audit_v1.json"
WAN_REPO_URL = "https://github.com/Wan-Video/Wan2.2.git"
TMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp")) / "ayl-wan22-dependency-audit"
TMP_WAN_REPO_DIR = TMP_ROOT / "Wan2.2"

FOCUS_IMPORTS = {
    "accelerate": "accelerate",
    "av": "av",
    "cv2": "opencv-python-headless",
    "dashscope": "dashscope",
    "decord": "decord",
    "diffusers": "diffusers",
    "easydict": "easydict",
    "einops": "einops",
    "ftfy": "ftfy",
    "huggingface_hub": "huggingface_hub",
    "imageio": "imageio",
    "imageio_ffmpeg": "imageio-ffmpeg",
    "librosa": "librosa",
    "moviepy": "moviepy",
    "numpy": "numpy",
    "omegaconf": "omegaconf",
    "PIL": "Pillow",
    "regex": "regex",
    "safetensors": "safetensors",
    "scipy": "scipy",
    "soundfile": "soundfile",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
    "tqdm": "tqdm",
    "transformers": "transformers",
}

DEFERRED_OPTIONAL_PACKAGES = {
    "cosyvoice",
    "flash-attn",
    "hydra",
    "matplotlib",
    "sam2",
}

IMPORT_TO_PACKAGE = {
    **FOCUS_IMPORTS,
    "PIL": "Pillow",
    "cv2": "opencv-python-headless",
    "dotenv": "python-dotenv",
    "fastapi": "fastapi",
    "boto3": "boto3",
    "botocore": "boto3",
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
}

PACKAGE_TO_IMPORT = {
    "opencv-python-headless": "cv2",
    "opencv-python": "cv2",
    "pillow": "PIL",
    "python-dotenv": "dotenv",
    "imageio-ffmpeg": "imageio_ffmpeg",
    "pyyaml": "yaml",
}

MANUAL_STDLIB = {
    "__future__",
    "argparse",
    "array",
    "ast",
    "asyncio",
    "base64",
    "collections",
    "contextlib",
    "copy",
    "csv",
    "dataclasses",
    "datetime",
    "decimal",
    "enum",
    "functools",
    "glob",
    "hashlib",
    "importlib",
    "inspect",
    "io",
    "itertools",
    "json",
    "logging",
    "math",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "platform",
    "queue",
    "random",
    "re",
    "shutil",
    "signal",
    "socket",
    "statistics",
    "string",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "time",
    "traceback",
    "types",
    "typing",
    "urllib",
    "uuid",
    "warnings",
    "weakref",
    "zipfile",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_package_name(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def module_to_requirement_name(module_name: str) -> str:
    package_name = IMPORT_TO_PACKAGE.get(module_name, module_name)
    return normalize_package_name(package_name)


def requirement_to_import_name(package_name: str) -> str:
    normalized = normalize_package_name(package_name)
    return PACKAGE_TO_IMPORT.get(normalized, normalized.replace("-", "_"))


def parse_requirements(path: Path) -> dict[str, Any]:
    packages: dict[str, dict[str, Any]] = {}
    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        package_part = re.split(r"\s*(?:==|>=|<=|~=|!=|>|<)\s*", line, maxsplit=1)[0]
        package_part = package_part.split("[", 1)[0].strip()
        if not package_part:
            continue
        normalized = normalize_package_name(package_part)
        packages[normalized] = {
            "source": "requirements.txt",
            "line_number": line_number,
            "raw": raw_line,
            "import_name": requirement_to_import_name(package_part),
        }
    return {
        "path": str(path),
        "exists": path.exists(),
        "packages": packages,
        "raw_lines": lines,
    }


def parse_dockerfile_implicit_packages(path: Path) -> dict[str, dict[str, Any]]:
    implicit: dict[str, dict[str, Any]] = {}
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if "pip install" in text and "torch torchvision torchaudio" in text:
        for name in ("torch", "torchvision", "torchaudio"):
            implicit[normalize_package_name(name)] = {
                "source": "Dockerfile",
                "reason": "Installed explicitly from PyTorch cu128 index.",
                "import_name": name,
            }
    return implicit


def stdlib_modules() -> set[str]:
    names = set(MANUAL_STDLIB)
    names.update(getattr(sys, "stdlib_module_names", set()))
    stdlib_path = sysconfig.get_paths().get("stdlib")
    if stdlib_path:
        names.add(Path(stdlib_path).name)
    return names


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
            "stdout_truncated": (completed.stdout or "")[-2000:],
            "stderr_truncated": (completed.stderr or "")[-2000:],
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_truncated": str(exc)[:1000],
        }


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


def looks_like_wan_repo(path: Path) -> bool:
    return path.exists() and (path / "generate.py").exists() and (path / "wan").is_dir()


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
            "source": "tmp_cache",
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
                "source": "git_clone",
                "clone_attempted": True,
                "clone_result": clone_result,
            }

    return {
        "status": "cloned_or_cached",
        "path": str(TMP_WAN_REPO_DIR),
        "source": "tmp_cache",
        "clone_attempted": True,
    }


def repo_internal_top_level_names(repo_path: Path) -> set[str]:
    names = {"wan", "generate"}
    for child in repo_path.iterdir():
        if child.name.startswith("."):
            continue
        if child.is_dir() and (child / "__init__.py").exists():
            names.add(child.name)
        elif child.is_file() and child.suffix == ".py":
            names.add(child.stem)
    for child in repo_path.rglob("*.py"):
        if not any(part.startswith(".") for part in child.relative_to(repo_path).parts):
            names.add(child.stem)
    return names


def extract_imports_from_file(path: Path) -> list[dict[str, Any]]:
    imports: list[dict[str, Any]] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception as exc:
        return [
            {
                "module": None,
                "line_number": None,
                "error_type": type(exc).__name__,
                "error_truncated": str(exc)[:500],
            }
        ]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_name = alias.name.split(".", 1)[0]
                imports.append({"module": top_name, "line_number": node.lineno, "raw": alias.name})
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            if not node.module:
                continue
            top_name = node.module.split(".", 1)[0]
            imports.append({"module": top_name, "line_number": node.lineno, "raw": node.module})
    return imports


def scan_imports(wan_repo_path: Path) -> dict[str, Any]:
    stdlib = stdlib_modules()
    internal = repo_internal_top_level_names(wan_repo_path)
    import_records: dict[str, dict[str, Any]] = {}
    scanned_files = 0
    parse_errors: list[dict[str, Any]] = []

    py_files = sorted(wan_repo_path.rglob("*.py"))
    source_files = [(path, "wan_repo") for path in py_files]
    source_files.extend((path, "blackwell_app") for path in APP_FILES if path.exists())

    for path, source in source_files:
        scanned_files += 1
        for item in extract_imports_from_file(path):
            module = item.get("module")
            rel_path = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
            if not module:
                parse_errors.append({"path": rel_path, **item})
                continue
            category = "external"
            if module in {"wan", "generate"}:
                category = "internal_wan"
            elif module in stdlib:
                category = "stdlib"
            elif source == "wan_repo" and module in internal:
                category = "internal_wan"
            elif source == "blackwell_app" and module == "app":
                category = "internal_app"
            record = import_records.setdefault(
                module,
                {
                    "module": module,
                    "category": category,
                    "sources": [],
                    "sample_locations": [],
                    "count": 0,
                },
            )
            record["count"] += 1
            if source not in record["sources"]:
                record["sources"].append(source)
            if len(record["sample_locations"]) < 12:
                record["sample_locations"].append(
                    {"path": rel_path, "line_number": item.get("line_number"), "raw": item.get("raw")}
                )
            if record["category"] == "external" and category != "external":
                record["category"] = category

    external = {
        module: record
        for module, record in sorted(import_records.items())
        if record["category"] == "external"
    }
    return {
        "scanned_files": scanned_files,
        "parse_errors": parse_errors,
        "internal_top_level_names": sorted(internal),
        "external_imports": external,
        "all_import_count": len(import_records),
    }


def import_status(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
        return {"status": "importable", "version": str(version) if version is not None else None}
    except Exception as exc:
        return {
            "status": "not_importable_locally",
            "error_type": type(exc).__name__,
            "error_truncated": str(exc)[:500],
        }


def declared_import_modules(requirements: dict[str, Any], dockerfile_packages: dict[str, Any]) -> set[str]:
    modules = set()
    for package in requirements["packages"].values():
        modules.add(str(package["import_name"]).split(".", 1)[0])
    for package in dockerfile_packages.values():
        modules.add(str(package["import_name"]).split(".", 1)[0])
    return modules


def run_generate_global_import_probe_once(wan_repo_path: Path, stub_top_modules: set[str], timeout: int = 30) -> dict[str, Any]:
    code = r'''
import importlib.abc
import importlib.machinery
import json
import sys
import traceback
import types

wan_repo = sys.argv[1]
stub_tops = set(json.loads(sys.argv[2]))

class Dummy:
    def __mro_entries__(self, bases):
        return ()
    def __call__(self, *args, **kwargs):
        return self
    def __iter__(self):
        return iter(())
    def __bool__(self):
        return False
    def __len__(self):
        return 0
    def __getitem__(self, _key):
        return self
    def __getattr__(self, _name):
        return self

class StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__file__ = "<stubbed_by_dependency_audit>"
        module.__path__ = []
        module.__package__ = spec.name
        module.__all__ = []
        def _getattr(_name):
            value = Dummy()
            setattr(module, _name, value)
            return value
        module.__getattr__ = _getattr
        return module
    def exec_module(self, module):
        return None

class StubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".", 1)[0]
        if top in stub_tops:
            return importlib.machinery.ModuleSpec(fullname, StubLoader(), is_package=True)
        return None

sys.meta_path.insert(0, StubFinder())
sys.path.insert(0, wan_repo)

try:
    import generate  # noqa: F401
    print(json.dumps({"status": "import_generate_ok"}))
except ModuleNotFoundError as exc:
    print(json.dumps({
        "status": "missing_module",
        "missing_module": exc.name,
        "error_type": type(exc).__name__,
        "error_truncated": str(exc)[:1000],
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }))
    raise SystemExit(2)
except Exception as exc:
    print(json.dumps({
        "status": "failed_non_module_not_found",
        "error_type": type(exc).__name__,
        "error_truncated": str(exc)[:1000],
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }))
    raise SystemExit(3)
'''
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(wan_repo_path),
                json.dumps(sorted(stub_top_modules)),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        return {
            "status": "failed_probe_subprocess",
            "error_type": type(exc).__name__,
            "error_truncated": str(exc)[:1000],
        }
    stdout = (completed.stdout or "").strip()
    parsed = None
    for line in reversed(stdout.splitlines()):
        try:
            parsed = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    return {
        "status": parsed.get("status", "failed_unparseable_probe_output") if isinstance(parsed, dict) else "failed_unparseable_probe_output",
        "returncode": completed.returncode,
        "parsed": parsed,
        "stdout_truncated": (completed.stdout or "")[-2000:],
        "stderr_truncated": (completed.stderr or "")[-2000:],
    }


def run_generate_global_import_probe(
    wan_repo_path: Path,
    requirements: dict[str, Any],
    dockerfile_packages: dict[str, Any],
    max_iterations: int,
) -> dict[str, Any]:
    declared_packages = dict(requirements["packages"])
    declared_packages.update(dockerfile_packages)
    declared_names = set(declared_packages)
    stub_top_modules = declared_import_modules(requirements, dockerfile_packages)
    missing_modules: list[str] = []
    iterations: list[dict[str, Any]] = []

    for index in range(1, max_iterations + 1):
        result = run_generate_global_import_probe_once(wan_repo_path, stub_top_modules)
        parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
        iteration = {
            "iteration": index,
            "status": result["status"],
            "returncode": result.get("returncode"),
            "missing_module": parsed.get("missing_module"),
            "error_type": parsed.get("error_type"),
            "error_truncated": parsed.get("error_truncated"),
            "traceback_tail": parsed.get("traceback_tail", []),
        }
        iterations.append(iteration)
        if result["status"] == "import_generate_ok":
            break
        if result["status"] != "missing_module" or not parsed.get("missing_module"):
            break
        missing_module = str(parsed["missing_module"]).split(".", 1)[0]
        if missing_module not in missing_modules:
            missing_modules.append(missing_module)
        stub_top_modules.add(missing_module)

    missing_packages = sorted({module_to_requirement_name(module) for module in missing_modules})
    missing_not_declared = sorted(package for package in missing_packages if package not in declared_names)
    deferred_optional = sorted(package for package in missing_not_declared if package in DEFERRED_OPTIONAL_PACKAGES)
    recommended = sorted(package for package in missing_not_declared if package not in DEFERRED_OPTIONAL_PACKAGES)
    final_status = iterations[-1]["status"] if iterations else "not_attempted"
    return {
        "status": final_status,
        "attempted": True,
        "command_intent": f"{sys.executable} -c \"import sys; sys.path.insert(0, '<WAN22_REPO>'); import generate\"",
        "uses_gpu": False,
        "downloads_model_weights": False,
        "runs_inference": False,
        "max_iterations": max_iterations,
        "iterations": iterations,
        "missing_modules_detected": missing_modules,
        "missing_packages_detected": missing_packages,
        "missing_packages_not_declared": missing_not_declared,
        "recommended_requirements_additions": recommended,
        "deferred_optional_additions": deferred_optional,
        "stubbed_declared_or_previously_missing_modules": sorted(stub_top_modules),
    }


def build_dependency_audit(requirements: dict[str, Any], dockerfile_packages: dict[str, Any], scan: dict[str, Any]) -> dict[str, Any]:
    declared_packages = dict(requirements["packages"])
    declared_packages.update(dockerfile_packages)
    declared_names = set(declared_packages)
    external_imports = scan["external_imports"]

    external_dependency_rows = []
    missing_from_requirements = []
    declared_external = []
    for module, record in external_imports.items():
        package_name = module_to_requirement_name(module)
        declared = package_name in declared_names
        s2v_relevant = any(
            "/wan/speech2video.py" in location["path"]
            or "/wan/modules/s2v/" in location["path"]
            or location["path"].endswith("/generate.py")
            or location["path"].endswith("wan22_s2v_generate_wrapper.py")
            or location["path"].endswith("wan22_s2v_runner.py")
            for location in record["sample_locations"]
        )
        row = {
            "module": module,
            "package": package_name,
            "declared_in_image_requirements": declared,
            "is_focus_dependency": module in FOCUS_IMPORTS,
            "s2v_relevant_sample_seen": s2v_relevant,
            "import_count": record["count"],
            "sources": record["sources"],
            "sample_locations": record["sample_locations"],
        }
        external_dependency_rows.append(row)
        if declared:
            declared_external.append(row)
        else:
            missing_from_requirements.append(row)

    focus_rows = []
    for module, package in FOCUS_IMPORTS.items():
        package_name = normalize_package_name(package)
        observed = module in external_imports
        declared = package_name in declared_names
        record = external_imports.get(module, {})
        s2v_relevant = any(
            "/wan/speech2video.py" in location["path"]
            or "/wan/modules/s2v/" in location["path"]
            or location["path"].endswith("/generate.py")
            or location["path"].endswith("wan22_s2v_generate_wrapper.py")
            or location["path"].endswith("wan22_s2v_runner.py")
            for location in record.get("sample_locations", [])
        )
        focus_rows.append(
            {
                "module": module,
                "package": package_name,
                "observed_in_wan22_or_app_imports": observed,
                "declared_in_image_requirements": declared,
                "s2v_relevant_sample_seen": s2v_relevant,
                "local_import_status": import_status(module),
                "sample_locations": external_imports.get(module, {}).get("sample_locations", []),
            }
        )

    missing_focus = [
        row
        for row in focus_rows
        if row["observed_in_wan22_or_app_imports"] and not row["declared_in_image_requirements"]
    ]
    probably_missing = [
        row
        for row in missing_from_requirements
        if row["module"] in FOCUS_IMPORTS or row["import_count"] >= 2
    ]

    recommended_focus_packages = sorted({row["package"] for row in missing_focus})
    recommended_s2v_packages = sorted(
        {
            row["package"]
            for row in missing_from_requirements
            if row["s2v_relevant_sample_seen"]
        }
    )
    recommended_full_repo_packages = sorted({row["package"] for row in missing_focus + probably_missing})
    deferred_optional_additions = sorted(
        package
        for package in set(recommended_focus_packages) | set(recommended_s2v_packages)
        if package in DEFERRED_OPTIONAL_PACKAGES
    )
    recommended_packages = sorted(
        package
        for package in set(recommended_focus_packages) | set(recommended_s2v_packages)
        if package not in DEFERRED_OPTIONAL_PACKAGES
    )
    status = "missing_dependencies_found" if recommended_packages else "no_missing_focus_dependencies_found"

    return {
        "status": status,
        "declared_packages": declared_packages,
        "external_dependency_rows": external_dependency_rows,
        "missing_from_requirements": missing_from_requirements,
        "probably_missing": probably_missing,
        "focus_dependency_rows": focus_rows,
        "missing_focus_dependencies": missing_focus,
        "already_present_focus_dependencies": [
            row for row in focus_rows if row["declared_in_image_requirements"]
        ],
        "recommended_requirements_additions": recommended_packages,
        "recommended_focus_additions": recommended_focus_packages,
        "recommended_s2v_additions": recommended_s2v_packages,
        "deferred_optional_additions": deferred_optional_additions,
        "optional_full_repo_additions": [
            package
            for package in recommended_full_repo_packages
            if package not in recommended_packages and package not in deferred_optional_additions
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Blackwell Wan2.2 Python dependencies offline.")
    parser.add_argument(
        "--skip-global-import-check",
        action="store_true",
        help="Skip the subprocess import-generate probe.",
    )
    parser.add_argument(
        "--global-import-max-iterations",
        type=int,
        default=80,
        help="Maximum ModuleNotFoundError discovery iterations for import generate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[{TEST_ID}] start offline dependency audit")
    requirements = parse_requirements(REQUIREMENTS_PATH)
    dockerfile_packages = parse_dockerfile_implicit_packages(DOCKERFILE_PATH)
    wan_repo = resolve_wan_repo()

    report: dict[str, Any] = {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": "started",
        "dockerfile_path": str(DOCKERFILE_PATH),
        "requirements_path": str(REQUIREMENTS_PATH),
        "blackwell_app_files": [str(path) for path in APP_FILES],
        "wan_repo_url": WAN_REPO_URL,
        "wan_repo": wan_repo,
        "safety_guards": {
            "calls_simplepod": False,
            "starts_instance": False,
            "downloads_model_weights": False,
            "runs_inference": False,
            "generates_video": False,
            "clones_code_only_if_needed": True,
            "global_import_probe_downloads_weights": False,
            "global_import_probe_runs_inference": False,
        },
    }

    if not str(wan_repo.get("status", "")).startswith(("found", "cloned")):
        report["status"] = "failed_wan_repo_unavailable"
        report["requirements"] = requirements
        report["dockerfile_implicit_packages"] = dockerfile_packages
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[{TEST_ID}] status={report['status']}")
        print(f"[{TEST_ID}] report={REPORT_PATH}")
        return 0

    scan = scan_imports(Path(str(wan_repo["path"])))
    audit = build_dependency_audit(requirements, dockerfile_packages, scan)
    global_import_probe = (
        {
            "status": "skipped",
            "attempted": False,
            "reason": "--skip-global-import-check",
            "recommended_requirements_additions": [],
            "deferred_optional_additions": [],
        }
        if args.skip_global_import_check
        else run_generate_global_import_probe(
            Path(str(wan_repo["path"])),
            requirements,
            dockerfile_packages,
            max(1, int(args.global_import_max_iterations)),
        )
    )
    combined_recommendations = sorted(
        set(audit["recommended_requirements_additions"])
        | set(global_import_probe.get("recommended_requirements_additions", []))
    )
    combined_deferred = sorted(
        set(audit.get("deferred_optional_additions", []))
        | set(global_import_probe.get("deferred_optional_additions", []))
    )
    if combined_recommendations:
        audit["status"] = "missing_dependencies_found"
    audit["combined_recommended_requirements_additions"] = combined_recommendations
    audit["combined_deferred_optional_additions"] = combined_deferred
    report.update(
        {
            "status": audit["status"],
            "requirements": requirements,
            "dockerfile_implicit_packages": dockerfile_packages,
            "scan": scan,
            "global_import_probe": global_import_probe,
            "audit": audit,
        }
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    missing_focus = [row["package"] for row in audit["missing_focus_dependencies"]]
    recommended = audit["combined_recommended_requirements_additions"]
    print(f"[{TEST_ID}] status={report['status']}")
    print(f"[{TEST_ID}] wan_repo_path={wan_repo['path']}")
    print(f"[{TEST_ID}] missing_focus_dependencies={missing_focus}")
    print(f"[{TEST_ID}] recommended_requirements_additions={recommended}")
    print(f"[{TEST_ID}] deferred_optional_additions={audit['combined_deferred_optional_additions']}")
    print(f"[{TEST_ID}] report={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
