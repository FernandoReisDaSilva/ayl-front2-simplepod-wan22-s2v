import argparse
import errno
import gc
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_ID = "TEMP_FP8_WAN_GATE0_PROBE_V1"
PROBE_BUILD_ID = "gate0-mount-audit-v1"
REPORT_SCHEMA_VERSION = "fp8-wan-gate0-v3"
DEFAULT_REPORT_PATH = Path(os.getenv("AYL_FP8_WAN_GATE0_REPORT_PATH", "/tmp/fp8_wan_gate0_probe_v1.json"))
DEFAULT_MODEL_DIR = Path(os.getenv("WAN22_S2V_MODEL_DIR", "/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B"))
DEFAULT_WAN_REPO_DIR = Path(os.getenv("WAN22_REPO_DIR", "/opt/Wan2.2"))
DEFAULT_IMAGE_TAG = os.getenv("AYL_IMAGE_TAG", "0.3.07-blackwell-fp8-wan-gate0-exception-capture-v1")
DEFAULT_WAN_COMMIT = os.getenv("AYL_WAN22_GIT_COMMIT", "42bf4cfaa384bc21833865abc2f9e6c0e67233dc")
TASK = "s2v-14B"
MIN_LINEAR_PARAMS = int(os.getenv("AYL_FP8_GATE0_MIN_LINEAR_PARAMS", "16384"))
DEFAULT_INFER_FRAMES = int(os.getenv("AYL_FP8_GATE0_INFER_FRAMES", "1"))
DEFAULT_MAX_AREA = int(os.getenv("AYL_FP8_GATE0_MAX_AREA", str(256 * 256)))
LOADER_ENTRYPOINT = "wan.speech2video.WanS2V"
LOADER_REQUIRED_RELATIVE_PATHS = (
    "generate.py",
    "wan/__init__.py",
    "wan/speech2video.py",
    "wan/configs/__init__.py",
    "wan/modules/s2v/model_s2v.py",
)
MODEL_SEARCH_NAMES = ("Wan2.2-S2V-14B", "wan2.2-s2v-14b", "Wan2.2", "wan2.2")
MODEL_REQUIRED_FILE_MARKERS = (
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
    "wav2vec2-large-xlsr-53-english/model.safetensors",
)
MODEL_REQUIRED_GLOB_MARKERS = (
    "diffusion_pytorch_model-*.safetensors",
)
STANDARD_MODEL_CANDIDATE_DIRS = (
    "/mnt",
    "/storage",
    "/workspace",
    "/runpod-volume",
    "/volume",
    "/models",
    "/data",
    "/root",
    "/opt",
)
STORAGE_MODEL_CANDIDATES = (
    Path("/storage/wan2.2/Wan2.2-S2V-14B"),
    Path("/storage/wan2.2"),
)
LEGACY_MODEL_CANDIDATES = (
    Path("/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B"),
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit_stage(stage: str, **values: Any) -> None:
    suffix = "".join(f" {key}={value}" for key, value in values.items() if value is not None and value != "")
    print(f"[{SCRIPT_ID}] {stage}{suffix}", flush=True)


def emit_structured_field(key: str, value: Any, limit: int = 6000) -> None:
    try:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError:
        rendered = str(value)
    if len(rendered) > limit:
        rendered = rendered[:limit] + "...<truncated>"
    emit_stage(f"{key}={rendered}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def truncate(value: Any, limit: int = 2000) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def path_check(path: Path) -> dict[str, Any]:
    expanded = path.expanduser()
    try:
        resolved = expanded.resolve(strict=False)
    except (OSError, RuntimeError):
        resolved = expanded.absolute()
    result = {
        "checked_path": str(path),
        "resolved_path": str(resolved),
        "exists": False,
        "is_dir": False,
        "is_file": False,
    }
    try:
        result["exists"] = expanded.exists()
        result["is_dir"] = expanded.is_dir()
        result["is_file"] = expanded.is_file()
    except (PermissionError, FileNotFoundError, OSError) as exc:
        result["access_error_type"] = type(exc).__name__
        result["access_error"] = truncate(exc)
        return result
    return result


def safe_path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except (PermissionError, FileNotFoundError, OSError):
        return False


def safe_path_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except (PermissionError, FileNotFoundError, OSError):
        return False


def safe_path_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except (PermissionError, FileNotFoundError, OSError):
        return False


def skipped_path_entry(path: Path, exc: BaseException) -> dict[str, str]:
    return {
        "path": str(path),
        "error_type": type(exc).__name__,
        "error": truncate(exc),
    }


def relevant_env_snapshot() -> dict[str, str]:
    keys = ("WAN_ROOT", "WAN_HOME", "MODEL_DIR", "CHECKPOINT_DIR", "WAN22_S2V_MODEL_DIR", "WAN22_REPO_DIR", "HF_HOME", "HOME", "PWD")
    return {key: os.environ[key] for key in keys if os.environ.get(key)}


def directory_inventory(path: Path, *, max_entries: int = 80, max_depth: int = 2) -> dict[str, Any]:
    root = path.expanduser()
    try:
        resolved_root = str(root.resolve(strict=False))
    except (OSError, RuntimeError):
        resolved_root = str(root.absolute())
    result = {
        "root": str(path),
        "resolved_root": resolved_root,
        "exists": safe_path_exists(root),
        "entries": [],
        "skipped_entries": [],
        "truncated": False,
        "max_entries": max_entries,
        "max_depth": max_depth,
    }
    if not result["exists"] or not safe_path_is_dir(root):
        return result
    count = 0
    queue: list[tuple[Path, int]] = [(root, 0)]
    visited = set()
    while queue and count < max_entries:
        current, depth = queue.pop(0)
        current_text = str(current)
        if current_text in visited:
            continue
        visited.add(current_text)
        try:
            children = sorted(current.iterdir(), key=lambda entry: str(entry))
        except (PermissionError, FileNotFoundError, OSError) as exc:
            result["skipped_entries"].append(skipped_path_entry(current, exc))
            continue
        for item in children:
            if count >= max_entries:
                result["truncated"] = True
                break
            try:
                relative = item.relative_to(root)
            except ValueError:
                continue
            if len(relative.parts) > max_depth:
                continue
            try:
                is_dir = item.is_dir()
                is_file = item.is_file()
                is_symlink = item.is_symlink()
            except (PermissionError, FileNotFoundError, OSError) as exc:
                result["skipped_entries"].append(skipped_path_entry(item, exc))
                continue
            entry = {
                "path": str(relative),
                "is_dir": is_dir,
                "is_file": is_file,
                "is_symlink": is_symlink,
            }
            if is_file:
                try:
                    entry["size_bytes"] = item.stat().st_size
                except (PermissionError, FileNotFoundError, OSError):
                    entry["size_bytes"] = None
            result["entries"].append(entry)
            count += 1
            if is_dir and not is_symlink and len(relative.parts) < max_depth:
                queue.append((item, depth + 1))
    return result


def statvfs_summary(path: Path) -> dict[str, Any]:
    result = {"path": str(path), "exists": safe_path_exists(path)}
    if not result["exists"]:
        return result
    try:
        stats = os.statvfs(path)
    except OSError as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = truncate(exc)
        return result
    block_size = int(stats.f_frsize or stats.f_bsize or 0)
    total_bytes = block_size * int(stats.f_blocks)
    available_bytes = block_size * int(stats.f_bavail)
    free_bytes = block_size * int(stats.f_bfree)
    result.update(
        {
            "total_bytes": total_bytes,
            "available_bytes": available_bytes,
            "free_bytes": free_bytes,
            "total_gb": round(total_bytes / (1024**3), 3) if total_bytes else 0,
            "available_gb": round(available_bytes / (1024**3), 3) if available_bytes else 0,
            "free_gb": round(free_bytes / (1024**3), 3) if free_bytes else 0,
        }
    )
    return result


def parse_proc_mounts(proc_mounts_path: Path = Path("/proc/mounts")) -> list[dict[str, Any]]:
    mounts: list[dict[str, Any]] = []
    try:
        lines = proc_mounts_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return mounts
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        mounts.append(
            {
                "source": parts[0].replace("\\040", " "),
                "mount_point": parts[1].replace("\\040", " "),
                "fs_type": parts[2],
                "options": parts[3] if len(parts) > 3 else "",
            }
        )
    return mounts


def unique_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    result = []
    for path in paths:
        text = str(path)
        if text in seen:
            continue
        seen.add(text)
        result.append(path)
    return result


def collect_mount_inventory(
    expected_model_path: Path,
    *,
    proc_mounts_path: Path = Path("/proc/mounts"),
    candidate_dirs: tuple[str, ...] = STANDARD_MODEL_CANDIDATE_DIRS,
) -> dict[str, Any]:
    proc_mounts = parse_proc_mounts(proc_mounts_path)
    mount_points = unique_paths([Path(item["mount_point"]) for item in proc_mounts] + [Path(item) for item in candidate_dirs])
    shallow_dirs = [Path(item) for item in candidate_dirs]
    existing_mount_points = [path for path in mount_points if safe_path_exists(path)]
    return {
        "proc_mounts_available": safe_path_exists(proc_mounts_path),
        "proc_mounts": proc_mounts[:200],
        "detected_mount_points": [str(path) for path in existing_mount_points[:200]],
        "df": [statvfs_summary(path) for path in existing_mount_points[:80]],
        "shallow_directory_inventory": {
            str(path): directory_inventory(path, max_entries=500, max_depth=4) for path in shallow_dirs
        },
        "expected_model_path": str(expected_model_path),
    }


def candidate_model_roots(mount_inventory: dict[str, Any], expected_model_path: Path) -> list[str]:
    candidates = [Path(item) for item in mount_inventory.get("detected_mount_points", [])]
    candidates.extend(Path(item) for item in STANDARD_MODEL_CANDIDATE_DIRS)
    candidates.extend([expected_model_path.parent, expected_model_path.parent.parent])
    existing = []
    for path in unique_paths(candidates):
        if safe_path_exists(path) and safe_path_is_dir(path):
            existing.append(str(path))
    return existing


def search_model_directories(root_paths: list[str], *, max_depth: int = 4, max_entries: int = 500) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    visited = set()
    scanned = 0
    target_names = set(MODEL_SEARCH_NAMES)
    queue: list[tuple[Path, int]] = [(Path(path), 0) for path in root_paths]
    while queue and scanned < max_entries:
        current, depth = queue.pop(0)
        current_text = str(current)
        if current_text in visited:
            continue
        visited.add(current_text)
        scanned += 1
        try:
            is_dir = current.is_dir()
        except (PermissionError, FileNotFoundError, OSError) as exc:
            skipped.append(skipped_path_entry(current, exc))
            is_dir = False
        if is_dir and current.name in target_names:
            results.append(
                {
                    "path": current_text,
                    "exists": True,
                    "is_dir": True,
                    "depth": depth,
                    "matched_name": current.name,
                }
            )
        if not is_dir or depth >= max_depth:
            continue
        try:
            children = sorted(current.iterdir(), key=lambda item: item.name)
        except (PermissionError, FileNotFoundError, OSError) as exc:
            skipped.append(skipped_path_entry(current, exc))
            continue
        for child in children:
            try:
                if child.is_dir() and not child.is_symlink():
                    queue.append((child, depth + 1))
            except (PermissionError, FileNotFoundError, OSError) as exc:
                skipped.append(skipped_path_entry(child, exc))
                continue
    if queue:
        results.append({"truncated": True, "scanned_entries": scanned, "remaining_queue": len(queue)})
    if skipped:
        results.append({"skipped_entries": skipped[:80], "skipped_entries_count": len(skipped)})
    return results


def model_marker_validation(path: Path) -> dict[str, Any]:
    candidate = path.expanduser()
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        resolved = candidate.absolute()
        resolve_error = {"error_type": type(exc).__name__, "error": truncate(exc)}
    else:
        resolve_error = {}
    validation: dict[str, Any] = {
        "path": str(path),
        "resolved_path": str(resolved),
        "exists": False,
        "is_dir": False,
        "is_symlink": False,
        "accepted": False,
        "reject_reason": "",
        "required_file_markers": list(MODEL_REQUIRED_FILE_MARKERS),
        "required_glob_markers": list(MODEL_REQUIRED_GLOB_MARKERS),
        "markers_found": [],
        "markers_missing": [],
        "glob_markers_found": {},
        "direct_inventory": {},
    }
    if resolve_error:
        validation["resolve_error"] = resolve_error
    try:
        validation["exists"] = candidate.exists()
        validation["is_dir"] = candidate.is_dir()
        validation["is_symlink"] = candidate.is_symlink()
    except (PermissionError, FileNotFoundError, OSError) as exc:
        validation["access_error_type"] = type(exc).__name__
        validation["access_error"] = truncate(exc)
        validation["reject_reason"] = "access_error"
        return validation
    if not validation["exists"]:
        validation["reject_reason"] = "path_missing"
        return validation
    if validation["is_symlink"]:
        validation["reject_reason"] = "candidate_is_symlink"
        return validation
    if not validation["is_dir"]:
        validation["reject_reason"] = "not_directory"
        return validation
    validation["direct_inventory"] = directory_inventory(candidate, max_entries=120, max_depth=2)
    for marker in MODEL_REQUIRED_FILE_MARKERS:
        marker_path = candidate / marker
        if safe_path_is_file(marker_path):
            validation["markers_found"].append(marker)
        else:
            validation["markers_missing"].append(marker)
    for pattern in MODEL_REQUIRED_GLOB_MARKERS:
        try:
            matches = sorted(str(item.name) for item in candidate.glob(pattern) if item.is_file())
        except (PermissionError, FileNotFoundError, OSError) as exc:
            validation.setdefault("glob_errors", {})[pattern] = {
                "error_type": type(exc).__name__,
                "error": truncate(exc),
            }
            matches = []
        validation["glob_markers_found"][pattern] = matches[:20]
        if not matches:
            validation["markers_missing"].append(pattern)
    if validation["markers_missing"]:
        validation["reject_reason"] = "missing_required_model_markers"
        return validation
    validation["accepted"] = True
    validation["reject_reason"] = ""
    return validation


def model_path_source_from_argv(argv: list[str] | None = None) -> str:
    argv = sys.argv[1:] if argv is None else argv
    if any(arg == "--model-dir" or arg.startswith("--model-dir=") for arg in argv):
        return "cli_arg_model_dir"
    if os.getenv("WAN22_S2V_MODEL_DIR"):
        return "env_WAN22_S2V_MODEL_DIR"
    return "probe_default_model_dir"


def resolve_model_directory(
    configured_path: Path,
    *,
    configured_source: str,
    search_results: list[dict[str, Any]] | None = None,
    storage_candidates: tuple[Path, ...] = STORAGE_MODEL_CANDIDATES,
    legacy_candidates: tuple[Path, ...] = LEGACY_MODEL_CANDIDATES,
) -> dict[str, Any]:
    candidate_items: list[dict[str, Any]] = [{"source": configured_source, "path": configured_path}]
    candidate_items.extend({"source": f"storage_candidate_{idx}", "path": path} for idx, path in enumerate(storage_candidates, start=1))
    candidate_items.extend({"source": f"legacy_candidate_{idx}", "path": path} for idx, path in enumerate(legacy_candidates, start=1))
    for item in search_results or []:
        path_text = item.get("path")
        if not path_text:
            continue
        candidate_items.append({"source": f"discovered:{item.get('matched_name', 'unknown')}", "path": Path(path_text)})
    deduped: list[dict[str, Any]] = []
    seen = set()
    for item in candidate_items:
        text = str(item["path"])
        if text in seen:
            continue
        seen.add(text)
        deduped.append(item)
    resolved: dict[str, Any] = {
        "configured_model_path": str(configured_path),
        "resolved_model_path": "",
        "model_path_source": "",
        "model_path_candidates": [],
        "model_path_validation": {},
        "model_path_resolution_status": "failed_no_structurally_valid_model_dir",
    }
    for item in deduped:
        validation = model_marker_validation(Path(item["path"]))
        candidate_record = {
            "source": item["source"],
            "path": str(item["path"]),
            "accepted": validation.get("accepted", False),
            "reject_reason": validation.get("reject_reason", ""),
            "markers_found": validation.get("markers_found", []),
            "markers_missing": validation.get("markers_missing", []),
            "resolved_path": validation.get("resolved_path", ""),
        }
        resolved["model_path_candidates"].append(candidate_record)
        if validation.get("accepted"):
            resolved.update(
                {
                    "resolved_model_path": validation.get("resolved_path") or str(item["path"]),
                    "model_path_source": item["source"],
                    "model_path_validation": validation,
                    "model_path_resolution_status": "resolved",
                }
            )
            break
    if not resolved["model_path_validation"]:
        resolved["model_path_validation"] = {
            "accepted": False,
            "reject_reason": "no_candidate_passed_structural_validation",
        }
    return resolved


def add_mount_and_model_audit(report: dict[str, Any], expected_model_path: Path) -> None:
    mount_inventory = collect_mount_inventory(expected_model_path)
    roots = candidate_model_roots(mount_inventory, expected_model_path)
    search_results = search_model_directories(roots)
    storage_direct_inventory = {
        "/storage": directory_inventory(Path("/storage"), max_entries=120, max_depth=2),
        "/storage/wan2.2": directory_inventory(Path("/storage/wan2.2"), max_entries=160, max_depth=2),
    }
    report["mount_inventory"] = mount_inventory
    report["candidate_model_roots"] = roots
    report["model_search_results"] = search_results
    report["storage_direct_inventory"] = storage_direct_inventory
    report["expected_model_path"] = str(expected_model_path)
    emit_structured_field("detected_mount_points_json", mount_inventory.get("detected_mount_points", []))
    emit_structured_field("candidate_model_roots_json", roots)
    emit_structured_field("model_search_results_json", search_results)
    emit_structured_field("storage_direct_inventory_json", storage_direct_inventory)
    emit_stage(f"expected_model_path={expected_model_path}")


def add_loader_preflight(report: dict[str, Any], args: argparse.Namespace) -> None:
    wan_root = Path(args.wan_repo_dir)
    model_dir = Path(args.model_dir)
    required_paths = [
        ("wan_root", wan_root),
        ("model_dir", model_dir),
        ("checkpoint_dir", model_dir),
        *[(f"wan_loader:{relative}", wan_root / relative) for relative in LOADER_REQUIRED_RELATIVE_PATHS],
    ]
    report["wan_load_preflight"] = {
        "cwd": os.getcwd(),
        "path_cwd": str(Path.cwd()),
        "probe_file": str(Path(__file__).resolve()),
        "loader_entrypoint": LOADER_ENTRYPOINT,
        "environment_variables": relevant_env_snapshot(),
        "path_checks": [{"label": label, **path_check(path)} for label, path in required_paths],
        "tree_inventory": {
            "wan_root": directory_inventory(wan_root, max_entries=80, max_depth=2),
            "model_dir": directory_inventory(model_dir, max_entries=80, max_depth=2),
            "checkpoint_dir": directory_inventory(model_dir, max_entries=80, max_depth=2),
        },
    }


def raise_missing_path(path: Path, message: str) -> None:
    resolved = path.expanduser().resolve(strict=False)
    raise FileNotFoundError(errno.ENOENT, message, str(resolved))


def memory_snapshot(torch_module: Any | None) -> dict[str, Any]:
    result = {
        "cuda_available": False,
        "allocated_gb": None,
        "reserved_gb": None,
        "peak_allocated_gb": None,
        "peak_reserved_gb": None,
        "device_name": "",
        "device_capability": None,
    }
    if torch_module is None or not torch_module.cuda.is_available():
        return result
    props = torch_module.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "allocated_gb": round(float(torch_module.cuda.memory_allocated()) / (1024**3), 6),
        "reserved_gb": round(float(torch_module.cuda.memory_reserved()) / (1024**3), 6),
        "peak_allocated_gb": round(float(torch_module.cuda.max_memory_allocated()) / (1024**3), 6),
        "peak_reserved_gb": round(float(torch_module.cuda.max_memory_reserved()) / (1024**3), 6),
        "device_name": str(props.name),
        "device_capability": list(torch_module.cuda.get_device_capability(0)),
    }


def stage_seconds(started: float) -> float:
    return round(time.monotonic() - started, 6)


def initial_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "script_id": SCRIPT_ID,
        "probe_build_id": PROBE_BUILD_ID,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "probe_script_path": str(Path(__file__).resolve()),
        "probe_started_at": now_iso(),
        "created_at": now_iso(),
        "status": "started",
        "scope": {
            "loads_wan": True,
            "loads_wan_model_s2v": True,
            "loads_wan_s2v": True,
            "uses_fastapi": False,
            "uses_simplepod_api": False,
            "uses_r2": False,
            "downloads_weights": False,
            "runs_minimal_inference": True,
            "generates_long_video": False,
            "benchmarks_quality": False,
        },
        "config": {
            "task": TASK,
            "wan_repo_dir": str(args.wan_repo_dir),
            "model_dir": str(args.model_dir),
            "configured_model_path": str(args.model_dir),
            "configured_model_path_source": model_path_source_from_argv(),
            "resolved_model_path": "",
            "t5_cpu": False,
            "offload_model": True,
            "convert_model_dtype": True,
            "fp8_min_linear_params": MIN_LINEAR_PARAMS,
            "minimal_infer_frames": args.infer_frames,
            "minimal_max_area": args.max_area,
        },
        "environment": {
            "image_tag": DEFAULT_IMAGE_TAG,
            "wan_commit": DEFAULT_WAN_COMMIT,
            "torch_version": "",
            "torchao_version": package_version("torchao"),
            "cuda_version": "",
            "python_version": sys.version,
        },
        "architecture": {
            "wan_s2v_class": "wan.speech2video.WanS2V",
            "noise_model_factory": "wan.modules.s2v.model_s2v.WanModel_S2V.from_pretrained",
            "safetensors_patch": "app.wan22_s2v_generate_wrapper.install_scoped_from_pretrained_patch",
            "attention_patch": "app.wan22_s2v_generate_wrapper.install_sdpa_attention_fallback_patch",
            "quantization_api": "torchao.quantization.quantize_ + Float8WeightOnlyConfig",
        },
        "quantization_plan": quantization_plan(),
        "memory": {},
        "timings": {},
        "wan_load": {},
        "model_path_resolution": {},
        "fp8_quantization": {},
        "first_inference": {},
        "cleanup": {},
        "errors": [],
    }


def quantization_plan() -> dict[str, Any]:
    return {
        "target_component": "WanS2V.noise_model",
        "quantize_module_type": "torch.nn.Linear",
        "config": "Float8WeightOnlyConfig",
        "minimum_parameter_count": MIN_LINEAR_PARAMS,
        "excluded_components": [
            "LayerNorm/RMSNorm and all normalization modules",
            "Embedding modules",
            "T5 text encoder",
            "VAE",
            "wav2vec/audio encoder",
            "tokenizer/text preprocessing objects",
            "non-Linear modules",
            "small Linear modules below the minimum parameter threshold",
        ],
        "rationale": (
            "Gate 0 validates TorchAO FP8 on the transformer/noise model only. "
            "Text/audio/vae components remain BF16/native to reduce unsupported-kernel and quality risk."
        ),
        "known_risks": [
            "TorchAO may reject specific Linear subclasses or wrapped Accelerate modules.",
            "FP8 weight-only may not reduce activation memory.",
            "Applying quantize_ after Accelerate dispatch may alter wrapper internals; module tree is audited before/after.",
            "Minimal inference validates runtime compatibility, not editorial quality.",
        ],
    }


def append_error(report: dict[str, Any], stage: str, exc: BaseException) -> None:
    traceback_text = traceback.format_exc()
    missing_module_name = getattr(exc, "name", "") if isinstance(exc, ModuleNotFoundError) else ""
    missing_module_path = getattr(exc, "path", "") if isinstance(exc, ModuleNotFoundError) else ""
    error = {
        "stage": stage,
        "error_type": type(exc).__name__,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "exception_repr": repr(exc),
        "exception_traceback": traceback_text,
        "missing_module_name": str(missing_module_name or ""),
        "missing_module_path": str(missing_module_path or ""),
        "error_truncated": truncate(exc),
        "traceback_tail": traceback_text.splitlines()[-24:],
    }
    report.update(
        {
            "failure_stage": stage,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "exception_repr": repr(exc),
            "exception_traceback": traceback_text,
            "traceback": traceback_text,
            "missing_module_name": str(missing_module_name or ""),
            "missing_module_path": str(missing_module_path or ""),
        }
    )
    if isinstance(exc, FileNotFoundError):
        filename = getattr(exc, "filename", None)
        resolved_path = str(Path(filename).expanduser().resolve(strict=False)) if filename else ""
        error.update(
            {
                "missing_path": str(filename or ""),
                "resolved_path": resolved_path,
                "cwd": os.getcwd(),
                "probe_file": str(Path(__file__).resolve()),
                "loader_entrypoint": LOADER_ENTRYPOINT,
                "exception_errno": getattr(exc, "errno", None),
                "exception_filename": str(filename or ""),
                "exception_message": str(exc),
                "traceback": traceback_text,
            }
        )
        report.update(
            {
                "missing_path": str(filename or ""),
                "resolved_path": resolved_path,
                "cwd": os.getcwd(),
                "probe_file": str(Path(__file__).resolve()),
                "loader_entrypoint": LOADER_ENTRYPOINT,
                "exception_errno": getattr(exc, "errno", None),
                "exception_filename": str(filename or ""),
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback_text,
                "exception_traceback": traceback_text,
            }
        )
    report.setdefault("errors", []).append(
        error
    )


def emit_failure_diagnostics(report: dict[str, Any]) -> None:
    for key in (
        "failure_stage",
        "exception_type",
        "exception_message",
        "exception_repr",
        "exception_traceback",
        "missing_module_name",
        "missing_module_path",
        "missing_path",
        "resolved_path",
        "exception_filename",
        "exception_errno",
        "cwd",
        "probe_file",
        "loader_entrypoint",
    ):
        if key in report:
            emit_stage(f"{key}={report.get(key)}")
    for key in (
        "exception_message",
        "exception_repr",
        "exception_traceback",
    ):
        if key in report:
            emit_structured_field(f"{key}_json", report.get(key), limit=12000)


def resolve_wan_config(task: str):
    import wan.configs as configs

    mapping = getattr(configs, "WAN_CONFIGS", {})
    if isinstance(mapping, dict) and task in mapping:
        return mapping[task]
    raise RuntimeError(f"Could not resolve Wan config for task={task!r}")


def module_tree_signature(module: Any) -> list[tuple[str, str]]:
    if module is None or not hasattr(module, "named_modules"):
        return []
    return [(name, f"{type(child).__module__}.{type(child).__qualname__}") for name, child in module.named_modules()]


def linear_inventory(module: Any, torch_module: Any) -> list[dict[str, Any]]:
    if module is None or not hasattr(module, "named_modules"):
        return []
    items = []
    for name, child in module.named_modules():
        if not isinstance(child, torch_module.nn.Linear):
            continue
        weight = getattr(child, "weight", None)
        params = int(weight.numel()) if weight is not None and hasattr(weight, "numel") else 0
        items.append(
            {
                "name": name,
                "type": f"{type(child).__module__}.{type(child).__qualname__}",
                "in_features": int(getattr(child, "in_features", 0) or 0),
                "out_features": int(getattr(child, "out_features", 0) or 0),
                "parameter_count": params,
                "weight_type": f"{type(weight).__module__}.{type(weight).__qualname__}" if weight is not None else "",
                "weight_dtype": str(getattr(weight, "dtype", "")),
                "weight_device": str(getattr(weight, "device", "")),
                **fp8_module_decision(name, child, torch_module),
            }
        )
    return items


def fp8_module_decision(name: str, child: Any, torch_module: Any) -> dict[str, Any]:
    if not isinstance(child, torch_module.nn.Linear):
        return {
            "status": "skipped",
            "reason": "non_linear_module",
            "module_class": f"{type(child).__module__}.{type(child).__qualname__}",
        }
    weight = getattr(child, "weight", None)
    parameter_count = int(weight.numel()) if weight is not None and hasattr(weight, "numel") else 0
    lower = name.lower()
    excluded_tokens = (
        "norm",
        "embedding",
        "embed",
        "token",
        "t5",
        "text",
        "vae",
        "wav2vec",
        "audio_encoder",
        "audio",
    )
    matched_exclusion = next((token for token in excluded_tokens if token in lower), "")
    if matched_exclusion:
        return {
            "status": "skipped",
            "reason": f"excluded_name_token:{matched_exclusion}",
            "module_class": f"{type(child).__module__}.{type(child).__qualname__}",
        }
    if parameter_count < MIN_LINEAR_PARAMS:
        return {
            "status": "skipped",
            "reason": "below_min_parameter_count",
            "module_class": f"{type(child).__module__}.{type(child).__qualname__}",
        }
    return {
        "status": "eligible",
        "reason": "large_linear_in_noise_model",
        "module_class": f"{type(child).__module__}.{type(child).__qualname__}",
    }


def module_quantization_inventory(module: Any, torch_module: Any) -> list[dict[str, Any]]:
    if module is None or not hasattr(module, "named_modules"):
        return []
    items = []
    for name, child in module.named_modules():
        if name == "":
            continue
        weight = getattr(child, "weight", None)
        parameter_count = int(weight.numel()) if weight is not None and hasattr(weight, "numel") else 0
        decision = fp8_module_decision(name, child, torch_module)
        items.append(
            {
                "name": name,
                "type": f"{type(child).__module__}.{type(child).__qualname__}",
                "parameter_count": parameter_count,
                "weight_type": f"{type(weight).__module__}.{type(weight).__qualname__}" if weight is not None else "",
                "weight_dtype": str(getattr(weight, "dtype", "")),
                "weight_device": str(getattr(weight, "device", "")),
                "status": decision["status"],
                "reason": decision["reason"],
            }
        )
    return items


def apply_fp8_to_eligible_linears(module: Any, torch_module: Any) -> dict[str, Any]:
    from torchao.quantization import Float8WeightOnlyConfig, quantize_

    result = {
        "status": "started",
        "eligible_modules": [],
        "quantized_modules": [],
        "skipped_modules": [],
        "failed_modules": [],
        "module_decisions": [],
    }
    for name, child in module.named_modules():
        if name == "":
            continue
        weight = getattr(child, "weight", None)
        params = int(weight.numel()) if weight is not None and hasattr(weight, "numel") else 0
        decision = fp8_module_decision(name, child, torch_module)
        decision_record = {
            "name": name,
            "type": f"{type(child).__module__}.{type(child).__qualname__}",
            "parameter_count": params,
            "status": decision["status"],
            "reason": decision["reason"],
        }
        result["module_decisions"].append(decision_record)
        if decision["status"] != "eligible":
            result["skipped_modules"].append(decision_record)
            continue
        result["eligible_modules"].append({"name": name, "parameter_count": params})
        try:
            before_type = type(child)
            before_weight_type = type(weight)
            quantize_(child, Float8WeightOnlyConfig())
            after_weight = getattr(child, "weight", None)
            result["quantized_modules"].append(
                {
                    "name": name,
                    "parameter_count": params,
                    "module_class_preserved": type(child) is before_type,
                    "weight_type_before": f"{before_weight_type.__module__}.{before_weight_type.__qualname__}",
                    "weight_type_after": f"{type(after_weight).__module__}.{type(after_weight).__qualname__}",
                    "weight_dtype_after": str(getattr(after_weight, "dtype", "")),
                    "weight_device_after": str(getattr(after_weight, "device", "")),
                    "status": "quantized",
                }
            )
            decision_record["status"] = "quantized"
            decision_record["reason"] = "Float8WeightOnlyConfig_applied"
        except Exception as exc:
            result["failed_modules"].append(
                {
                    "name": name,
                    "parameter_count": params,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_truncated": truncate(exc),
                }
            )
            decision_record["status"] = "failed"
            decision_record["reason"] = f"{type(exc).__name__}:{truncate(exc, 300)}"
    result["status"] = "succeeded" if result["quantized_modules"] and not result["failed_modules"] else "partial_or_failed"
    return result


def create_minimal_inputs(work_dir: Path) -> tuple[Path, Path]:
    from PIL import Image

    work_dir.mkdir(parents=True, exist_ok=True)
    image_path = work_dir / "fp8_gate0_reference.png"
    audio_path = work_dir / "fp8_gate0_audio.wav"
    Image.new("RGB", (512, 512), color=(120, 120, 120)).save(image_path)
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 16000)
    return image_path, audio_path


def run_gate0(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    report = initial_report(args)
    torch = None
    pipeline = None
    restore_from_pretrained = None
    restore_attention_patch = None
    old_patch_env = os.getenv("AYL_SAFETENSORS_CUDA_TO_CPU_PATCH")
    try:
        emit_stage("bootstrap_started")
        emit_stage(f"probe_build_id={PROBE_BUILD_ID}")
        emit_stage(f"report_schema_version={REPORT_SCHEMA_VERSION}")
        emit_stage(f"probe_script_path={report.get('probe_script_path', '')}")
        if str(args.wan_repo_dir) not in sys.path:
            sys.path.insert(0, str(args.wan_repo_dir))
        if str(Path.cwd()) not in sys.path:
            sys.path.insert(0, str(Path.cwd()))

        emit_stage("torch_import_started")
        import torch as torch_module

        torch = torch_module
        report["environment"].update(
            {
                "torch_version": str(getattr(torch, "__version__", "")),
                "cuda_version": str(getattr(getattr(torch, "version", None), "cuda", "") or ""),
                "torchao_version": package_version("torchao"),
            }
        )
        emit_stage("torch_import_passed", version=getattr(torch, "__version__", ""))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for FP8 Wan Gate 0.")
        torch.cuda.reset_peak_memory_stats()
        report["memory"]["cuda_memory_before"] = memory_snapshot(torch)
        emit_stage("cuda_memory_before", allocated=report["memory"]["cuda_memory_before"].get("allocated_gb"), reserved=report["memory"]["cuda_memory_before"].get("reserved_gb"))

        emit_stage("wan_load_started")
        load_started = time.monotonic()
        add_loader_preflight(report, args)
        add_mount_and_model_audit(report, args.model_dir)
        model_path_resolution = resolve_model_directory(
            args.model_dir,
            configured_source=report["config"].get("configured_model_path_source", model_path_source_from_argv()),
            search_results=report.get("model_search_results", []),
        )
        report["model_path_resolution"] = model_path_resolution
        report["configured_model_path"] = model_path_resolution["configured_model_path"]
        report["resolved_model_path"] = model_path_resolution["resolved_model_path"]
        report["model_path_source"] = model_path_resolution["model_path_source"]
        report["model_path_candidates"] = model_path_resolution["model_path_candidates"]
        report["model_path_validation"] = model_path_resolution["model_path_validation"]
        report["model_path_resolution_status"] = model_path_resolution["model_path_resolution_status"]
        report["config"]["resolved_model_path"] = model_path_resolution["resolved_model_path"]
        resolved_model_dir = Path(model_path_resolution["resolved_model_path"]) if model_path_resolution["resolved_model_path"] else None
        if resolved_model_dir is not None:
            report["wan_load_preflight"]["path_checks"].append(
                {"label": "resolved_model_dir", **path_check(resolved_model_dir)}
            )
            report["wan_load_preflight"]["tree_inventory"]["resolved_model_dir"] = directory_inventory(
                resolved_model_dir, max_entries=120, max_depth=2
            )
        emit_structured_field("environment", report.get("environment", {}))
        emit_structured_field("loader_preflight", report.get("wan_load_preflight", {}))
        emit_structured_field("path_checks", report.get("wan_load_preflight", {}).get("path_checks", []))
        emit_stage(f"configured_model_path={model_path_resolution['configured_model_path']}")
        emit_stage(f"resolved_model_path={model_path_resolution['resolved_model_path']}")
        emit_stage(f"model_path_source={model_path_resolution['model_path_source']}")
        emit_stage(f"model_path_resolution_status={model_path_resolution['model_path_resolution_status']}")
        emit_structured_field("model_path_candidates_json", model_path_resolution["model_path_candidates"])
        emit_structured_field("model_path_validation_json", model_path_resolution["model_path_validation"])
        if not args.wan_repo_dir.exists():
            raise_missing_path(args.wan_repo_dir, "Wan repo not found")
        if resolved_model_dir is None:
            raise_missing_path(args.model_dir, "No structurally valid Wan model dir found")
        for relative_path in LOADER_REQUIRED_RELATIVE_PATHS:
            required_path = args.wan_repo_dir / relative_path
            if not required_path.exists():
                raise_missing_path(required_path, f"Wan loader required path not found: {relative_path}")

        from app.wan22_s2v_generate_wrapper import (
            RUNTIME_PATCH_REPORT,
            install_scoped_from_pretrained_patch,
            install_sdpa_attention_fallback_patch,
        )
        from wan.speech2video import WanS2V

        restore_attention_patch = install_sdpa_attention_fallback_patch()
        os.environ["AYL_SAFETENSORS_CUDA_TO_CPU_PATCH"] = "1"
        restore_from_pretrained = install_scoped_from_pretrained_patch()
        config = resolve_wan_config(TASK)
        pipeline = WanS2V(
            config=config,
            checkpoint_dir=str(resolved_model_dir),
            device_id=0,
            t5_cpu=False,
            init_on_cpu=True,
            convert_model_dtype=True,
        )
        if restore_from_pretrained is not None:
            restore_from_pretrained()
            restore_from_pretrained = None
        torch.cuda.synchronize()
        report["timings"]["load_seconds"] = stage_seconds(load_started)
        report["memory"]["cuda_memory_after_load"] = memory_snapshot(torch)
        report["wan_load"] = {
            "status": "succeeded",
            "pipeline_type": type(pipeline).__name__,
            "pipeline_module": type(pipeline).__module__,
            "objects_present": [name for name in ("t5", "vae", "wav2vec", "audio_encoder", "noise_model") if hasattr(pipeline, name) and getattr(pipeline, name, None) is not None],
            "safetensors_cuda_to_cpu_patch": RUNTIME_PATCH_REPORT.get("safetensors_cuda_to_cpu_patch", {}),
            "attention_sdpa_patch": RUNTIME_PATCH_REPORT.get("attention_sdpa_patch", {}),
        }
        emit_stage("wan_load_finished", load_seconds=report["timings"]["load_seconds"])
        emit_stage("cuda_memory_after_load", allocated=report["memory"]["cuda_memory_after_load"].get("allocated_gb"), reserved=report["memory"]["cuda_memory_after_load"].get("reserved_gb"))

        noise_model = getattr(pipeline, "noise_model", None)
        before_signature = module_tree_signature(noise_model)
        before_inventory = linear_inventory(noise_model, torch)
        before_module_inventory = module_quantization_inventory(noise_model, torch)
        report["fp8_quantization"]["inventory_before"] = before_inventory
        report["fp8_quantization"]["module_inventory_before"] = before_module_inventory

        emit_stage("fp8_quantization_started")
        quant_started = time.monotonic()
        quantization = apply_fp8_to_eligible_linears(noise_model, torch)
        torch.cuda.synchronize()
        after_signature = module_tree_signature(noise_model)
        after_inventory = linear_inventory(noise_model, torch)
        after_module_inventory = module_quantization_inventory(noise_model, torch)
        report["timings"]["quantization_seconds"] = stage_seconds(quant_started)
        report["memory"]["cuda_memory_after_quantization"] = memory_snapshot(torch)
        report["fp8_quantization"].update(
            {
                **quantization,
                "inventory_after": after_inventory,
                "module_inventory_after": after_module_inventory,
                "module_tree_preserved": before_signature == after_signature,
                "linear_count_before": len(before_inventory),
                "linear_count_after": len(after_inventory),
                "module_count_before": len(before_module_inventory),
                "module_count_after": len(after_module_inventory),
            }
        )
        emit_stage("fp8_quantization_finished", quantized=len(quantization["quantized_modules"]), failed=len(quantization["failed_modules"]))
        emit_stage("cuda_memory_after_quantization", allocated=report["memory"]["cuda_memory_after_quantization"].get("allocated_gb"), reserved=report["memory"]["cuda_memory_after_quantization"].get("reserved_gb"))
        if quantization["status"] != "succeeded" or before_signature != after_signature:
            raise RuntimeError(
                "FP8 quantization gate failed: "
                f"status={quantization['status']} module_tree_preserved={before_signature == after_signature}"
            )

        image_path, audio_path = create_minimal_inputs(Path(args.work_dir))
        emit_stage("first_inference_started")
        inference_started = time.monotonic()
        video = pipeline.generate(
            input_prompt="A neutral person faces the camera and speaks calmly.",
            ref_image_path=str(image_path),
            audio_path=str(audio_path),
            enable_tts=False,
            tts_prompt_audio=None,
            tts_prompt_text=None,
            tts_text=None,
            num_repeat=None,
            pose_video=None,
            max_area=int(args.max_area),
            infer_frames=int(args.infer_frames),
            shift=4.0,
            sample_solver="unipc",
            sampling_steps=1,
            guide_scale=1.0,
            seed=42,
            offload_model=True,
            init_first_frame=False,
        )
        torch.cuda.synchronize()
        report["timings"]["first_inference_seconds"] = stage_seconds(inference_started)
        report["first_inference"] = {
            "status": "succeeded",
            "output_type": type(video).__name__,
            "output_shape": list(getattr(video, "shape", []) or []),
            "infer_frames": int(args.infer_frames),
            "max_area": int(args.max_area),
            "video_saved": False,
            "quality_measured": False,
        }
        emit_stage("first_inference_finished", first_inference_seconds=report["timings"]["first_inference_seconds"])
        report["status"] = "succeeded"
    except Exception as exc:
        report["status"] = "failed"
        report["failure_stage"] = current_failure_stage(report)
        append_error(report, report["failure_stage"], exc)
        emit_failure_diagnostics(report)
        emit_stage("runtime_certification_pending=FAIL", failure_stage=report["failure_stage"], exception_type=type(exc).__name__)
    finally:
        cleanup_started = time.monotonic()
        if restore_from_pretrained is not None:
            try:
                restore_from_pretrained()
            except Exception:
                pass
        if restore_attention_patch is not None:
            try:
                restore_attention_patch()
            except Exception:
                pass
        if old_patch_env is None:
            os.environ.pop("AYL_SAFETENSORS_CUDA_TO_CPU_PATCH", None)
        else:
            os.environ["AYL_SAFETENSORS_CUDA_TO_CPU_PATCH"] = old_patch_env
        try:
            del pipeline
        except Exception:
            pass
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        cleanup_memory = memory_snapshot(torch)
        report["memory"]["cuda_memory_after_cleanup"] = cleanup_memory
        report["cleanup"] = {
            "cleanup_seconds": stage_seconds(cleanup_started),
            "cuda_memory_after_cleanup": cleanup_memory,
        }
        emit_stage("cuda_memory_after_cleanup", allocated=cleanup_memory.get("allocated_gb"), reserved=cleanup_memory.get("reserved_gb"), peak=cleanup_memory.get("peak_allocated_gb"))
        report["timings"]["runtime_seconds"] = stage_seconds(started)
        report["runtime_certification"] = "PASS" if report.get("status") == "succeeded" else "FAIL"
        write_json(args.report_path, report)
        emit_stage("report_written", report=args.report_path)
        emit_stage("runtime_certification=" + report["runtime_certification"])
        emit_stage("probe_exit", exit_code=0 if report["runtime_certification"] == "PASS" else 1)
    return report


def current_failure_stage(report: dict[str, Any]) -> str:
    if not report.get("wan_load"):
        return "wan_load"
    if not report.get("fp8_quantization", {}).get("status"):
        return "fp8_quantization"
    if not report.get("first_inference", {}).get("status"):
        return "first_inference"
    return "unknown"


def run_mock_subprocess(stage: str, report_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mock-stage",
            stage,
            "--report-path",
            str(report_path),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def run_mock_tests() -> int:
    with tempfile.TemporaryDirectory(prefix="ayl_fp8_wan_gate0_tests_") as tmpdir:
        tmp = Path(tmpdir)

        def make_valid_model_dir(path: Path) -> Path:
            path.mkdir(parents=True, exist_ok=True)
            for marker in MODEL_REQUIRED_FILE_MARKERS:
                marker_path = path / marker
                marker_path.parent.mkdir(parents=True, exist_ok=True)
                marker_path.write_bytes(b"mock")
            (path / "diffusion_pytorch_model-00001-of-00004.safetensors").write_bytes(b"mock")
            return path

        explicit_valid = make_valid_model_dir(tmp / "explicit" / "Wan2.2-S2V-14B")
        storage_valid = make_valid_model_dir(tmp / "storage" / "wan2.2")
        explicit_resolution = resolve_model_directory(
            explicit_valid,
            configured_source="cli_arg_model_dir",
            storage_candidates=(storage_valid,),
            legacy_candidates=(),
        )
        assert explicit_resolution["model_path_resolution_status"] == "resolved", explicit_resolution
        assert explicit_resolution["model_path_source"] == "cli_arg_model_dir", explicit_resolution
        assert explicit_resolution["resolved_model_path"] == str(explicit_valid.resolve(strict=False)), explicit_resolution
        print("model_resolution_explicit_valid_priority: PASS", flush=True)

        missing_configured = tmp / "missing-configured" / "Wan2.2-S2V-14B"
        fallback_resolution = resolve_model_directory(
            missing_configured,
            configured_source="probe_default_model_dir",
            storage_candidates=(storage_valid,),
            legacy_candidates=(),
        )
        assert fallback_resolution["model_path_resolution_status"] == "resolved", fallback_resolution
        assert fallback_resolution["model_path_source"] == "storage_candidate_1", fallback_resolution
        assert fallback_resolution["resolved_model_path"] == str(storage_valid.resolve(strict=False)), fallback_resolution
        print("model_resolution_missing_configured_falls_back_to_storage_wan2_2: PASS", flush=True)

        invalid_storage = tmp / "invalid-storage" / "wan2.2"
        invalid_storage.mkdir(parents=True)
        invalid_resolution = resolve_model_directory(
            missing_configured,
            configured_source="probe_default_model_dir",
            storage_candidates=(invalid_storage,),
            legacy_candidates=(),
        )
        assert invalid_resolution["model_path_resolution_status"].startswith("failed"), invalid_resolution
        assert any(item.get("reject_reason") == "missing_required_model_markers" for item in invalid_resolution["model_path_candidates"]), invalid_resolution
        print("model_resolution_storage_without_markers_rejected: PASS", flush=True)

        storage_parent = tmp / "storage-parent" / "wan2.2"
        storage_child = make_valid_model_dir(storage_parent / "Wan2.2-S2V-14B")
        storage_child_resolution = resolve_model_directory(
            missing_configured,
            configured_source="probe_default_model_dir",
            storage_candidates=(storage_child, storage_parent),
            legacy_candidates=(),
        )
        assert storage_child_resolution["model_path_source"] == "storage_candidate_1", storage_child_resolution
        assert storage_child_resolution["resolved_model_path"] == str(storage_child.resolve(strict=False)), storage_child_resolution
        print("model_resolution_valid_storage_child_accepted: PASS", flush=True)

        legacy_valid = make_valid_model_dir(tmp / "mnt" / "ayl_models" / "wan2.2" / "Wan2.2-S2V-14B")
        legacy_resolution = resolve_model_directory(
            missing_configured,
            configured_source="probe_default_model_dir",
            storage_candidates=(tmp / "storage-missing" / "wan2.2",),
            legacy_candidates=(legacy_valid,),
        )
        assert legacy_resolution["model_path_source"] == "legacy_candidate_1", legacy_resolution
        assert legacy_resolution["resolved_model_path"] == str(legacy_valid.resolve(strict=False)), legacy_resolution
        print("model_resolution_legacy_mnt_ayl_models_still_supported: PASS", flush=True)

        if hasattr(os, "symlink"):
            symlink_target = make_valid_model_dir(tmp / "symlink-target" / "Wan2.2-S2V-14B")
            symlink_path = tmp / "symlink-candidate"
            os.symlink(symlink_target, symlink_path)
            symlink_validation = model_marker_validation(symlink_path)
            assert symlink_validation["reject_reason"] == "candidate_is_symlink", symlink_validation
            loop_inventory = directory_inventory(tmp, max_entries=300, max_depth=3)
            assert not loop_inventory.get("truncated"), loop_inventory
            print("model_resolution_symlink_candidate_rejected_without_loop: PASS", flush=True)

        permission_candidate = tmp / "permission-candidate"
        original_exists = Path.exists

        def mocked_exists(self):
            if self == permission_candidate:
                raise PermissionError(13, "mock permission denied", str(self))
            return original_exists(self)

        Path.exists = mocked_exists
        try:
            permission_resolution = resolve_model_directory(
                permission_candidate,
                configured_source="probe_default_model_dir",
                storage_candidates=(),
                legacy_candidates=(),
            )
            assert permission_resolution["model_path_resolution_status"] == "failed_no_structurally_valid_model_dir", permission_resolution
            assert permission_resolution["model_path_candidates"][0]["reject_reason"] == "access_error", permission_resolution
        finally:
            Path.exists = original_exists
        print("model_resolution_permission_error_tolerated: PASS", flush=True)

        no_valid_resolution = resolve_model_directory(
            missing_configured,
            configured_source="probe_default_model_dir",
            storage_candidates=(tmp / "no-storage" / "wan2.2",),
            legacy_candidates=(tmp / "no-legacy" / "Wan2.2-S2V-14B",),
        )
        assert no_valid_resolution["model_path_resolution_status"] == "failed_no_structurally_valid_model_dir", no_valid_resolution
        assert not no_valid_resolution["resolved_model_path"], no_valid_resolution
        print("model_resolution_no_valid_candidate_structured_failure: PASS", flush=True)

        expected_mount = tmp / "mnt" / "ayl_models"
        expected_model = expected_mount / "wan2.2" / "Wan2.2-S2V-14B"
        expected_model.mkdir(parents=True, exist_ok=True)
        expected_proc = tmp / "proc_mounts_expected"
        expected_proc.write_text(f"mockdev {expected_mount} ext4 rw 0 0\n", encoding="utf-8")
        expected_inventory = collect_mount_inventory(
            expected_model,
            proc_mounts_path=expected_proc,
            candidate_dirs=(str(expected_mount),),
        )
        expected_roots = candidate_model_roots(expected_inventory, expected_model)
        expected_results = search_model_directories(expected_roots)
        assert any(item.get("path") == str(expected_model) for item in expected_results), expected_results
        print("mount_expected_path_present: PASS", flush=True)

        alternate_mount = tmp / "runpod-volume"
        alternate_model = alternate_mount / "wan2.2" / "Wan2.2-S2V-14B"
        alternate_model.mkdir(parents=True)
        alternate_proc = tmp / "proc_mounts_alternate"
        alternate_proc.write_text(f"mockdev {alternate_mount} ext4 rw 0 0\n", encoding="utf-8")
        alternate_inventory = collect_mount_inventory(
            alternate_model,
            proc_mounts_path=alternate_proc,
            candidate_dirs=(str(alternate_mount),),
        )
        alternate_results = search_model_directories(candidate_model_roots(alternate_inventory, alternate_model))
        assert any(item.get("path") == str(alternate_model) for item in alternate_results), alternate_results
        print("mount_alternate_path_present: PASS", flush=True)

        missing_mount = tmp / "missing-volume"
        missing_mount.mkdir()
        missing_model = missing_mount / "wan2.2" / "Wan2.2-S2V-14B"
        missing_inventory = collect_mount_inventory(missing_model, candidate_dirs=(str(missing_mount),))
        missing_results = search_model_directories(candidate_model_roots(missing_inventory, missing_model))
        assert not any(item.get("path") == str(missing_model) for item in missing_results), missing_results
        print("mount_present_model_missing: PASS", flush=True)

        accessible_mount = tmp / "accessible-volume"
        inaccessible_mount = tmp / "inaccessible-volume"
        accessible_child = accessible_mount / "visible"
        accessible_child.mkdir(parents=True)
        inaccessible_mount.mkdir()
        original_iterdir = Path.iterdir

        def mocked_iterdir(self):
            if self == inaccessible_mount:
                raise PermissionError(13, "mock permission denied", str(self))
            return original_iterdir(self)

        Path.iterdir = mocked_iterdir
        try:
            permission_inventory = collect_mount_inventory(
                accessible_mount / "wan2.2" / "Wan2.2-S2V-14B",
                candidate_dirs=(str(accessible_mount), str(inaccessible_mount)),
            )
            accessible_listing = permission_inventory["shallow_directory_inventory"][str(accessible_mount)]
            inaccessible_listing = permission_inventory["shallow_directory_inventory"][str(inaccessible_mount)]
            assert accessible_listing["entries"], permission_inventory
            assert inaccessible_listing["skipped_entries"][0]["error_type"] == "PermissionError", permission_inventory
            search_results = search_model_directories([str(accessible_mount), str(inaccessible_mount)])
            assert any(item.get("path") == "visible" or item.get("path", "").endswith("/visible") for item in accessible_listing["entries"]), permission_inventory
            assert any(item.get("skipped_entries_count") for item in search_results), search_results
        finally:
            Path.iterdir = original_iterdir
        print("mount_permission_error_tolerated: PASS", flush=True)

        multi_a = tmp / "multi-a"
        multi_b = tmp / "multi-b"
        multi_model = multi_b / "Wan2.2" / "Wan2.2-S2V-14B"
        multi_a.mkdir()
        multi_model.mkdir(parents=True)
        multi_results = search_model_directories([str(multi_a), str(multi_b)])
        assert any(item.get("path") == str(multi_model) for item in multi_results), multi_results
        assert not any(item.get("path") == str(multi_a / "Wan2.2-S2V-14B") for item in multi_results), multi_results
        print("mount_multiple_candidates: PASS", flush=True)

        missing_root_report = tmp / "mock_missing_root.json"
        missing_root = run_mock_subprocess("wan_missing_root", missing_root_report)
        assert missing_root.returncode != 0, missing_root.stdout + missing_root.stderr
        missing_root_json = json.loads(missing_root_report.read_text(encoding="utf-8"))
        assert missing_root_json["failure_stage"] == "wan_load", missing_root_json
        assert missing_root_json["missing_path"], missing_root_json
        assert missing_root_json["resolved_path"], missing_root_json
        assert missing_root_json["exception_type"] == "FileNotFoundError", missing_root_json
        assert missing_root_json["exception_message"], missing_root_json
        print("wan_load_missing_root: PASS", flush=True)

        missing_checkpoint_report = tmp / "mock_missing_checkpoint.json"
        missing_checkpoint = run_mock_subprocess("wan_missing_checkpoint", missing_checkpoint_report)
        assert missing_checkpoint.returncode != 0, missing_checkpoint.stdout + missing_checkpoint.stderr
        missing_checkpoint_json = json.loads(missing_checkpoint_report.read_text(encoding="utf-8"))
        assert missing_checkpoint_json["failure_stage"] == "wan_load", missing_checkpoint_json
        assert missing_checkpoint_json["missing_path"], missing_checkpoint_json
        assert missing_checkpoint_json["resolved_path"], missing_checkpoint_json
        assert missing_checkpoint_json["exception_type"] == "FileNotFoundError", missing_checkpoint_json
        assert missing_checkpoint_json["exception_message"], missing_checkpoint_json
        print("wan_load_missing_checkpoint: PASS", flush=True)

        missing_config_report = tmp / "mock_missing_config.json"
        missing_config = run_mock_subprocess("wan_missing_config", missing_config_report)
        assert missing_config.returncode != 0, missing_config.stdout + missing_config.stderr
        missing_config_json = json.loads(missing_config_report.read_text(encoding="utf-8"))
        assert missing_config_json["failure_stage"] == "wan_load", missing_config_json
        assert missing_config_json["missing_path"], missing_config_json
        assert missing_config_json["resolved_path"], missing_config_json
        assert missing_config_json["exception_type"] == "FileNotFoundError", missing_config_json
        assert missing_config_json["exception_message"], missing_config_json
        print("wan_load_missing_config: PASS", flush=True)

        module_missing_report = tmp / "mock_module_not_found.json"
        module_missing = run_mock_subprocess("wan_module_not_found", module_missing_report)
        assert module_missing.returncode != 0, module_missing.stdout + module_missing.stderr
        module_missing_json = json.loads(module_missing_report.read_text(encoding="utf-8"))
        assert module_missing_json["failure_stage"] == "wan_load", module_missing_json
        assert module_missing_json["exception_type"] == "ModuleNotFoundError", module_missing_json
        assert "mock_missing_dependency" in module_missing_json["exception_message"], module_missing_json
        assert "ModuleNotFoundError" in module_missing_json["exception_traceback"], module_missing_json
        assert module_missing_json["missing_module_name"] == "mock_missing_dependency", module_missing_json
        assert "exception_message_json=" in module_missing.stdout, module_missing.stdout
        assert "exception_traceback_json=" in module_missing.stdout, module_missing.stdout
        assert module_missing.stdout.rfind("report_written") < module_missing.stdout.rfind("runtime_certification=FAIL"), module_missing.stdout
        print("wan_load_module_not_found_preserved: PASS", flush=True)

        success_report = tmp / "mock_success.json"
        success = run_mock_subprocess("success", success_report)
        assert success.returncode == 0, success.stdout + success.stderr
        success_json = json.loads(success_report.read_text(encoding="utf-8"))
        assert success_json["runtime_certification"] == "PASS", success_json
        assert success_json["wan_load"]["status"] == "succeeded", success_json
        assert success_json["probe_build_id"] == PROBE_BUILD_ID, success_json
        assert success_json["report_schema_version"] == REPORT_SCHEMA_VERSION, success_json
        assert success_json["probe_script_path"], success_json
        assert success_json["probe_started_at"], success_json
        for marker in (
            "bootstrap_started",
            f"probe_build_id={PROBE_BUILD_ID}",
            f"report_schema_version={REPORT_SCHEMA_VERSION}",
            "wan_load_started",
            "wan_load_finished",
            "fp8_quantization_started",
            "fp8_quantization_finished",
            "first_inference_started",
            "first_inference_finished",
        ):
            assert marker in success.stdout, success.stdout
        print("wan_load_mock_success: PASS", flush=True)

        quant_report = tmp / "mock_quant_fail.json"
        quant = run_mock_subprocess("quantization_failure", quant_report)
        assert quant.returncode != 0, quant.stdout + quant.stderr
        quant_json = json.loads(quant_report.read_text(encoding="utf-8"))
        assert quant_json["runtime_certification"] == "FAIL", quant_json
        assert quant_json["failure_stage"] == "fp8_quantization", quant_json
        print("fp8_wan_gate0_mock_quantization_failure: PASS", flush=True)

        inference_report = tmp / "mock_inference_fail.json"
        inference = run_mock_subprocess("inference_failure", inference_report)
        assert inference.returncode != 0, inference.stdout + inference.stderr
        inference_json = json.loads(inference_report.read_text(encoding="utf-8"))
        assert inference_json["failure_stage"] == "first_inference", inference_json
        print("fp8_wan_gate0_mock_inference_failure: PASS", flush=True)

    return 0


def run_mock_gate0(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    report = initial_report(args)
    emit_stage("bootstrap_started")
    emit_stage(f"probe_build_id={PROBE_BUILD_ID}")
    emit_stage(f"report_schema_version={REPORT_SCHEMA_VERSION}")
    report["memory"]["cuda_memory_before"] = {"allocated_gb": 1.0, "reserved_gb": 2.0, "peak_allocated_gb": 1.0}
    emit_stage("cuda_memory_before", allocated=1.0, reserved=2.0)
    emit_stage("wan_load_started")
    report["wan_load_preflight"] = {
        "cwd": os.getcwd(),
        "path_cwd": str(Path.cwd()),
        "probe_file": str(Path(__file__).resolve()),
        "loader_entrypoint": LOADER_ENTRYPOINT,
        "environment_variables": relevant_env_snapshot(),
        "path_checks": [],
        "tree_inventory": {},
    }
    missing_path_by_stage = {
        "wan_missing_root": Path("/mock/missing/Wan2.2"),
        "wan_missing_checkpoint": Path("/mock/missing/Wan2.2-S2V-14B"),
        "wan_missing_config": Path("/mock/Wan2.2/wan/configs/__init__.py"),
    }
    if args.mock_stage == "wan_module_not_found":
        try:
            raise ModuleNotFoundError("No module named 'mock_missing_dependency'", name="mock_missing_dependency")
        except ModuleNotFoundError as exc:
            report["status"] = "failed"
            report["failure_stage"] = "wan_load"
            append_error(report, "wan_load", exc)
            emit_failure_diagnostics(report)
        report["memory"]["cuda_memory_after_cleanup"] = {"allocated_gb": 0.0, "reserved_gb": 0.0, "peak_allocated_gb": 1.0}
        report["cleanup"] = {"cleanup_seconds": 0.1, "cuda_memory_after_cleanup": report["memory"]["cuda_memory_after_cleanup"]}
        emit_stage("cuda_memory_after_cleanup", allocated=0.0, reserved=0.0, peak=1.0)
        report["timings"]["runtime_seconds"] = stage_seconds(started)
        report["runtime_certification"] = "FAIL"
        write_json(args.report_path, report)
        emit_stage("report_written", report=args.report_path)
        emit_stage("runtime_certification=FAIL", failure_stage="wan_load", exception_type="ModuleNotFoundError")
        emit_stage("probe_exit", exit_code=1)
        return report
    if args.mock_stage in missing_path_by_stage:
        try:
            raise_missing_path(missing_path_by_stage[args.mock_stage], f"Mock missing path for {args.mock_stage}")
        except FileNotFoundError as exc:
            report["status"] = "failed"
            report["failure_stage"] = "wan_load"
            append_error(report, "wan_load", exc)
            emit_failure_diagnostics(report)
        report["memory"]["cuda_memory_after_cleanup"] = {"allocated_gb": 0.0, "reserved_gb": 0.0, "peak_allocated_gb": 1.0}
        report["cleanup"] = {"cleanup_seconds": 0.1, "cuda_memory_after_cleanup": report["memory"]["cuda_memory_after_cleanup"]}
        emit_stage("cuda_memory_after_cleanup", allocated=0.0, reserved=0.0, peak=1.0)
        report["timings"]["runtime_seconds"] = stage_seconds(started)
        report["runtime_certification"] = "FAIL"
        write_json(args.report_path, report)
        emit_stage("report_written", report=args.report_path)
        emit_stage("runtime_certification=FAIL", failure_stage="wan_load", exception_type="FileNotFoundError")
        emit_stage("probe_exit", exit_code=1)
        return report

    report["wan_load"] = {"status": "succeeded", "objects_present": ["t5", "vae", "wav2vec", "noise_model"]}
    report["timings"]["load_seconds"] = 0.1
    report["memory"]["cuda_memory_after_load"] = {"allocated_gb": 40.0, "reserved_gb": 42.0, "peak_allocated_gb": 40.0}
    emit_stage("wan_load_finished", load_seconds=0.1)
    emit_stage("cuda_memory_after_load", allocated=40.0, reserved=42.0)
    emit_stage("fp8_quantization_started")
    if args.mock_stage == "quantization_failure":
        report["fp8_quantization"] = {"status": "failed", "failed_modules": [{"name": "mock.linear"}]}
        report["failure_stage"] = "fp8_quantization"
        report["status"] = "failed"
    else:
        report["fp8_quantization"] = {
            "status": "succeeded",
            "quantized_modules": [{"name": "mock.linear", "module_class_preserved": True}],
            "failed_modules": [],
            "module_tree_preserved": True,
        }
        emit_stage("fp8_quantization_finished", quantized=1, failed=0)
        report["timings"]["quantization_seconds"] = 0.1
        report["memory"]["cuda_memory_after_quantization"] = {"allocated_gb": 32.0, "reserved_gb": 42.0, "peak_allocated_gb": 40.0}
        emit_stage("cuda_memory_after_quantization", allocated=32.0, reserved=42.0)
        emit_stage("first_inference_started")
        if args.mock_stage == "inference_failure":
            report["first_inference"] = {"status": "failed"}
            report["failure_stage"] = "first_inference"
            report["status"] = "failed"
        else:
            report["first_inference"] = {"status": "succeeded", "output_type": "Tensor", "output_shape": [1, 3, 1, 16, 16]}
            report["timings"]["first_inference_seconds"] = 0.1
            report["status"] = "succeeded"
            emit_stage("first_inference_finished", first_inference_seconds=0.1)
    report["memory"]["cuda_memory_after_cleanup"] = {"allocated_gb": 0.0, "reserved_gb": 0.0, "peak_allocated_gb": 40.0}
    report["cleanup"] = {"cleanup_seconds": 0.1, "cuda_memory_after_cleanup": report["memory"]["cuda_memory_after_cleanup"]}
    emit_stage("cuda_memory_after_cleanup", allocated=0.0, reserved=0.0, peak=40.0)
    report["timings"]["runtime_seconds"] = stage_seconds(started)
    report["runtime_certification"] = "PASS" if report["status"] == "succeeded" else "FAIL"
    write_json(args.report_path, report)
    emit_stage("report_written", report=args.report_path)
    emit_stage("runtime_certification=" + report["runtime_certification"])
    emit_stage("probe_exit", exit_code=0 if report["runtime_certification"] == "PASS" else 1)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimental isolated FP8 Wan Gate 0 probe. No R2, no SimplePod API, no benchmark.")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--wan-repo-dir", type=Path, default=DEFAULT_WAN_REPO_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/fp8_wan_gate0_probe_v1"))
    parser.add_argument("--infer-frames", type=int, default=DEFAULT_INFER_FRAMES)
    parser.add_argument("--max-area", type=int, default=DEFAULT_MAX_AREA)
    parser.add_argument("--mock-stage", default="", help=argparse.SUPPRESS)
    parser.add_argument("--run-mock-tests", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_mock_tests:
        return run_mock_tests()
    if args.mock_stage:
        report = run_mock_gate0(args)
    else:
        report = run_gate0(args)
    return 0 if report.get("runtime_certification") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
