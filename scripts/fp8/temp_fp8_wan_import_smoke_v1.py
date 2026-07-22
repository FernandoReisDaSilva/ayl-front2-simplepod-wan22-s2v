#!/usr/bin/env python3
"""Import-only smoke test for the FP8 Wan2.2 S2V image.

This script intentionally does not load model weights, run inference, touch R2,
or start any API server. It only proves that the Python dependency closure is
complete enough to import the Wan S2V entrypoints used by Gate 0.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


SCRIPT_ID = "TEMP_FP8_WAN_IMPORT_SMOKE_V1"


REQUIRED_PACKAGE_IMPORTS = [
    "torch",
    "torchvision",
    "torchvision.transforms.functional",
    "torchaudio",
    "torchao",
    "accelerate",
    "diffusers",
    "transformers",
    "safetensors",
    "PIL",
    "cv2",
    "numpy",
    "scipy",
    "decord",
    "librosa",
    "moviepy",
    "omegaconf",
    "peft",
    "einops",
    "easydict",
    "ftfy",
    "imageio",
]


WAN_IMPORTS = [
    "generate",
    "wan.image2video",
    "wan.speech2video",
]


PACKAGE_NAMES = {
    "PIL": "Pillow",
    "cv2": "opencv-python-headless",
    "easydict": "easydict",
    "imageio": "imageio",
    "torchvision.transforms.functional": "torchvision",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(import_name: str) -> str:
    package_name = PACKAGE_NAMES.get(import_name, import_name.replace("_", "-"))
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return ""


def import_module(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {
            "name": name,
            "status": "ok",
            "version": package_version(name),
            "file": str(getattr(module, "__file__", "") or ""),
        }
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic smoke.
        return {
            "name": name,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_truncated": str(exc)[:500],
            "traceback_tail": traceback.format_exc()[-2000:],
        }


def torchvision_c_extension_status() -> dict[str, Any]:
    """Validate torchvision's compiled extension without requiring PyInit__C.

    Recent torchvision wheels load the _C shared library through
    torch.ops.load_library during `import torchvision`. Directly importing
    torchvision._C can fail with PyInit__C even when the compiled ops are loaded
    and usable, so nms remains the ABI gate below.
    """
    result: dict[str, Any] = {
        "direct_import": {
            "attempted": False,
            "result": "NOT_STARTED",
            "error_type": "",
            "error_message": "",
        },
        "direct_import_status": "not_started",
        "load_status": "not_started",
        "file": "",
        "has_ops": False,
    }
    try:
        import torchvision.extension as torchvision_extension
        from torchvision._internally_replaced_utils import _get_extension_path

        result["file"] = str(_get_extension_path("_C"))
        has_ops_fn = getattr(torchvision_extension, "_has_ops", None)
        result["has_ops"] = bool(has_ops_fn()) if callable(has_ops_fn) else bool(
            getattr(torchvision_extension, "_HAS_OPS", False)
        )
        result["load_status"] = "ok" if result["has_ops"] else "failed"
    except Exception as exc:  # noqa: BLE001 - diagnostic smoke.
        result.update(
            {
                "load_status": "failed",
                "error_type": type(exc).__name__,
                "error_truncated": str(exc)[:500],
                "traceback_tail": traceback.format_exc()[-2000:],
            }
        )
        return result

    try:
        result["direct_import"]["attempted"] = True
        module = importlib.import_module("torchvision._C")
        direct_file = str(getattr(module, "__file__", "") or "")
        result["direct_import"].update(
            {
                "result": "IMPORT_OK",
                "file": direct_file,
            }
        )
        result.update(
            {
                "direct_import_status": "ok",
                "direct_import_file": direct_file,
            }
        )
    except ImportError as exc:
        message = str(exc)
        direct_result = "EXPECTED_PYINIT_BEHAVIOR" if "PyInit__C" in message else "IMPORT_ERROR"
        result["direct_import"].update(
            {
                "result": direct_result,
                "error_type": type(exc).__name__,
                "error_message": message,
            }
        )
        result.update(
            {
                "direct_import_status": "expected_pyinit_behavior"
                if direct_result == "EXPECTED_PYINIT_BEHAVIOR"
                else "failed",
                "direct_import_error_type": type(exc).__name__,
                "direct_import_error_truncated": message[:500],
            }
        )
        if "PyInit__C" not in message or not result["has_ops"]:
            result["load_status"] = "failed"
    except Exception as exc:  # noqa: BLE001 - diagnostic smoke.
        result["direct_import"].update(
            {
                "result": "FAILED",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        result.update(
            {
                "direct_import_status": "failed",
                "direct_import_error_type": type(exc).__name__,
                "direct_import_error_truncated": str(exc)[:500],
                "direct_import_traceback_tail": traceback.format_exc()[-2000:],
                "load_status": "failed",
            }
        )

    return result


def torch_stack_abi_summary(torch_stack: dict[str, Any], status: str) -> dict[str, Any]:
    torchvision_c = torch_stack.get("torchvision_c_extension_status", {})
    if not isinstance(torchvision_c, dict):
        torchvision_c = {}
    direct_import = torchvision_c.get("direct_import", {})
    if not isinstance(direct_import, dict):
        direct_import = {}

    native_extension_loaded = bool(torchvision_c.get("has_ops"))
    ops_namespace_available = bool(torch_stack.get("torchvision_ops_namespace_available"))
    nms_import_ok = bool(torch_stack.get("torchvision_nms_imported"))
    nms_execution_ok = bool(torch_stack.get("torchvision_nms_execution_ok"))
    abi_pass = native_extension_loaded and ops_namespace_available and nms_import_ok and nms_execution_ok
    abi_validation = "PASS" if abi_pass else "FAIL"
    if status.startswith("skipped_") and not torch_stack:
        abi_validation = "NOT_RUN"

    return {
        "torchvision_c_direct_import": {
            "attempted": bool(direct_import.get("attempted")),
            "result": str(direct_import.get("result") or "NOT_ATTEMPTED"),
            "error_type": str(direct_import.get("error_type") or ""),
            "error_message": str(direct_import.get("error_message") or ""),
        },
        "torchvision_native_extension_loaded": native_extension_loaded,
        "torchvision_ops_namespace_available": ops_namespace_available,
        "torchvision_nms_import_ok": nms_import_ok,
        "torchvision_nms_execution_ok": nms_execution_ok,
        "abi_validation_method": "torchvision_import_plus_real_nms_execution",
        "abi_validation": abi_validation,
    }


def dependency_validation_summary(report: dict[str, Any]) -> dict[str, Any]:
    torch_stack = report.get("torch_stack", {}) if isinstance(report.get("torch_stack"), dict) else {}
    wan_s2v = report.get("wan_s2v_import", {}) if isinstance(report.get("wan_s2v_import"), dict) else {}
    status = str(report.get("status") or "")
    if status == "ok":
        result = "PASS"
    elif status.startswith("skipped_"):
        result = "SKIPPED"
    else:
        result = "FAIL"
    summary = {
        "torch": {
            "version": torch_stack.get("torch_version", ""),
        },
        "torchvision": {
            "version": torch_stack.get("torchvision_version", ""),
            "ops_repr": torch_stack.get("torchvision_ops_repr", ""),
            "ops_sample": torch_stack.get("torchvision_ops_sample", []),
            "c_extension_file": torch_stack.get("torchvision_c_extension_file", ""),
        },
        "torchaudio": {
            "version": torch_stack.get("torchaudio_version", ""),
        },
        "torchao": {
            "version": torch_stack.get("torchao_version", ""),
        },
        "cuda": {
            "version": torch_stack.get("torch_cuda_version", ""),
            "is_available": torch_stack.get("torch_cuda_is_available"),
        },
        "torchvision_ops_ok": bool(torch_stack.get("torchvision_ops_ok")),
        "wan_import_ok": wan_s2v.get("status") == "ok",
        "result": result,
        "status": status,
        "error_type": report.get("error_type") or torch_stack.get("error_type") or wan_s2v.get("error_type"),
        "error_truncated": report.get("error_truncated")
        or torch_stack.get("error_truncated")
        or wan_s2v.get("error_truncated"),
    }
    summary.update(torch_stack_abi_summary(torch_stack, status))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wan-repo-dir",
        default=os.getenv("WAN22_REPO_DIR", "/opt/Wan2.2"),
        help="Wan2.2 repository path inside the image.",
    )
    parser.add_argument(
        "--allow-missing-repo",
        action="store_true",
        help="Return success when Wan2.2 is not present; useful for local Mac py checks.",
    )
    parser.add_argument(
        "--report-path",
        default=os.getenv("AYL_FP8_WAN_IMPORT_SMOKE_REPORT_PATH", ""),
        help="Optional JSON report output path.",
    )
    return parser.parse_args()


def write_report(path: str, report: dict[str, Any]) -> None:
    report["fp8_dependency_validation"] = dependency_validation_summary(report)
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "script_id": SCRIPT_ID,
        "created_at": utc_now(),
        "wan_repo_dir": args.wan_repo_dir,
        "loads_model_weights": False,
        "runs_inference": False,
        "generates_video": False,
        "imports": [],
        "wan_imports": [],
        "status": "not_started",
    }

    repo = Path(args.wan_repo_dir)
    if not repo.exists():
        report["status"] = "skipped_missing_wan_repo"
        report["error_truncated"] = f"Wan repo not found: {repo}"
        write_report(args.report_path, report)
        print(json.dumps(report, sort_keys=True))
        return 0 if args.allow_missing_repo else 2

    sys.path.insert(0, str(repo))

    for name in REQUIRED_PACKAGE_IMPORTS:
        result = import_module(name)
        report["imports"].append(result)
        if result["status"] != "ok":
            report["status"] = "failed_dependency_import"
            write_report(args.report_path, report)
            print(json.dumps(report, sort_keys=True))
            return 1

    try:
        import torch
        import torchao
        import torchaudio
        import torchvision
        import torchvision.transforms.functional as TF  # noqa: F401
        from torchvision.ops import nms

        torchvision_c = torchvision_c_extension_status()
        if torchvision_c.get("load_status") != "ok":
            raise RuntimeError(f"torchvision C extension load failed: {torchvision_c}")

        torchvision_ops = torch.ops.torchvision

        boxes = torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]],
            dtype=torch.float32,
        )
        scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
        nms_result = nms(boxes, scores, 0.5)
        torchvision_ops_sample = sorted(name for name in dir(torchvision_ops) if not name.startswith("_"))[:50]
        report["torch_stack"] = {
            "status": "ok",
            "torch_version": getattr(torch, "__version__", ""),
            "torch_cuda_version": str(getattr(torch.version, "cuda", "") or ""),
            "torch_cuda_is_available": bool(torch.cuda.is_available()),
            "torchvision_version": getattr(torchvision, "__version__", ""),
            "torchaudio_version": getattr(torchaudio, "__version__", ""),
            "torchao_version": getattr(torchao, "__version__", ""),
            "torchvision_transforms_functional_imported": True,
            "torchvision_nms_imported": True,
            "torchvision_nms_execution_ok": True,
            "torchvision_nms_result": nms_result.detach().cpu().tolist(),
            "torchvision_ops_ok": True,
            "torchvision_ops_namespace_available": True,
            "torchvision_ops_repr": repr(torchvision_ops),
            "torchvision_ops_sample": torchvision_ops_sample,
            "torchvision_c_extension_imported": torchvision_c.get("direct_import_status") == "ok",
            "torchvision_c_extension_status": torchvision_c,
            "torchvision_c_extension_file": str(torchvision_c.get("file") or torchvision_c.get("direct_import_file") or ""),
        }
    except Exception as exc:  # noqa: BLE001 - this is an ABI/operator smoke.
        message = str(exc)
        report["torch_stack"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_truncated": message[:500],
            "traceback_tail": traceback.format_exc()[-3000:],
            "detects_torchvision_nms_missing": "torchvision::nms" in message,
            "detects_undefined_symbol": "undefined symbol" in message.lower(),
            "detects_shared_library_import_error": "ImportError" in type(exc).__name__
            or "cannot open shared object file" in message,
            "detects_cuda_incompatibility": "cuda" in message.lower() and "incompat" in message.lower(),
        }
        report["status"] = "failed_torch_stack_abi_or_operator_check"
        write_report(args.report_path, report)
        print(f"[{SCRIPT_ID}] FAILED status={report['status']} error={type(exc).__name__}: {message[:300]}")
        print(json.dumps(report, sort_keys=True))
        return 1

    for name in WAN_IMPORTS:
        result = import_module(name)
        report["wan_imports"].append(result)
        if result["status"] != "ok":
            report["status"] = "failed_wan_import"
            write_report(args.report_path, report)
            print(f"[{SCRIPT_ID}] FAILED status={report['status']} module={name} error={result.get('error_type')}: {result.get('error_truncated')}")
            print(json.dumps(report, sort_keys=True))
            return 1

    try:
        from wan.speech2video import WanS2V  # noqa: F401

        report["wan_s2v_import"] = {"status": "ok", "symbol": "WanS2V"}
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic smoke.
        report["wan_s2v_import"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_truncated": str(exc)[:500],
            "traceback_tail": traceback.format_exc()[-2000:],
        }
        report["status"] = "failed_wan_s2v_import"
        write_report(args.report_path, report)
        print(f"[{SCRIPT_ID}] FAILED status={report['status']} error={type(exc).__name__}: {str(exc)[:300]}")
        print(json.dumps(report, sort_keys=True))
        return 1

    report["status"] = "ok"
    write_report(args.report_path, report)
    print(f"[{SCRIPT_ID}] PASS torch={report['torch_stack']['torch_version']} torchvision={report['torch_stack']['torchvision_version']} torchaudio={report['torch_stack']['torchaudio_version']} torchao={report['torch_stack']['torchao_version']} cuda={report['torch_stack']['torch_cuda_version']}")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
