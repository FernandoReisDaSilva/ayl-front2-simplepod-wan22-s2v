import argparse
import ctypes
import gc
import glob
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_ID = "TEMP_FP8_RUNTIME_PROBE_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = REPO_ROOT / "logs" / "fp8_runtime_probe_v1.json"
DEFAULT_CERTIFICATION_PATH = REPO_ROOT / "logs" / "fp8_runtime_certification_v1.json"
TORCHAO_EXTENSION_PATTERNS = ("_C_cutlass_90a*.so", "_C_mxfp8*.so")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def module_version(module: Any) -> str:
    return str(getattr(module, "__version__", "") or "")


def module_file(module: Any) -> str:
    return str(getattr(module, "__file__", "") or "")


def safe_repr(value: Any, limit: int = 1000) -> str:
    try:
        text = repr(value)
    except Exception as exc:
        text = f"<repr_failed:{type(exc).__name__}>"
    return text[:limit]


def runtime_scope() -> dict[str, bool]:
    return {
        "loads_wan": False,
        "loads_wan_model_s2v": False,
        "loads_wan_s2v": False,
        "uses_fastapi": False,
        "uses_simplepod_api": False,
        "uses_r2": False,
        "runs_inference": False,
        "generates_video": False,
        "generates_audio": False,
    }


def torchao_extension_probe(torchao_module: Any) -> dict[str, Any]:
    torchao_file = module_file(torchao_module)
    torchao_dir = Path(torchao_file).resolve().parent if torchao_file else Path("")
    results = []
    for pattern in TORCHAO_EXTENSION_PATTERNS:
        paths = sorted(glob.glob(str(torchao_dir / pattern)))
        if not paths:
            results.append(
                {
                    "pattern": pattern,
                    "status": "missing",
                    "path": "",
                    "error_type": "MissingExtension",
                    "error_truncated": f"No TorchAO extension matched {pattern}",
                }
            )
            continue
        for path in paths:
            try:
                ctypes.CDLL(path)
                status = "ok"
                error_type = ""
                error_truncated = ""
            except Exception as exc:
                status = "failed"
                error_type = type(exc).__name__
                error_truncated = str(exc)[:1000]
            results.append(
                {
                    "pattern": pattern,
                    "status": status,
                    "path": path,
                    "size_bytes": Path(path).stat().st_size if Path(path).exists() else None,
                    "error_type": error_type,
                    "error_truncated": error_truncated,
                }
            )
    failed = [item for item in results if item.get("status") != "ok"]
    return {
        "torchao_file": torchao_file,
        "torchao_dir": str(torchao_dir),
        "extension_patterns": list(TORCHAO_EXTENSION_PATTERNS),
        "extensions": results,
        "failed_extensions": failed,
        "all_required_extensions_loadable": not failed,
    }


def memory_snapshot(torch_module: Any) -> dict[str, Any]:
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


def tensor_facts(tensor: Any) -> dict[str, Any]:
    return {
        "type": type(tensor).__name__,
        "fqcn": f"{type(tensor).__module__}.{type(tensor).__qualname__}",
        "repr": safe_repr(tensor),
        "dtype": str(getattr(tensor, "dtype", "")),
        "device": str(getattr(tensor, "device", "")),
        "shape": list(getattr(tensor, "shape", []) or []),
        "numel": int(tensor.numel()) if hasattr(tensor, "numel") else None,
        "element_size": int(tensor.element_size()) if hasattr(tensor, "element_size") else None,
    }


def module_signature(module: Any) -> list[tuple[str, str]]:
    return [(name, f"{type(child).__module__}.{type(child).__qualname__}") for name, child in module.named_modules()]


def module_inventory(module: Any, torch_module: Any) -> dict[str, Any]:
    modules = list(module.named_modules())
    linear_modules = [(name, child) for name, child in modules if isinstance(child, torch_module.nn.Linear)]
    parameters = list(module.parameters())
    dtype_counts: dict[str, int] = {}
    device_counts: dict[str, int] = {}
    parameter_count = 0
    for parameter in parameters:
        dtype_counts[str(parameter.dtype)] = dtype_counts.get(str(parameter.dtype), 0) + int(parameter.numel())
        device_counts[str(parameter.device)] = device_counts.get(str(parameter.device), 0) + int(parameter.numel())
        parameter_count += int(parameter.numel())
    linear_parameter_count = 0
    for _, child in linear_modules:
        for parameter in child.parameters(recurse=False):
            linear_parameter_count += int(parameter.numel())
    return {
        "module_count": len(modules),
        "nn_linear_count": len(linear_modules),
        "parameter_count": parameter_count,
        "linear_parameter_count": linear_parameter_count,
        "parameter_dtype_counts_by_numel": dtype_counts,
        "parameter_device_counts_by_numel": device_counts,
        "linear_modules": [
            {
                "name": name,
                "type": f"{type(child).__module__}.{type(child).__qualname__}",
                "in_features": int(getattr(child, "in_features", 0) or 0),
                "out_features": int(getattr(child, "out_features", 0) or 0),
                "weight": tensor_facts(getattr(child, "weight", None)),
                "bias": tensor_facts(getattr(child, "bias", None)) if getattr(child, "bias", None) is not None else None,
            }
            for name, child in linear_modules
        ],
    }


def import_torchao_apis() -> dict[str, Any]:
    result: dict[str, Any] = {
        "torchao_import_status": "not_started",
        "torchao_version": "",
        "quantize_import_status": "not_started",
        "float8_config_import_status": "not_started",
        "quantize_available": False,
        "float8_weight_only_config_available": False,
        "error_type": "",
        "error_truncated": "",
        "traceback": [],
    }
    try:
        import torchao

        result["torchao_import_status"] = "ok"
        result["torchao_version"] = module_version(torchao)
        result["torchao_file"] = module_file(torchao)
        result["extension_probe"] = torchao_extension_probe(torchao)
    except Exception as exc:
        result.update(
            {
                "torchao_import_status": "failed",
                "error_type": type(exc).__name__,
                "error_truncated": str(exc)[:1000],
                "traceback": traceback.format_exc().splitlines(),
            }
        )
        return result

    try:
        from torchao.quantization import quantize_

        result["quantize_import_status"] = "ok"
        result["quantize_available"] = callable(quantize_)
        result["quantize_fqcn"] = f"{quantize_.__module__}.{getattr(quantize_, '__qualname__', 'quantize_')}"
        result["_quantize"] = quantize_
    except Exception as exc:
        result.update(
            {
                "quantize_import_status": "failed",
                "error_type": type(exc).__name__,
                "error_truncated": str(exc)[:1000],
                "traceback": traceback.format_exc().splitlines(),
            }
        )
        return result

    try:
        from torchao.quantization import Float8WeightOnlyConfig

        result["float8_config_import_status"] = "ok"
        result["float8_weight_only_config_available"] = True
        result["float8_config_fqcn"] = f"{Float8WeightOnlyConfig.__module__}.{Float8WeightOnlyConfig.__qualname__}"
        result["_float8_config"] = Float8WeightOnlyConfig
    except Exception as exc:
        result.update(
            {
                "float8_config_import_status": "failed",
                "error_type": type(exc).__name__,
                "error_truncated": str(exc)[:1000],
                "traceback": traceback.format_exc().splitlines(),
            }
        )
    return result


def runtime_info(torch_module: Any | None, torchao_info: dict[str, Any] | None = None) -> dict[str, Any]:
    info = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "os_name": os.name,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": "",
        "torch_cuda_version": "",
        "torchao_version": "",
        "cuda_is_available": False,
        "cuda_device_count": 0,
    }
    if torch_module is not None:
        info.update(
            {
                "torch_version": module_version(torch_module),
                "torch_cuda_version": str(getattr(torch_module.version, "cuda", "") or ""),
                "cuda_is_available": bool(torch_module.cuda.is_available()),
                "cuda_device_count": int(torch_module.cuda.device_count()) if hasattr(torch_module.cuda, "device_count") else 0,
            }
        )
    if torchao_info:
        info["torchao_version"] = str(torchao_info.get("torchao_version", "") or "")
    return info


def gpu_info(torch_module: Any) -> dict[str, Any]:
    if torch_module is None or not torch_module.cuda.is_available():
        return {"cuda_available": False}
    props = torch_module.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "device_name": str(props.name),
        "compute_capability": [int(props.major), int(props.minor)],
        "total_memory_gb": round(float(props.total_memory) / (1024**3), 6),
        "multiprocessors": int(props.multi_processor_count),
        "major": int(props.major),
        "minor": int(props.minor),
    }


def cuda_backend_info(torch_module: Any) -> dict[str, Any]:
    if torch_module is None:
        return {}
    info = {
        "arch_list": [],
        "bf16_available": False,
        "fp16_available": False,
        "fp8_e4m3fn_dtype_available": hasattr(torch_module, "float8_e4m3fn"),
        "fp8_e5m2_dtype_available": hasattr(torch_module, "float8_e5m2"),
        "matmul_tf32_allowed": None,
        "cudnn_available": None,
        "cuda_built": str(getattr(torch_module.version, "cuda", "") or ""),
    }
    try:
        info["arch_list"] = list(torch_module.cuda.get_arch_list())
    except Exception as exc:
        info["arch_list_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    try:
        info["bf16_available"] = bool(torch_module.cuda.is_bf16_supported())
    except Exception as exc:
        info["bf16_available_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    try:
        info["fp16_available"] = bool(torch_module.cuda.is_available())
    except Exception as exc:
        info["fp16_available_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    try:
        info["matmul_tf32_allowed"] = bool(torch_module.backends.cuda.matmul.allow_tf32)
    except Exception:
        pass
    try:
        info["cudnn_available"] = bool(torch_module.backends.cudnn.is_available())
    except Exception:
        pass
    return info


def certification_from_report(report: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "torch_import_ok": report.get("torch", {}).get("import_status") == "ok",
        "torchao_import_ok": report.get("torchao", {}).get("import_status") == "ok",
        "cuda_available": bool(report.get("runtime", {}).get("cuda_is_available")),
        "cuda_device_present": int(report.get("runtime", {}).get("cuda_device_count") or 0) >= 1,
        "float8_weight_only_config_available": bool(report.get("apis", {}).get("float8_weight_only_config_available")),
        "quantize_available": bool(report.get("apis", {}).get("quantize_available")),
        "torchao_extensions_loadable": bool(report.get("apis", {}).get("extension_probe", {}).get("all_required_extensions_loadable")),
        "quantization_succeeded": report.get("quantization_result", {}).get("status") == "succeeded",
        "nn_linear_preserved": bool(report.get("module_tree_check", {}).get("nn_linear_still_nn_linear")),
        "module_tree_preserved": bool(report.get("module_tree_check", {}).get("named_modules_preserved")),
        "weight_changed_or_wrapped": bool(report.get("weight_change_check", {}).get("weight_object_changed")),
        "no_wan_imports": not any(report.get("scope", {}).get(key) for key in ("loads_wan", "loads_wan_model_s2v", "loads_wan_s2v")),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "runtime_certification": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed_checks": failed,
        "status": report.get("status"),
        "objective_reason": "all runtime FP8 infrastructure checks passed" if not failed else "failed checks: " + ", ".join(failed),
    }


def run_probe(report_path: Path, certification_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    report: dict[str, Any] = {
        "script_id": SCRIPT_ID,
        "status": "started",
        "created_at": now_iso(),
        "report_path": str(report_path),
        "certification_path": str(certification_path),
        "scope": runtime_scope(),
        "runtime": runtime_info(None),
        "gpu": {},
        "cuda": {},
        "torch": {},
        "torchao": {},
        "apis": {},
        "inventory_before_quantization": {},
        "inventory_after_quantization": {},
        "linear_before_quantization": {},
        "linear_after_quantization": {},
        "module_tree_check": {},
        "weight_change_check": {},
        "hooks_check": {
            "accelerate_hooks_expected": False,
            "accelerate_hooks_present_before": False,
            "accelerate_hooks_present_after": False,
            "reason": "Synthetic nn.Linear probe does not use Accelerate or WanModel_S2V.",
        },
        "memory": {},
        "timings": {},
        "errors": [],
        "runtime_certification": "FAIL",
    }

    torch = None
    linear = None
    try:
        import torch as torch_module

        torch = torch_module
        report["torch"] = {
            "import_status": "ok",
            "version": module_version(torch),
            "cuda_version": str(getattr(torch.version, "cuda", "") or ""),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "float8_e4m3fn_available": hasattr(torch, "float8_e4m3fn"),
            "float8_e5m2_available": hasattr(torch, "float8_e5m2"),
        }
        report["runtime"] = runtime_info(torch)
        report["cuda"] = cuda_backend_info(torch)
    except Exception as exc:
        report["status"] = "failed_import_torch"
        report["errors"].append(
            {
                "phase": "import_torch",
                "error_type": type(exc).__name__,
                "error_truncated": str(exc)[:1000],
                "traceback": traceback.format_exc().splitlines(),
            }
        )
        return finalize_report(report, report_path, certification_path, started, torch)

    apis = import_torchao_apis()
    quantize = apis.pop("_quantize", None)
    float8_config = apis.pop("_float8_config", None)
    report["torchao"] = {
        "import_status": apis.get("torchao_import_status"),
        "version": apis.get("torchao_version", ""),
        "file": apis.get("torchao_file", ""),
        "extension_probe": apis.get("extension_probe", {}),
    }
    report["runtime"] = runtime_info(torch, apis)
    report["apis"] = apis

    if not torch.cuda.is_available():
        report["status"] = "blocked_cuda_unavailable"
        report["gpu"] = gpu_info(torch)
        report["memory"]["initial"] = memory_snapshot(torch)
        return finalize_report(report, report_path, certification_path, started, torch)

    report["gpu"] = gpu_info(torch)
    torch.cuda.reset_peak_memory_stats()
    report["memory"]["before"] = memory_snapshot(torch)

    if not apis.get("quantize_available") or not apis.get("float8_weight_only_config_available"):
        report["status"] = "blocked_torchao_fp8_api_unavailable"
        return finalize_report(report, report_path, certification_path, started, torch)

    try:
        create_started = time.monotonic()
        linear = torch.nn.Linear(4096, 4096, bias=True, device="cuda").to(dtype=torch.bfloat16)
        torch.cuda.synchronize()
        report["timings"]["creation_seconds"] = round(time.monotonic() - create_started, 6)
        report["memory"]["after_creation"] = memory_snapshot(torch)

        before_type = type(linear)
        before_signature = module_signature(linear)
        before_weight = getattr(linear, "weight", None)
        before_bias = getattr(linear, "bias", None)
        report["inventory_before_quantization"] = module_inventory(linear, torch)
        report["linear_before_quantization"] = report["inventory_before_quantization"]["linear_modules"][0]

        quant_started = time.monotonic()
        quantize(linear, float8_config())
        torch.cuda.synchronize()
        report["timings"]["quantization_seconds"] = round(time.monotonic() - quant_started, 6)
        report["memory"]["during_after_quantization"] = memory_snapshot(torch)

        after_type = type(linear)
        after_signature = module_signature(linear)
        after_weight = getattr(linear, "weight", None)
        after_bias = getattr(linear, "bias", None)
        report["inventory_after_quantization"] = module_inventory(linear, torch)
        report["linear_after_quantization"] = report["inventory_after_quantization"]["linear_modules"][0]
        report["module_tree_check"] = {
            "module_disappeared": len(after_signature) < len(before_signature),
            "module_count_before": len(before_signature),
            "module_count_after": len(after_signature),
            "type_preserved": before_type is after_type,
            "module_type_before": f"{before_type.__module__}.{before_type.__qualname__}",
            "module_type_after": f"{after_type.__module__}.{after_type.__qualname__}",
            "named_modules_preserved": before_signature == after_signature,
            "named_modules_before": before_signature,
            "named_modules_after": after_signature,
            "nn_linear_still_nn_linear": isinstance(linear, torch.nn.Linear),
        }
        report["weight_change_check"] = {
            "weight_object_changed": before_weight is not after_weight,
            "bias_object_changed": before_bias is not after_bias,
            "only_weight_changed": before_weight is not after_weight and before_bias is after_bias,
            "weight_before": tensor_facts(before_weight),
            "weight_after": tensor_facts(after_weight),
            "bias_before": tensor_facts(before_bias) if before_bias is not None else None,
            "bias_after": tensor_facts(after_bias) if after_bias is not None else None,
        }
        report["quantization_result"] = {
            "status": "succeeded",
            "weight_type_after": report["weight_change_check"]["weight_after"]["fqcn"],
            "weight_repr_after": report["weight_change_check"]["weight_after"]["repr"],
            "weight_dtype_after": report["weight_change_check"]["weight_after"]["dtype"],
            "weight_device_after": report["weight_change_check"]["weight_after"]["device"],
            "parameter_count_after": report["inventory_after_quantization"]["parameter_count"],
        }
        report["status"] = "succeeded"
    except Exception as exc:
        report["status"] = "failed_quantization_probe"
        report["errors"].append(
            {
                "phase": "quantize_linear",
                "error_type": type(exc).__name__,
                "error_truncated": str(exc)[:1000],
                "traceback": traceback.format_exc().splitlines(),
            }
        )
    finally:
        cleanup_started = time.monotonic()
        try:
            del linear
        except Exception:
            pass
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        report["timings"]["cleanup_seconds"] = round(time.monotonic() - cleanup_started, 6)
        report["memory"]["after_cleanup"] = memory_snapshot(torch)

    return finalize_report(report, report_path, certification_path, started, torch)


def finalize_report(
    report: dict[str, Any],
    report_path: Path,
    certification_path: Path,
    started: float,
    torch_module: Any | None,
) -> dict[str, Any]:
    if torch_module is not None:
        report.setdefault("memory", {})["final"] = memory_snapshot(torch_module)
    report["runtime_seconds"] = round(time.monotonic() - started, 6)
    certification = certification_from_report(report)
    report.update(certification)
    write_json(report_path, report)
    write_json(
        certification_path,
        {
            "script_id": SCRIPT_ID,
            "created_at": now_iso(),
            "runtime_certification": certification["runtime_certification"],
            "objective_reason": certification["objective_reason"],
            "checks": certification["checks"],
            "failed_checks": certification["failed_checks"],
            "status": report.get("status"),
            "runtime": report.get("runtime", {}),
            "gpu": report.get("gpu", {}),
            "cuda": report.get("cuda", {}),
            "torchao": report.get("torchao", {}),
            "apis": report.get("apis", {}),
            "quantization_result": report.get("quantization_result", {}),
            "module_tree_check": report.get("module_tree_check", {}),
            "weight_change_check": report.get("weight_change_check", {}),
            "memory": report.get("memory", {}),
            "timings": report.get("timings", {}),
            "errors": report.get("errors", []),
            "scope": report.get("scope", {}),
        },
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated TorchAO FP8 runtime certification probe for Blackwell. Does not import Wan or run inference."
    )
    parser.add_argument(
        "--report-path",
        default=os.getenv("AYL_FP8_PROBE_REPORT_PATH", str(DEFAULT_REPORT_PATH)),
        help="Path for fp8_runtime_probe_v1.json.",
    )
    parser.add_argument(
        "--certification-path",
        default=os.getenv("AYL_FP8_CERTIFICATION_REPORT_PATH", str(DEFAULT_CERTIFICATION_PATH)),
        help="Path for fp8_runtime_certification_v1.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = Path(args.report_path).expanduser().resolve()
    certification_path = Path(args.certification_path).expanduser().resolve()
    report = run_probe(report_path, certification_path)
    print(
        f"[{SCRIPT_ID}] status={report.get('status')} "
        f"runtime_certification={report.get('runtime_certification')} "
        f"report={report_path} certification={certification_path}",
        flush=True,
    )
    return 0 if report.get("runtime_certification") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
