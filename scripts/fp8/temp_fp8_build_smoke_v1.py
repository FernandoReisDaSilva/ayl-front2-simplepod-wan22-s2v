import ctypes
import glob
import json
import os
from pathlib import Path


SCRIPT_ID = "TEMP_FP8_BUILD_SMOKE_V1"
EXTENSION_PATTERNS = ("_C_cutlass_90a*.so", "_C_mxfp8*.so")


def extension_results(torchao_file: str) -> list[dict]:
    torchao_dir = Path(torchao_file).resolve().parent
    results = []
    for pattern in EXTENSION_PATTERNS:
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
    return results


def main() -> int:
    import torch
    import torchao
    from torchao.quantization import Float8WeightOnlyConfig, quantize_

    linear = torch.nn.Linear(4096, 4096, bias=True).to(dtype=torch.bfloat16)
    quantization_status = "not_attempted"
    quantization_error = ""
    try:
        quantize_(linear, Float8WeightOnlyConfig())
        quantization_status = "succeeded"
    except Exception as exc:
        quantization_status = "failed"
        quantization_error = f"{type(exc).__name__}: {str(exc)[:1000]}"

    extensions = extension_results(getattr(torchao, "__file__", ""))
    failed_extensions = [item for item in extensions if item.get("status") != "ok"]
    result = {
        "script_id": SCRIPT_ID,
        "torch_version": getattr(torch, "__version__", ""),
        "torch_cuda_version": getattr(torch.version, "cuda", ""),
        "torch_file": getattr(torch, "__file__", ""),
        "torchao_version": getattr(torchao, "__version__", ""),
        "torchao_file": getattr(torchao, "__file__", ""),
        "expected_torch_version": os.getenv("AYL_TORCH_VERSION", ""),
        "expected_torchao_version": os.getenv("AYL_TORCHAO_VERSION", ""),
        "quantize_available": callable(quantize_),
        "float8_weight_only_config": str(Float8WeightOnlyConfig),
        "cpu_linear_created": isinstance(linear, torch.nn.Linear),
        "cpu_linear_dtype": str(linear.weight.dtype),
        "cpu_quantization_status": quantization_status,
        "cpu_quantization_error": quantization_error,
        "extensions": extensions,
        "failed_extensions": failed_extensions,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if failed_extensions:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
