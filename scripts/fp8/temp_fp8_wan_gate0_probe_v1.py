import argparse
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
DEFAULT_REPORT_PATH = Path(os.getenv("AYL_FP8_WAN_GATE0_REPORT_PATH", "/tmp/fp8_wan_gate0_probe_v1.json"))
DEFAULT_MODEL_DIR = Path(os.getenv("WAN22_S2V_MODEL_DIR", "/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B"))
DEFAULT_WAN_REPO_DIR = Path(os.getenv("WAN22_REPO_DIR", "/opt/Wan2.2"))
DEFAULT_IMAGE_TAG = os.getenv("AYL_IMAGE_TAG", "0.3.02-blackwell-fp8-wan-gate0-v1")
DEFAULT_WAN_COMMIT = os.getenv("AYL_WAN22_GIT_COMMIT", "42bf4cfaa384bc21833865abc2f9e6c0e67233dc")
TASK = "s2v-14B"
MIN_LINEAR_PARAMS = int(os.getenv("AYL_FP8_GATE0_MIN_LINEAR_PARAMS", "16384"))
DEFAULT_INFER_FRAMES = int(os.getenv("AYL_FP8_GATE0_INFER_FRAMES", "1"))
DEFAULT_MAX_AREA = int(os.getenv("AYL_FP8_GATE0_MAX_AREA", str(256 * 256)))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit_stage(stage: str, **values: Any) -> None:
    suffix = "".join(f" {key}={value}" for key, value in values.items() if value is not None and value != "")
    print(f"[{SCRIPT_ID}] {stage}{suffix}", flush=True)


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
    report.setdefault("errors", []).append(
        {
            "stage": stage,
            "error_type": type(exc).__name__,
            "error_truncated": truncate(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-24:],
        }
    )


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
        if not args.wan_repo_dir.exists():
            raise FileNotFoundError(f"Wan repo not found: {args.wan_repo_dir}")
        if not args.model_dir.exists():
            raise FileNotFoundError(f"Wan model dir not found: {args.model_dir}")

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
            checkpoint_dir=str(args.model_dir),
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
        emit_stage("runtime_certification=FAIL", failure_stage=report["failure_stage"], exception_type=type(exc).__name__)
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
        emit_stage("runtime_certification=" + report["runtime_certification"])
        write_json(args.report_path, report)
        emit_stage("report_written", report=args.report_path)
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

        success_report = tmp / "mock_success.json"
        success = run_mock_subprocess("success", success_report)
        assert success.returncode == 0, success.stdout + success.stderr
        success_json = json.loads(success_report.read_text(encoding="utf-8"))
        assert success_json["runtime_certification"] == "PASS", success_json
        for marker in (
            "bootstrap_started",
            "wan_load_started",
            "wan_load_finished",
            "fp8_quantization_started",
            "fp8_quantization_finished",
            "first_inference_started",
            "first_inference_finished",
        ):
            assert marker in success.stdout, success.stdout
        print("fp8_wan_gate0_mock_success: PASS", flush=True)

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
    report["memory"]["cuda_memory_before"] = {"allocated_gb": 1.0, "reserved_gb": 2.0, "peak_allocated_gb": 1.0}
    emit_stage("cuda_memory_before", allocated=1.0, reserved=2.0)
    emit_stage("wan_load_started")
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
    emit_stage("runtime_certification=" + report["runtime_certification"])
    write_json(args.report_path, report)
    emit_stage("report_written", report=args.report_path)
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
