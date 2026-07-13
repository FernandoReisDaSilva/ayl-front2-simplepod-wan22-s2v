import gc
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .reporting import now_iso
from .settings import Settings
from .wan22_s2v_generate_wrapper import (
    RUNTIME_PATCH_REPORT,
    install_scoped_from_pretrained_patch,
    install_sdpa_attention_fallback_patch,
    parameter_device_summary,
)


WORKER_STATES = {"unloaded", "loading", "ready", "running", "recycling", "failed"}
SAFETENSORS_PATCH_ENV = "AYL_SAFETENSORS_CUDA_TO_CPU_PATCH"


@dataclass(frozen=True)
class Wan22S2VPersistentWorkerConfig:
    model_dir: Path
    wan_repo_dir: Path = Path("/opt/Wan2.2")
    task: str = "s2v-14B"
    device_id: int = 0
    t5_cpu: bool = False
    offload_model: bool = True
    convert_model_dtype: bool = True
    max_concurrent_jobs: int = 1
    warning_growth_gb: float = 1.0
    recycle_growth_gb: float = 2.0
    minimum_free_margin_gb: float = 8.0
    max_jobs_per_worker: int = 30

    @classmethod
    def from_settings(cls, settings: Settings) -> "Wan22S2VPersistentWorkerConfig":
        return cls(
            model_dir=settings.wan22_s2v_model_dir,
            wan_repo_dir=Path(os.getenv("WAN22_REPO_DIR", "/opt/Wan2.2")),
            max_concurrent_jobs=settings.max_concurrent_jobs,
        )


def _truncate(value: Any, limit: int = 1000) -> str:
    return str(value)[:limit]


def _import_torch():
    try:
        import torch

        return torch
    except Exception:
        return None


def _cuda_memory_snapshot() -> dict:
    torch = _import_torch()
    result = {
        "cuda_available": False,
        "memory_allocated_gib": None,
        "memory_reserved_gib": None,
        "max_memory_allocated_gib": None,
        "max_memory_reserved_gib": None,
        "free_margin_gib": None,
        "total_memory_gib": None,
    }
    if torch is None or not torch.cuda.is_available():
        return result
    allocated = float(torch.cuda.memory_allocated()) / (1024**3)
    reserved = float(torch.cuda.memory_reserved()) / (1024**3)
    max_allocated = float(torch.cuda.max_memory_allocated()) / (1024**3)
    max_reserved = float(torch.cuda.max_memory_reserved()) / (1024**3)
    props = torch.cuda.get_device_properties(0)
    total = float(props.total_memory) / (1024**3)
    return {
        "cuda_available": True,
        "memory_allocated_gib": round(allocated, 3),
        "memory_reserved_gib": round(reserved, 3),
        "max_memory_allocated_gib": round(max_allocated, 3),
        "max_memory_reserved_gib": round(max_reserved, 3),
        "free_margin_gib": round(total - reserved, 3),
        "total_memory_gib": round(total, 3),
    }


def _safe_cuda_synchronize() -> None:
    torch = _import_torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _safe_cuda_empty_cache() -> None:
    torch = _import_torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _safe_cuda_ipc_collect() -> None:
    torch = _import_torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.ipc_collect()


def _safe_reset_peak_memory_stats() -> None:
    torch = _import_torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _resolve_wan_config(task: str):
    import wan.configs as configs

    for name in ("WAN_CONFIGS", "TASK_CONFIGS", "CONFIGS"):
        mapping = getattr(configs, name, None)
        if isinstance(mapping, dict) and task in mapping:
            return mapping[task]
    for name in ("s2v_14B", "s2v_14b", "S2V_14B"):
        if hasattr(configs, name):
            return getattr(configs, name)
    raise RuntimeError(f"Could not resolve Wan2.2 config for task={task!r}.")


def _resident_model_object_summary(pipeline: Any) -> dict:
    if pipeline is None:
        return {}
    summary = {
        "pipeline_type": type(pipeline).__name__,
        "pipeline_module": getattr(type(pipeline), "__module__", ""),
        "attributes_present": [],
        "parameter_summaries": {},
    }
    interesting_names = (
        "t5",
        "vae",
        "wav2vec",
        "audio_encoder",
        "noise_model",
        "model",
        "text_encoder",
        "tokenizer",
    )
    for name in interesting_names:
        if not hasattr(pipeline, name):
            continue
        obj = getattr(pipeline, name, None)
        if obj is None:
            continue
        summary["attributes_present"].append(name)
        if hasattr(obj, "parameters"):
            summary["parameter_summaries"][name] = parameter_device_summary(obj, sample_limit=200)
        else:
            summary["parameter_summaries"][name] = {
                "object_type": type(obj).__name__,
                "object_module": getattr(type(obj), "__module__", ""),
                "has_parameters": False,
            }
    return summary


class Wan22S2VPersistentWorker:
    def __init__(self, config: Wan22S2VPersistentWorkerConfig):
        self.config = config
        self.lock = threading.Lock()
        self.worker_state = "unloaded"
        self.load_count = 0
        self.last_load_seconds = None
        self.jobs_completed = 0
        self.current_job_id = None
        self.model_loaded_at = None
        self.last_cleanup_at = None
        self.last_error = None
        self.resident_model_objects = {}
        self.t5_cpu_effective = None
        self.offload_model_effective = None
        self.convert_model_dtype_effective = None
        self.worker_baseline_memory = {}
        self.last_job_memory = {}
        self.last_recycle_decision = {}
        self.safetensors_patch_report = {}
        self.attention_patch_report = {}
        self._pipeline = None
        self._restore_attention_patch = None

    def validate_effective_config(self) -> dict:
        mismatches = {}
        if self.config.t5_cpu is not False:
            mismatches["t5_cpu"] = {"expected": False, "received": self.config.t5_cpu}
        if self.config.offload_model is not True:
            mismatches["offload_model"] = {"expected": True, "received": self.config.offload_model}
        if self.config.convert_model_dtype is not True:
            mismatches["convert_model_dtype"] = {
                "expected": True,
                "received": self.config.convert_model_dtype,
            }
        if self.config.max_concurrent_jobs != 1:
            mismatches["max_concurrent_jobs"] = {
                "expected": 1,
                "received": self.config.max_concurrent_jobs,
            }
        status = "passed" if not mismatches else "failed"
        return {
            "status": status,
            "mismatches": mismatches,
            "effective": {
                "t5_cpu": self.config.t5_cpu,
                "offload_model": self.config.offload_model,
                "convert_model_dtype": self.config.convert_model_dtype,
                "max_concurrent_jobs": self.config.max_concurrent_jobs,
            },
        }

    def status(self) -> dict:
        with self.lock:
            return self._status_unlocked()

    def _status_unlocked(self) -> dict:
        return {
            "worker_state": self.worker_state,
            "load_count": self.load_count,
            "last_load_seconds": self.last_load_seconds,
            "jobs_completed": self.jobs_completed,
            "current_job_id": self.current_job_id,
            "model_loaded_at": self.model_loaded_at,
            "last_cleanup_at": self.last_cleanup_at,
            "last_error": self.last_error,
            "resident_model_objects": self.resident_model_objects,
            "t5_cpu_effective": self.t5_cpu_effective,
            "offload_model_effective": self.offload_model_effective,
            "convert_model_dtype_effective": self.convert_model_dtype_effective,
            "worker_baseline_memory": self.worker_baseline_memory,
            "resident_vram_allocated_gb": self.worker_baseline_memory.get("memory_allocated_gib"),
            "resident_vram_reserved_gb": self.worker_baseline_memory.get("memory_reserved_gib"),
            "last_job_memory": self.last_job_memory,
            "last_recycle_decision": self.last_recycle_decision,
            "safetensors_cuda_to_cpu_patch": self.safetensors_patch_report,
            "attention_sdpa_patch": self.attention_patch_report,
            "config": {
                "model_dir": str(self.config.model_dir),
                "wan_repo_dir": str(self.config.wan_repo_dir),
                "task": self.config.task,
                "device_id": self.config.device_id,
                "t5_cpu": self.config.t5_cpu,
                "offload_model": self.config.offload_model,
                "convert_model_dtype": self.config.convert_model_dtype,
                "max_concurrent_jobs": self.config.max_concurrent_jobs,
                "warning_growth_gb": self.config.warning_growth_gb,
                "recycle_growth_gb": self.config.recycle_growth_gb,
                "minimum_free_margin_gb": self.config.minimum_free_margin_gb,
                "max_jobs_per_worker": self.config.max_jobs_per_worker,
            },
        }

    def load_once(self) -> dict:
        with self.lock:
            if self.worker_state == "ready":
                return {"status": "already_loaded", **self._status_unlocked()}
            if self.worker_state in {"loading", "running", "recycling"}:
                return {"status": "busy", **self._status_unlocked()}
            config_check = self.validate_effective_config()
            if config_check["status"] != "passed":
                self.worker_state = "failed"
                self.last_error = {
                    "type": "InvalidPersistentWorkerConfig",
                    "message": "Persistent worker probe config did not match required fixed values.",
                    "config_check": config_check,
                }
                return {"status": "failed_config_validation", "config_check": config_check, **self._status_unlocked()}

            self.worker_state = "loading"
            started = time.monotonic()
            self.last_error = None
            restore_from_pretrained = None
            old_patch_env = os.getenv(SAFETENSORS_PATCH_ENV)
            try:
                if str(self.config.wan_repo_dir) not in sys.path:
                    sys.path.insert(0, str(self.config.wan_repo_dir))
                if not self.config.wan_repo_dir.exists():
                    raise FileNotFoundError(f"Wan2.2 repo not found: {self.config.wan_repo_dir}")
                if not self.config.model_dir.exists():
                    raise FileNotFoundError(f"Wan2.2 S2V model dir not found: {self.config.model_dir}")

                _safe_cuda_synchronize()
                _safe_reset_peak_memory_stats()
                self._restore_attention_patch = install_sdpa_attention_fallback_patch()
                os.environ[SAFETENSORS_PATCH_ENV] = "1"
                restore_from_pretrained = install_scoped_from_pretrained_patch()

                from wan.speech2video import WanS2V

                config = _resolve_wan_config(self.config.task)
                self._pipeline = WanS2V(
                    config=config,
                    checkpoint_dir=str(self.config.model_dir),
                    device_id=self.config.device_id,
                    t5_cpu=self.config.t5_cpu,
                    init_on_cpu=True,
                    convert_model_dtype=self.config.convert_model_dtype,
                )
                if restore_from_pretrained is not None:
                    restore_from_pretrained()
                    restore_from_pretrained = None

                self.t5_cpu_effective = self.config.t5_cpu
                self.offload_model_effective = self.config.offload_model
                self.convert_model_dtype_effective = self.config.convert_model_dtype
                self.load_count += 1
                self.model_loaded_at = now_iso()
                self.resident_model_objects = _resident_model_object_summary(self._pipeline)
                self.safetensors_patch_report = {
                    **RUNTIME_PATCH_REPORT.get("safetensors_cuda_to_cpu_patch", {}),
                    "patch_scope": "load_once_only",
                    "restored_after_load": True,
                }
                self.attention_patch_report = {
                    **RUNTIME_PATCH_REPORT.get("attention_sdpa_patch", {}),
                    "patch_scope": "worker_lifetime",
                    "restore_on_unload": True,
                }
                _safe_cuda_synchronize()
                self.worker_baseline_memory = _cuda_memory_snapshot()
                self.worker_state = "ready"
                self.last_load_seconds = round(time.monotonic() - started, 3)
                return {
                    "status": "loaded",
                    "load_seconds": self.last_load_seconds,
                    **self._status_unlocked(),
                }
            except Exception as exc:
                if restore_from_pretrained is not None:
                    try:
                        restore_from_pretrained()
                    except Exception:
                        pass
                self.worker_state = "failed"
                self.last_error = {
                    "type": type(exc).__name__,
                    "message": _truncate(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-20:],
                }
                return {
                    "status": "failed",
                    "load_seconds": round(time.monotonic() - started, 3),
                    **self._status_unlocked(),
                }
            finally:
                if old_patch_env is None:
                    os.environ.pop(SAFETENSORS_PATCH_ENV, None)
                else:
                    os.environ[SAFETENSORS_PATCH_ENV] = old_patch_env

    def run_job(self, job: dict) -> dict:
        with self.lock:
            return {
                "status": "not_implemented_probe_only",
                "job_id": job.get("job_id") if isinstance(job, dict) else None,
                "message": "Persistent worker probe currently supports load/status/unload only; it does not run inference.",
                "inference_executed": False,
                "video_generated": False,
                **self._status_unlocked(),
            }

    def cleanup_after_job(self, temporary_objects: list[Any] | None = None, cuda_failure: bool = False) -> dict:
        started = time.monotonic()
        _safe_cuda_synchronize()
        before = _cuda_memory_snapshot()
        if temporary_objects:
            temporary_objects.clear()
        gc.collect()
        _safe_cuda_empty_cache()
        if cuda_failure:
            _safe_cuda_ipc_collect()
        _safe_cuda_synchronize()
        after = _cuda_memory_snapshot()
        baseline_allocated = self.worker_baseline_memory.get("memory_allocated_gib")
        residual_growth = None
        if baseline_allocated is not None and after.get("memory_allocated_gib") is not None:
            residual_growth = round(after["memory_allocated_gib"] - baseline_allocated, 3)
        cleanup = {
            "status": "succeeded",
            "cleanup_seconds": round(time.monotonic() - started, 3),
            "memory_before_cleanup": before,
            "memory_after_cleanup": after,
            "residual_growth_vs_worker_baseline_gib": residual_growth,
            "ipc_collect_used": bool(cuda_failure),
        }
        self.last_cleanup_at = now_iso()
        self.last_job_memory = cleanup
        return cleanup

    def should_recycle(self) -> dict:
        with self.lock:
            return self._should_recycle_unlocked()

    def _should_recycle_unlocked(self) -> dict:
        reasons = []
        memory = self.last_job_memory.get("memory_after_cleanup") or self.worker_baseline_memory
        residual_growth = self.last_job_memory.get("residual_growth_vs_worker_baseline_gib")
        free_margin = memory.get("free_margin_gib") if isinstance(memory, dict) else None
        if residual_growth is not None and residual_growth >= self.config.warning_growth_gb:
            reasons.append("residual_growth_warning")
        if residual_growth is not None and residual_growth >= self.config.recycle_growth_gb:
            reasons.append("residual_growth_recycle")
        if free_margin is not None and free_margin < self.config.minimum_free_margin_gb:
            reasons.append("minimum_free_margin_below_threshold")
        if self.jobs_completed >= self.config.max_jobs_per_worker:
            reasons.append("max_jobs_per_worker_reached")
        if self.last_error:
            error_text = str(self.last_error).lower()
            if "cuda" in error_text and "out of memory" in error_text:
                reasons.append("cuda_oom")
            if "accelerate" in error_text or "hook" in error_text:
                reasons.append("accelerate_or_hook_error")
        if self.worker_state not in WORKER_STATES:
            reasons.append("unknown_worker_state")
        if self.worker_state == "failed":
            reasons.append("worker_failed_state")
        decision = {
            "should_recycle": any(
                reason
                in {
                    "residual_growth_recycle",
                    "minimum_free_margin_below_threshold",
                    "max_jobs_per_worker_reached",
                    "cuda_oom",
                    "accelerate_or_hook_error",
                    "unknown_worker_state",
                    "worker_failed_state",
                }
                for reason in reasons
            ),
            "warning": "residual_growth_warning" in reasons,
            "reasons": reasons,
            "policy": {
                "warning_growth_gb": self.config.warning_growth_gb,
                "recycle_growth_gb": self.config.recycle_growth_gb,
                "minimum_free_margin_gb": self.config.minimum_free_margin_gb,
                "max_jobs_per_worker": self.config.max_jobs_per_worker,
            },
            "memory": memory,
        }
        self.last_recycle_decision = decision
        return decision

    def unload(self) -> dict:
        with self.lock:
            if self.worker_state in {"loading", "running", "recycling"}:
                return {
                    "status": "busy",
                    "worker_state": self.worker_state,
                    "unload_executed": False,
                    **self._status_unlocked(),
                }
            started = time.monotonic()
            self.worker_state = "recycling" if self._pipeline is not None else "unloaded"
            try:
                self._pipeline = None
                self.resident_model_objects = {}
                if self._restore_attention_patch is not None:
                    self._restore_attention_patch()
                    self._restore_attention_patch = None
                gc.collect()
                _safe_cuda_empty_cache()
                _safe_cuda_ipc_collect()
                self.worker_state = "unloaded"
                self.current_job_id = None
                self.worker_baseline_memory = {}
                self.last_load_seconds = None
                self.last_cleanup_at = now_iso()
                return {
                    "status": "unloaded",
                    "unload_seconds": round(time.monotonic() - started, 3),
                    "ipc_collect_used": True,
                    **self._status_unlocked(),
                }
            except Exception as exc:
                self.worker_state = "failed"
                self.last_error = {
                    "type": type(exc).__name__,
                    "message": _truncate(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-20:],
                }
                return {
                    "status": "failed_unload",
                    "unload_seconds": round(time.monotonic() - started, 3),
                    **self._status_unlocked(),
                }
