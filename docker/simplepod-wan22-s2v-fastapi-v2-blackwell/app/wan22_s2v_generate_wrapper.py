import os
import sys
import json
from contextlib import contextmanager
from pathlib import Path


WAN22_REPO_DIR = Path("/opt/Wan2.2")
DEFAULT_TARGET_SIZE = "1080*1080"
SAFETENSORS_CUDA_TO_CPU_ENV = "AYL_SAFETENSORS_CUDA_TO_CPU_PATCH"
SAFETENSORS_PATCH_REPORT_ENV = "AYL_SAFETENSORS_PATCH_REPORT_PATH"
RUNTIME_PATCH_REPORT = {
    "safetensors_cuda_to_cpu_patch": {
        "patch_requested": False,
        "patch_applied": False,
        "patched_calls_count": 0,
        "redirected_devices": [],
    },
    "attention_sdpa_patch": {
        "attention_backend_requested": "auto",
        "flash_attn_available": None,
        "flash_attn_2_available": None,
        "flash_attn_3_available": None,
        "attention_fallback_applied": False,
        "attention_backend_used": "",
        "attention_patch_status": "not_attempted",
        "attention_patch_calls_count": 0,
        "patched_modules": [],
    },
}


def should_redirect_device(device):
    if isinstance(device, str):
        return device.startswith("cuda")
    try:
        return str(device).startswith("cuda")
    except Exception:
        return False


@contextmanager
def scoped_safetensors_cuda_to_cpu_patch(enabled: bool):
    state = {
        "patch_requested": bool(enabled),
        "patch_applied": False,
        "patched_calls_count": 0,
        "redirected_devices": [],
        "restored": False,
    }
    if not enabled:
        yield state
        return
    import safetensors
    import safetensors.torch

    original_load_file = safetensors.torch.load_file
    original_safe_open = safetensors.safe_open
    original_torch_safe_open = getattr(safetensors.torch, "safe_open", original_safe_open)

    def redirect_device(device):
        if should_redirect_device(device):
            state["patched_calls_count"] += 1
            device_text = str(device)
            if device_text not in state["redirected_devices"]:
                state["redirected_devices"].append(device_text)
            return "cpu"
        return device

    def patched_load_file(filename, device="cpu", *args, **kwargs):
        return original_load_file(filename, device=redirect_device(device), *args, **kwargs)

    def patched_safe_open(filename, framework, device="cpu", *args, **kwargs):
        return original_safe_open(filename, framework=framework, device=redirect_device(device), *args, **kwargs)

    def patched_torch_safe_open(filename, framework, device="cpu", *args, **kwargs):
        return original_torch_safe_open(filename, framework=framework, device=redirect_device(device), *args, **kwargs)

    safetensors.torch.load_file = patched_load_file
    safetensors.safe_open = patched_safe_open
    safetensors.torch.safe_open = patched_torch_safe_open
    state["patch_applied"] = True
    try:
        yield state
    finally:
        safetensors.torch.load_file = original_load_file
        safetensors.safe_open = original_safe_open
        safetensors.torch.safe_open = original_torch_safe_open
        state["restored"] = True


def parameter_device_summary(model_obj, sample_limit=2000):
    summary = {
        "first_parameter_device": "",
        "first_parameter_dtype": "",
        "any_parameter_on_cuda": False,
        "parameter_device_counts": {},
        "sampled_parameters": 0,
        "sample_limit": sample_limit,
    }
    if model_obj is None or not hasattr(model_obj, "parameters"):
        return summary
    for idx, parameter in enumerate(model_obj.parameters()):
        if idx >= sample_limit:
            break
        device = str(getattr(parameter, "device", ""))
        dtype = str(getattr(parameter, "dtype", ""))
        if idx == 0:
            summary["first_parameter_device"] = device
            summary["first_parameter_dtype"] = dtype
        summary["parameter_device_counts"][device] = summary["parameter_device_counts"].get(device, 0) + 1
        if device.startswith("cuda"):
            summary["any_parameter_on_cuda"] = True
        summary["sampled_parameters"] += 1
    return summary


def write_patch_report(report: dict) -> None:
    path = os.getenv(SAFETENSORS_PATCH_REPORT_ENV, "")
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def update_patch_report(section: str, payload: dict) -> None:
    RUNTIME_PATCH_REPORT.setdefault(section, {}).update(payload)
    write_patch_report(RUNTIME_PATCH_REPORT)


def install_scoped_from_pretrained_patch() -> None:
    patch_requested = os.getenv(SAFETENSORS_CUDA_TO_CPU_ENV, "") == "1"
    from wan.modules.s2v.model_s2v import WanModel_S2V

    original_from_pretrained = WanModel_S2V.from_pretrained

    def patched_from_pretrained(*args, **kwargs):
        call_args = args[1:] if args and args[0] is WanModel_S2V else args
        model_obj = None
        patch_state = {"patch_requested": patch_requested, "patch_applied": False}
        try:
            with scoped_safetensors_cuda_to_cpu_patch(patch_requested) as active_patch_state:
                model_obj = original_from_pretrained(*call_args, **kwargs)
            patch_state = dict(active_patch_state)
            report = {
                **patch_state,
                "status": "succeeded",
                **parameter_device_summary(model_obj),
            }
            update_patch_report("safetensors_cuda_to_cpu_patch", report)
            return model_obj
        except Exception as exc:
            if "active_patch_state" in locals():
                patch_state = dict(active_patch_state)
            update_patch_report(
                "safetensors_cuda_to_cpu_patch",
                {
                    **patch_state,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_truncated": str(exc)[:1000],
                },
            )
            raise

    WanModel_S2V.from_pretrained = patched_from_pretrained
    return lambda: setattr(WanModel_S2V, "from_pretrained", original_from_pretrained)


def install_sdpa_attention_fallback_patch():
    import wan.modules.attention as attention_module
    import wan.modules.model as model_module
    import wan.modules.s2v.model_s2v as model_s2v_module

    flash2 = bool(getattr(attention_module, "FLASH_ATTN_2_AVAILABLE", False))
    flash3 = bool(getattr(attention_module, "FLASH_ATTN_3_AVAILABLE", False))
    flash_available = flash2 or flash3
    state = {
        "attention_backend_requested": "auto",
        "flash_attn_available": flash_available,
        "flash_attn_2_available": flash2,
        "flash_attn_3_available": flash3,
        "attention_fallback_applied": False,
        "attention_backend_used": "flash_attention" if flash_available else "",
        "attention_patch_status": "not_needed_flash_attention_available" if flash_available else "started",
        "attention_patch_calls_count": 0,
    }
    if flash_available:
        update_patch_report("attention_sdpa_patch", state)
        return lambda: None

    original_attention_flash = attention_module.flash_attention
    original_model_flash = getattr(model_module, "flash_attention", None)
    original_model_s2v_flash = getattr(model_s2v_module, "flash_attention", None)

    def sdpa_flash_attention_compat(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.0,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=None,
        version=None,
    ):
        state["attention_patch_calls_count"] += 1
        state["attention_backend_used"] = "torch_sdpa"
        return attention_module.attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype or q.dtype,
            fa_version=version,
        )

    attention_module.flash_attention = sdpa_flash_attention_compat
    model_module.flash_attention = sdpa_flash_attention_compat
    model_s2v_module.flash_attention = sdpa_flash_attention_compat
    state.update(
        {
            "attention_fallback_applied": True,
            "attention_backend_used": "torch_sdpa",
            "attention_patch_status": "applied",
            "patched_modules": [
                "wan.modules.attention.flash_attention",
                "wan.modules.model.flash_attention",
                "wan.modules.s2v.model_s2v.flash_attention",
            ],
        }
    )
    update_patch_report("attention_sdpa_patch", state)

    def restore():
        attention_module.flash_attention = original_attention_flash
        if original_model_flash is not None:
            model_module.flash_attention = original_model_flash
        if original_model_s2v_flash is not None:
            model_s2v_module.flash_attention = original_model_s2v_flash
        state["attention_patch_status"] = "restored"
        update_patch_report("attention_sdpa_patch", state)

    return restore


def ensure_supported_size(supported_sizes: dict, task: str, size: str) -> None:
    if task not in supported_sizes:
        return
    existing_sizes = supported_sizes[task]
    if size in existing_sizes:
        return
    if isinstance(existing_sizes, tuple):
        supported_sizes[task] = (*existing_sizes, size)
    elif isinstance(existing_sizes, list):
        supported_sizes[task] = [*existing_sizes, size]
    else:
        supported_sizes[task] = tuple([*list(existing_sizes), size])


def target_size_from_argv() -> tuple[str, int, int]:
    size = DEFAULT_TARGET_SIZE
    for index, arg in enumerate(sys.argv):
        if arg == "--size" and index + 1 < len(sys.argv):
            size = sys.argv[index + 1]
            break
    try:
        width_text, height_text = size.lower().replace("x", "*").split("*", 1)
        width = int(width_text)
        height = int(height_text)
    except Exception:
        return DEFAULT_TARGET_SIZE, 1080, 1080
    return f"{width}*{height}", width, height


def main() -> int:
    if str(WAN22_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(WAN22_REPO_DIR))

    import generate
    from wan.configs import SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES

    target_size, target_width, target_height = target_size_from_argv()
    SIZE_CONFIGS[target_size] = (target_width, target_height)
    MAX_AREA_CONFIGS[target_size] = target_width * target_height
    ensure_supported_size(SUPPORTED_SIZES, "s2v-14B", target_size)
    restore_from_pretrained = install_scoped_from_pretrained_patch()
    restore_attention_patch = install_sdpa_attention_fallback_patch()

    args = generate._parse_args()
    generate._validate_args(args)
    try:
        generate.generate(args)
    finally:
        restore_attention_patch()
        restore_from_pretrained()
        write_patch_report(RUNTIME_PATCH_REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
