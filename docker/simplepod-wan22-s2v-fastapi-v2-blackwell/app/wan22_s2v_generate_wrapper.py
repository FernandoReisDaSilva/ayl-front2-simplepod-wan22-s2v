import os
import sys
from pathlib import Path


WAN22_REPO_DIR = Path("/opt/Wan2.2")
TARGET_SIZE = "1080*1080"
SAFETENSORS_CUDA_TO_CPU_ENV = "AYL_SAFETENSORS_CUDA_TO_CPU_PATCH"


def redirect_device(device):
    if isinstance(device, str) and device.startswith("cuda"):
        return "cpu"
    return device


def install_safetensors_cuda_to_cpu_patch() -> None:
    if os.getenv(SAFETENSORS_CUDA_TO_CPU_ENV, "") != "1":
        return
    import safetensors
    import safetensors.torch

    original_load_file = safetensors.torch.load_file
    original_safe_open = safetensors.safe_open
    original_torch_safe_open = getattr(safetensors.torch, "safe_open", original_safe_open)

    def patched_load_file(filename, device="cpu", *args, **kwargs):
        return original_load_file(filename, device=redirect_device(device), *args, **kwargs)

    def patched_safe_open(filename, framework, device="cpu", *args, **kwargs):
        return original_safe_open(filename, framework=framework, device=redirect_device(device), *args, **kwargs)

    def patched_torch_safe_open(filename, framework, device="cpu", *args, **kwargs):
        return original_torch_safe_open(filename, framework=framework, device=redirect_device(device), *args, **kwargs)

    safetensors.torch.load_file = patched_load_file
    safetensors.safe_open = patched_safe_open
    safetensors.torch.safe_open = patched_torch_safe_open


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


def main() -> int:
    if str(WAN22_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(WAN22_REPO_DIR))

    install_safetensors_cuda_to_cpu_patch()

    import generate
    from wan.configs import SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES

    SIZE_CONFIGS[TARGET_SIZE] = (1080, 1080)
    MAX_AREA_CONFIGS[TARGET_SIZE] = 1080 * 1080
    ensure_supported_size(SUPPORTED_SIZES, "s2v-14B", TARGET_SIZE)

    args = generate._parse_args()
    generate._validate_args(args)
    generate.generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
