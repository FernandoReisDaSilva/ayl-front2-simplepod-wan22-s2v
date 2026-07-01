import sys
from pathlib import Path


WAN22_REPO_DIR = Path("/opt/Wan2.2")
TARGET_SIZE = "1080*1080"


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
