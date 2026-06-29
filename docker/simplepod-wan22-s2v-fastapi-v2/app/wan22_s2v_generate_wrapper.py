import sys
from pathlib import Path


WAN22_REPO_DIR = Path("/opt/Wan2.2")


def main() -> int:
    if str(WAN22_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(WAN22_REPO_DIR))

    import generate
    from wan.configs import SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES

    SIZE_CONFIGS["1080*1080"] = (1080, 1080)
    MAX_AREA_CONFIGS["1080*1080"] = 1080 * 1080
    if "s2v-14B" in SUPPORTED_SIZES and "1080*1080" not in SUPPORTED_SIZES["s2v-14B"]:
        SUPPORTED_SIZES["s2v-14B"].append("1080*1080")

    args = generate._parse_args()
    generate._validate_args(args)
    generate.generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
