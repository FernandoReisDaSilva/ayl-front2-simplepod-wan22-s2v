import argparse
from pathlib import Path

import temp_simplepod_check_safetensors_device_blackwell_v1 as base


TEST_ID = "TEMP_SIMPLEPOD_CHECK_WAN22_SAFETENSORS_SHARD_BLACKWELL_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "logs" / "simplepod_wan22_safetensors_shard_blackwell_v1.json"
IMAGE = "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.2.12-blackwell"
CHECK_ENDPOINT = "/admin/check-wan22-safetensors-shard"
RUNTIME_VERSION = "v2-blackwell-wan22-safetensors-shard-diagnostic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or execute SimplePod Blackwell Wan2.2 real safetensors shard diagnostic.")
    parser.add_argument("--execute", action="store_true", help="Create a real SimplePod instance for shard diagnostic.")
    parser.add_argument("--confirm-start", action="store_true", help="Required with --execute.")
    parser.add_argument("--confirm-delete", action="store_true", help="Required with --execute; deletes instance in finally.")
    parser.add_argument("--instance-market", default="", help="Optional explicit /instances/market/{id}.")
    parser.add_argument("--detail-attempts", type=int, default=36)
    parser.add_argument("--poll-interval-seconds", type=int, default=5)
    parser.add_argument("--ready-timeout-seconds", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    base.TEST_ID = TEST_ID
    base.REPORT_PATH = REPORT_PATH
    base.IMAGE = IMAGE
    base.CHECK_ENDPOINT = CHECK_ENDPOINT
    base.RUNTIME_VERSION = RUNTIME_VERSION
    return base.run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
