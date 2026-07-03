import argparse
from pathlib import Path

import temp_simplepod_check_safetensors_device_blackwell_v1 as base


TEST_ID = "TEMP_SIMPLEPOD_CHECK_WAN22_ACCELERATE_DISPATCH_BLACKWELL_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "logs" / "simplepod_wan22_accelerate_dispatch_blackwell_v1.json"
IMAGE = "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.2.15-blackwell"
CHECK_ENDPOINT = "/admin/check-wan22-accelerate-dispatch"
RUNTIME_VERSION = "v2-blackwell-wan22-accelerate-dispatch-diagnostic"
GPU_POLICY = "blackwell_full_96gb_accelerate_dispatch_diagnostic_policy"
ORIGINAL_BUILD_REPORT = base.build_report


def summarize_accelerate_dispatch(value) -> dict:
    if not isinstance(value, dict):
        return {"json_type": type(value).__name__}
    runtime_path = value.get("wan_s2v_runtime_path", {})
    import_result = runtime_path.get("noise_model", {}) if isinstance(runtime_path, dict) else {}
    selected_import = import_result.get("selected", {}) if isinstance(import_result, dict) else {}
    wan_s2v_pipeline = runtime_path.get("wan_s2v_pipeline", {}) if isinstance(runtime_path, dict) else {}
    from_pretrained_result = value.get("from_pretrained_result", {})
    checkpoint_inventory = value.get("checkpoint_inventory", {})
    return {
        "status": value.get("status"),
        "torch": value.get("torch", {}),
        "versions": value.get("versions", {}),
        "device_map": value.get("device_map"),
        "offload": value.get("offload"),
        "dtype": value.get("dtype"),
        "low_cpu_mem_usage": value.get("low_cpu_mem_usage"),
        "local_files_only": value.get("local_files_only"),
        "checkpoint_paths": {
            "model_dir": checkpoint_inventory.get("model_dir"),
            "safetensors_count": checkpoint_inventory.get("safetensors_count"),
            "safetensors_sample": checkpoint_inventory.get("safetensors_sample", [])[:5],
            "index_json_files": checkpoint_inventory.get("index_json_files", [])[:5],
            "config_json_files": checkpoint_inventory.get("config_json_files", [])[:5],
        },
        "wan_s2v_runtime_path_status": runtime_path.get("status") if isinstance(runtime_path, dict) else "",
        "wan_s2v_pipeline": {
            "status": wan_s2v_pipeline.get("status", ""),
            "module": wan_s2v_pipeline.get("module", ""),
            "class_name": wan_s2v_pipeline.get("class_name", ""),
            "module_file": wan_s2v_pipeline.get("module_file", ""),
            "init_signature": wan_s2v_pipeline.get("init_signature", ""),
            "real_init_noise_model_call": wan_s2v_pipeline.get("real_init_noise_model_call", ""),
        },
        "wan_model_s2v_import_status": import_result.get("status") if isinstance(import_result, dict) else "",
        "wan_model_s2v_import_selected": {
            "module": selected_import.get("module", ""),
            "class_name": selected_import.get("class_name", ""),
            "module_file": selected_import.get("module_file", ""),
            "has_from_pretrained": selected_import.get("has_from_pretrained"),
            "from_pretrained_signature": selected_import.get("from_pretrained_signature", ""),
        },
        "from_pretrained_result": {
            "status": from_pretrained_result.get("status"),
            "error_type": from_pretrained_result.get("error_type", ""),
            "error_truncated": from_pretrained_result.get("error_truncated", ""),
            "traceback_tail": from_pretrained_result.get("traceback_tail", []),
            "object_type": from_pretrained_result.get("object_type", ""),
            "object_module": from_pretrained_result.get("object_module", ""),
            "first_parameter_device": from_pretrained_result.get("first_parameter_device", ""),
            "first_parameter_dtype": from_pretrained_result.get("first_parameter_dtype", ""),
            "any_parameter_on_cuda": from_pretrained_result.get("any_parameter_on_cuda"),
            "parameter_device_counts": from_pretrained_result.get("parameter_device_counts", {}),
        },
        "safetensors_cuda_to_cpu_patch": value.get("safetensors_cuda_to_cpu_patch", {}),
        "download_attempted": value.get("download_attempted"),
        "loads_full_model": value.get("loads_full_model"),
        "sampling_executed": value.get("sampling_executed"),
        "generate_called": value.get("generate_called"),
        "inference_executed": value.get("inference_executed"),
        "video_generated": value.get("video_generated"),
    }


def build_report(args: argparse.Namespace, status: str, data: dict, error: str = "") -> dict:
    report = ORIGINAL_BUILD_REPORT(args, status, data, error)
    result = report.get("safetensors_device_result", {})
    report["accelerate_dispatch_result"] = result
    report["safetensors_device_result"] = {}
    report["gpu_policy"] = GPU_POLICY
    report["image_ref"] = IMAGE
    report["safety_guards"]["loads_full_model"] = True
    report["safety_guards"]["runs_inference"] = False
    report["safety_guards"]["generates_video"] = False
    report["safety_guards"]["downloads_model_weights"] = False
    report["safety_guards"]["uses_mig"] = False
    report["safety_guards"]["mig_allowed_for_diagnostic_only"] = False
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or execute SimplePod Blackwell Wan2.2 accelerate dispatch diagnostic.")
    parser.add_argument("--execute", action="store_true", help="Create a real SimplePod instance for accelerate dispatch diagnostic.")
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
    base.GPU_POLICY = GPU_POLICY
    base.NO_MODEL_LOAD = False
    base.summarize_safetensors_check = summarize_accelerate_dispatch
    base.build_report = build_report
    original_runtime_payload = base.runtime_payload

    def runtime_payload(instance_market: str) -> dict:
        payload = original_runtime_payload(instance_market)
        payload.setdefault("envVariables", []).append(
            {"name": "AYL_SAFETENSORS_CUDA_TO_CPU_PATCH", "value": "1"}
        )
        return payload

    base.runtime_payload = runtime_payload
    return base.run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
