import json
from datetime import datetime, timezone
from pathlib import Path


TEST_ID = "TEMP_PREPARE_SIMPLEPOD_TEMPLATE_CREATE_DRYRUN_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "logs" / "simplepod_template_create_dryrun_v1.json"

DOCS_URL = "https://api.simplepod.ai/docs"
CREATE_TEMPLATE_ENDPOINT = "POST /instances/templates"
LIST_PRIVATE_TEMPLATES_ENDPOINT = "GET /instances/templates/list"

TEMPLATE_NAME = "ayl-wan22-s2v-fastapi-v1"
DOCKER_IMAGE_PLACEHOLDER = (
    "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v1:0.1.0"
)
IMAGE_NAME = "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v1"
IMAGE_TAG = "0.1.0"
DATACENTER = "EU-PL-01"
VOLUME_NAME = "ayl_models_wan22_s2v_v1"
MODELS_ROOT = "/mnt/ayl_models"
MODEL_DIR = "/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B"
PORT = 8000


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def template_payload() -> dict:
    return {
        "name": TEMPLATE_NAME,
        "imageName": IMAGE_NAME,
        "categoryName": "ayl-wan22-s2v",
        "defaultTag": IMAGE_TAG,
        "diskSize": 32,
        "exposePorts": str(PORT),
        "startScript": "uvicorn app.main:app --host 0.0.0.0 --port 8000",
        "argOptions": "",
        "envVariables": [
            {"name": "SIMPLEPOD_MODELS_ROOT", "value": MODELS_ROOT},
            {"name": "WAN22_S2V_MODEL_DIR", "value": MODEL_DIR},
            {"name": "AYL_IMAGE_TAG", "value": IMAGE_TAG},
            {"name": "PYTHONUNBUFFERED", "value": "1"},
        ],
        "notes": (
            "Dry-run payload for AYL Wan2.2 S2V FastAPI skeleton. "
            "No inference, no model download, no secrets."
        ),
        "isPasswordProtected": False,
        "isRunSshServerOn": False,
        "isRunJupyterOn": False,
    }


def main() -> int:
    required_documented_fields = [
        "name",
        "imageName",
        "defaultTag",
        "diskSize",
        "exposePorts",
        "startScript",
        "envVariables",
    ]
    optional_documented_fields = [
        "host",
        "username",
        "password",
        "categoryName",
        "argOptions",
        "notes",
        "isPasswordProtected",
        "isRunSshServerOn",
        "isRunJupyterOn",
    ]
    payload = template_payload()
    report = {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": "dryrun_created",
        "docs_url": DOCS_URL,
        "identified_endpoints": {
            "create_private_template": CREATE_TEMPLATE_ENDPOINT,
            "list_private_templates": LIST_PRIVATE_TEMPLATES_ENDPOINT,
        },
        "documented_auth_method": "apiKey header X-AUTH-TOKEN",
        "documented_request_body_fields": {
            "required_for_ayl_dryrun": required_documented_fields,
            "optional_or_not_used": optional_documented_fields,
            "doc_required_array_present": False,
            "notes": [
                "OpenAPI requestBody for POST /instances/templates lists properties but no explicit required array.",
                "Volume/network drive and datacenter are not documented as POST /instances/templates request body fields.",
            ],
        },
        "docker_image_placeholder": DOCKER_IMAGE_PLACEHOLDER,
        "payload_dryrun": payload,
        "planned_runtime_context_not_in_template_post_body": {
            "datacenter": DATACENTER,
            "volume_name": VOLUME_NAME,
            "mount_path": MODELS_ROOT,
            "wan22_s2v_model_dir": MODEL_DIR,
            "port": PORT,
            "expected_service_url_path": "/health",
        },
        "safety_guards": {
            "write_endpoint_called": False,
            "template_created": False,
            "pod_created": False,
            "instance_started": False,
            "resource_deleted": False,
            "cost_actions": False,
            "model_downloads": False,
            "docker_build_or_push": False,
            "secrets_printed": False,
        },
        "next_requirements_before_real_template": [
            "Publish GHCR image for the FastAPI skeleton.",
            "Confirm SimplePod UI/API handling for persistent volume mount on template or instance create.",
            "Run a separate explicit approval gate before POST /instances/templates.",
        ],
    }
    write_json(REPORT_PATH, report)

    print(f"[{TEST_ID}] status={report['status']}")
    print(f"[{TEST_ID}] create_endpoint={CREATE_TEMPLATE_ENDPOINT}")
    print(f"[{TEST_ID}] write_endpoint_called=false")
    print(f"[{TEST_ID}] template_created=false")
    print(f"[{TEST_ID}] report={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
