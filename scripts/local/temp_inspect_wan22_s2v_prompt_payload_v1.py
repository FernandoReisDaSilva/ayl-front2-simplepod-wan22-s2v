import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


TEST_ID = "TEST_LOCAL_WAN22_S2V_PROMPT_PAYLOAD_INSPECT_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = REPO_ROOT / "logs" / "wan22_s2v_prompt_payload_inspect_v1_log.json"
PAYLOAD_CANDIDATES = (
    REPO_ROOT / "logs" / "wan22_s2v_prompt_payload_local_debug_v1.json",
    REPO_ROOT / "logs" / "wan22_s2v_prompt_payload_debug_v1.json",
    REPO_ROOT / "logs" / "wan22_s2v_prompt_payload_debug.json",
)
WORKFLOW_CANDIDATES = (
    REPO_ROOT / "docker" / "wan22-s2v-runpod-v1" / "wanvideo2_2_S2V_context_window_testing.json",
    REPO_ROOT / "docker" / "wan22-s2v-runpod-v1" / "workflows" / "wanvideo2_2_S2V_context_window_testing.json",
    REPO_ROOT / "workflows" / "wanvideo2_2_S2V_context_window_testing.json",
    REPO_ROOT / "data" / "workflows" / "wanvideo2_2_S2V_context_window_testing.json",
)
DECORATIVE_NODE_TYPES = {
    "MarkdownNote",
    "Note",
    "AnythingEverywhere",
    "Reroute",
}
NON_DECORATIVE_NODE_TYPES_TO_PRESERVE = {
    "PrimitiveNode",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_workflow_path(raw_path: str) -> Path | None:
    if raw_path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve()
    for candidate in WORKFLOW_CANDIDATES:
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def looks_like_api_prompt(value) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    sample = list(value.values())[:5]
    return all(isinstance(item, dict) and "class_type" in item for item in sample)


def extract_prompt(document: dict | None) -> tuple[dict, str]:
    if not document:
        return {}, "missing"
    if isinstance(document.get("prompt"), dict):
        return document["prompt"], "prompt"
    for key in ("workflow", "prompt_payload", "api_prompt", "payload"):
        value = document.get(key)
        if isinstance(value, dict) and isinstance(value.get("prompt"), dict):
            return value["prompt"], f"{key}.prompt"
        if looks_like_api_prompt(value):
            return value, key
    if looks_like_api_prompt(document):
        return document, "document"
    return {}, "not_found"


def first_payload_with_prompt() -> tuple[Path | None, dict | None, dict, str]:
    fallback_path = None
    fallback_doc = None
    for candidate in PAYLOAD_CANDIDATES:
        if not candidate.is_file():
            continue
        document = load_json(candidate)
        if fallback_path is None:
            fallback_path = candidate
            fallback_doc = document
        prompt, source = extract_prompt(document)
        if prompt:
            return candidate, document, prompt, source
    return fallback_path, fallback_doc, {}, "not_found"


def prompt_value_is_link(value) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[1], int)


def value_preview(value):
    if prompt_value_is_link(value):
        return {"value_kind": "link", "preview": value}
    if isinstance(value, list):
        return {"value_kind": "list", "preview": value[:6], "length": len(value)}
    if isinstance(value, dict):
        return {"value_kind": "dict", "preview": list(value.keys())[:12], "length": len(value)}
    return {"value_kind": type(value).__name__, "preview": value}


def special_input_nodes(prompt: dict) -> list[dict]:
    target_inputs = {
        "control_embeds",
        "pose_embeds",
        "image_embeds",
        "audio_embeds",
        "clip_embeds",
        "mask",
        "device",
    }
    matches = []
    for node_id, node in prompt.items():
        class_type = node.get("class_type", "")
        for input_name, value in node.get("inputs", {}).items():
            if input_name in target_inputs:
                matches.append(
                    {
                        "node_id": node_id,
                        "class_type": class_type,
                        "input_name": input_name,
                        **value_preview(value),
                    }
                )
    return matches


def link_lookup(workflow: dict) -> dict[int, list]:
    return {int(link[0]): [str(link[1]), int(link[2])] for link in workflow.get("links", [])}


def class_type_counts(prompt: dict) -> dict:
    counts: dict[str, int] = {}
    for node in prompt.values():
        class_type = str(node.get("class_type", ""))
        if class_type:
            counts[class_type] = counts.get(class_type, 0) + 1
    return dict(sorted(counts.items()))


def detect_broken_workflow_links(workflow: dict) -> list[dict]:
    node_ids = {str(node.get("id")) for node in workflow.get("nodes", [])}
    broken = []
    for link in workflow.get("links", []):
        if len(link) < 4:
            broken.append({"reason": "malformed_link", "link": link})
            continue
        source_node_id = str(link[1])
        target_node_id = str(link[3])
        reasons = []
        if source_node_id not in node_ids:
            reasons.append("missing_source_node")
        if target_node_id not in node_ids:
            reasons.append("missing_target_node")
        if reasons:
            broken.append(
                {
                    "reasons": reasons,
                    "link_id": str(link[0]),
                    "source_node_id": source_node_id,
                    "target_node_id": target_node_id,
                    "link": link,
                }
            )
    return broken


def filter_ui_workflow_for_api(workflow: dict) -> tuple[dict, dict]:
    removed_ids = {
        str(node.get("id"))
        for node in workflow.get("nodes", [])
        if node.get("type") in DECORATIVE_NODE_TYPES
    }
    removed_nodes = []
    class_counts: dict[str, int] = {}
    preserved_non_decorative = set()
    filtered_nodes = []
    for node in workflow.get("nodes", []):
        node_id = str(node.get("id"))
        node_type = str(node.get("type", ""))
        if node_id in removed_ids:
            removed_nodes.append({"id": node_id, "class_type": node_type, "title": node.get("title", "")})
            class_counts[node_type] = class_counts.get(node_type, 0) + 1
            continue
        if node_type in NON_DECORATIVE_NODE_TYPES_TO_PRESERVE:
            preserved_non_decorative.add(node_type)
        filtered_nodes.append(node)

    dependent_links = []
    removed_link_ids = set()
    for link in workflow.get("links", []):
        if len(link) < 4:
            continue
        link_id = str(link[0])
        source_node_id = str(link[1])
        target_node_id = str(link[3])
        if source_node_id in removed_ids or target_node_id in removed_ids:
            removed_link_ids.add(link_id)
            dependent_links.append(
                {
                    "link_id": link_id,
                    "source_node_id": source_node_id,
                    "target_node_id": target_node_id,
                    "link": link,
                }
            )

    dependent_inputs = []
    for node in filtered_nodes:
        for item in node.get("inputs", []):
            link_id = item.get("link")
            if link_id is not None and str(link_id) in removed_link_ids:
                dependent_inputs.append(
                    {
                        "node_id": str(node.get("id")),
                        "class_type": node.get("type", ""),
                        "input_name": item.get("name", ""),
                        "link_id": str(link_id),
                    }
                )

    status = "error" if dependent_links or dependent_inputs else "ok"
    filtered_workflow = {**workflow, "nodes": filtered_nodes}
    if status == "ok":
        filtered_workflow["links"] = [
            link for link in workflow.get("links", []) if str(link[0]) not in removed_link_ids
        ]
    return filtered_workflow, {
        "workflow_filter_status": status,
        "workflow_filter_removed_nodes": removed_nodes,
        "workflow_filter_removed_class_type_counts": dict(sorted(class_counts.items())),
        "workflow_filter_preserved_non_decorative_node_classes": sorted(preserved_non_decorative),
        "workflow_filter_dependent_links": dependent_links,
        "workflow_filter_dependent_inputs": dependent_inputs,
    }


def input_order(class_info: dict) -> list[str]:
    info = class_info.get("input", {})
    names = []
    for section in ("required", "optional"):
        values = info.get(section, {})
        if isinstance(values, dict):
            names.extend(values.keys())
    return names


def convert_ui_workflow_to_api(workflow: dict, object_info: dict | None = None) -> tuple[dict, list[dict]]:
    object_info = object_info or {}
    links = link_lookup(workflow)
    prompt = {}
    warnings = []
    for node in workflow.get("nodes", []):
        node_type = node.get("type")
        if not node_type or node_type in DECORATIVE_NODE_TYPES:
            continue
        node_id = str(node["id"])
        inputs = {}
        linked_names = set()
        for item in node.get("inputs", []):
            if item.get("link") is not None:
                link_id = int(item["link"])
                if link_id in links:
                    inputs[item["name"]] = links[link_id]
                    linked_names.add(item["name"])
                else:
                    warnings.append(
                        {
                            "warning": "missing_link_lookup",
                            "node_id": node_id,
                            "class_type": node_type,
                            "input_name": item.get("name", ""),
                            "link_id": str(link_id),
                        }
                    )
        widgets = node.get("widgets_values")
        if isinstance(widgets, dict):
            for key, value in widgets.items():
                if key != "videopreview":
                    inputs[key] = value
        elif isinstance(widgets, list):
            if node_type in object_info:
                candidates = [name for name in input_order(object_info[node_type]) if name not in linked_names]
                for name, value in zip(candidates, widgets):
                    inputs.setdefault(name, value)
            elif widgets:
                warnings.append(
                    {
                        "warning": "widget_list_requires_comfyui_object_info",
                        "node_id": node_id,
                        "class_type": node_type,
                        "widget_value_count": len(widgets),
                    }
                )
        prompt[node_id] = {"class_type": node_type, "inputs": inputs}
    return prompt, warnings


def patch_prompt(prompt: dict) -> list[dict]:
    warnings = []
    patches = {
        "73": {"image": "mae_reference.png"},
        "94": {"audio_file": "input/mae_audio_5s.wav"},
        "27": {
            "steps": 4,
            "cfg": 1.0,
            "shift": 4.0,
            "seed": 42,
            "denoise_strength": 1.0,
        },
        "101": {
            "audio_scale": 1.0,
            "pose_start_percent": 0.0,
            "pose_end_percent": 1.0,
        },
    }
    for node_id, inputs in patches.items():
        if node_id not in prompt:
            warnings.append({"warning": "patch_target_missing", "node_id": node_id})
            continue
        prompt[node_id]["inputs"].update(inputs)
    for node_id in ("30", "97"):
        if node_id in prompt:
            prompt[node_id]["inputs"].update(
                {
                    "filename_prefix": "ayl_wan22_s2v_probe_v1/video_out",
                    "save_output": True,
                    "format": "video/h264-mp4",
                    "trim_to_audio": True,
                }
            )
    return warnings


def detect_decorative_remnants(prompt: dict) -> list[dict]:
    remnants = []
    for node_id, node in prompt.items():
        class_type = node.get("class_type", "")
        if class_type in DECORATIVE_NODE_TYPES:
            remnants.append({"node_id": node_id, "class_type": class_type})
    return remnants


def detect_primitive_nodes(workflow: dict) -> list[dict]:
    primitive_ids = {
        str(node.get("id")): node
        for node in workflow.get("nodes", [])
        if node.get("type") == "PrimitiveNode"
    }
    results = []
    for node_id, node in primitive_ids.items():
        incoming_links = []
        outgoing_links = []
        for link in workflow.get("links", []):
            if len(link) < 4:
                continue
            if str(link[3]) == node_id:
                incoming_links.append(link)
            if str(link[1]) == node_id:
                outgoing_links.append(link)
        results.append(
            {
                "id": node_id,
                "title": node.get("title", ""),
                "functional_preserved": bool(incoming_links or outgoing_links),
                "incoming_link_count": len(incoming_links),
                "outgoing_link_count": len(outgoing_links),
                "incoming_links": incoming_links,
                "outgoing_links": outgoing_links,
            }
        )
    return results


def detect_prompt_broken_links(prompt: dict) -> list[dict]:
    node_ids = set(prompt.keys())
    broken = []
    for node_id, node in prompt.items():
        inputs = node.get("inputs", {})
        for input_name, value in inputs.items():
            if isinstance(value, list) and len(value) == 2:
                source_node_id = str(value[0])
                if source_node_id not in node_ids:
                    broken.append(
                        {
                            "node_id": node_id,
                            "class_type": node.get("class_type", ""),
                            "input_name": input_name,
                            "source_node_id": source_node_id,
                            "reason": "missing_source_node_in_prompt",
                        }
                    )
    return broken


def build_missing_workflow_report(workflow_path: Path | None, requested_path: str) -> dict:
    return {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": "workflow_missing",
        "requested_workflow_path": requested_path,
        "resolved_workflow_path": str(workflow_path) if workflow_path else "",
        "searched_workflow_candidates": [str(path) for path in WORKFLOW_CANDIDATES],
        "no_runpod": True,
        "no_r2": True,
        "no_download": True,
        "no_build_push": True,
        "not_latentsync": True,
        "not_wan27": True,
    }


def inspect(args: argparse.Namespace) -> int:
    payload_path, payload_document, prompt_from_payload, prompt_source = first_payload_with_prompt()
    if prompt_from_payload:
        counts = class_type_counts(prompt_from_payload)
        broken_prompt_links = detect_prompt_broken_links(prompt_from_payload)
        decorative_remnants = detect_decorative_remnants(prompt_from_payload)
        special_inputs = special_input_nodes(prompt_from_payload)
        top_level_keys = list(payload_document.keys()) if isinstance(payload_document, dict) else []
        status = "inspection_found_issues" if broken_prompt_links or decorative_remnants else "ok"
        report = {
            "test_id": TEST_ID,
            "created_at": now_iso(),
            "status": status,
            "payload_path": str(payload_path),
            "payload_source": prompt_source,
            "top_level_keys": top_level_keys,
            "node_count": len(prompt_from_payload),
            "class_type_counts": counts,
            "special_input_nodes": special_inputs,
            "broken_prompt_links": broken_prompt_links,
            "decorative_remnants": decorative_remnants,
            "no_runpod": True,
            "no_r2": True,
            "no_download": True,
            "no_build_push": True,
            "not_latentsync": True,
            "not_wan27": True,
        }
        write_json(LOG_PATH, report)
        print(f"[{TEST_ID}] status={status} log={LOG_PATH}")
        print(f"[{TEST_ID}] payload={payload_path}")
        print(f"[{TEST_ID}] top_level_keys={json.dumps(top_level_keys, ensure_ascii=False)}")
        print(f"[{TEST_ID}] node_count={len(prompt_from_payload)} prompt_source={prompt_source}")
        print(f"[{TEST_ID}] class_type_counts={json.dumps(counts, ensure_ascii=False, sort_keys=True)}")
        print(f"[{TEST_ID}] special_input_nodes={len(special_inputs)} broken_prompt_links={len(broken_prompt_links)}")
        return 0

    workflow_path = resolve_workflow_path(args.workflow_local)
    if workflow_path is None or not workflow_path.is_file():
        report = build_missing_workflow_report(workflow_path, args.workflow_local)
        report.update(
            {
                "payload_path": str(payload_path) if payload_path else "",
                "payload_top_level_keys": list(payload_document.keys()) if isinstance(payload_document, dict) else [],
                "payload_prompt_source": prompt_source,
            }
        )
        write_json(LOG_PATH, report)
        print(f"[{TEST_ID}] workflow_missing log={LOG_PATH}")
        if payload_path:
            print(f"[{TEST_ID}] payload_without_prompt={payload_path}")
            print(f"[{TEST_ID}] top_level_keys={json.dumps(report['payload_top_level_keys'], ensure_ascii=False)}")
        return 0

    object_info = load_json(Path(args.object_info_local).expanduser().resolve()) if args.object_info_local else None
    workflow = load_json(workflow_path)
    broken_workflow_links_before = detect_broken_workflow_links(workflow)
    filtered_workflow, workflow_filter = filter_ui_workflow_for_api(workflow)
    broken_workflow_links_after = detect_broken_workflow_links(filtered_workflow)
    prompt = {}
    conversion_warnings = []
    patch_warnings = []
    if workflow_filter["workflow_filter_status"] == "ok":
        prompt, conversion_warnings = convert_ui_workflow_to_api(filtered_workflow, object_info)
        patch_warnings = patch_prompt(prompt)
    payload = {"prompt": prompt, "client_id": "local-inspection-only"}
    decorative_remnants = detect_decorative_remnants(prompt)
    primitive_nodes = detect_primitive_nodes(filtered_workflow)
    broken_prompt_links = detect_prompt_broken_links(prompt)
    counts = class_type_counts(prompt)
    status = "ok"
    if workflow_filter["workflow_filter_status"] != "ok":
        status = "workflow_filter_error"
    elif broken_workflow_links_before or broken_workflow_links_after or broken_prompt_links or decorative_remnants:
        status = "inspection_found_issues"

    report = {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": status,
        "workflow_path": str(workflow_path),
        "object_info_path": str(Path(args.object_info_local).expanduser().resolve()) if args.object_info_local else "",
        "payload": payload,
        "class_type_counts": counts,
        "workflow_filter": workflow_filter,
        "broken_workflow_links_before_filter": broken_workflow_links_before,
        "broken_workflow_links_after_filter": broken_workflow_links_after,
        "broken_prompt_links": broken_prompt_links,
        "decorative_remnants": decorative_remnants,
        "primitive_nodes": primitive_nodes,
        "primitive_functional_preserved": [item for item in primitive_nodes if item["functional_preserved"]],
        "conversion_warnings": conversion_warnings,
        "patch_warnings": patch_warnings,
        "conversion_limitations": [
            "widgets_values lists require optional --object-info-local from ComfyUI /object_info for exact API input names"
        ]
        if object_info is None
        else [],
        "no_runpod": True,
        "no_r2": True,
        "no_download": True,
        "no_build_push": True,
        "not_latentsync": True,
        "not_wan27": True,
    }
    write_json(LOG_PATH, report)
    print(f"[{TEST_ID}] status={status} log={LOG_PATH}")
    print(f"[{TEST_ID}] workflow={workflow_path}")
    print(f"[{TEST_ID}] class_type_counts={json.dumps(counts, ensure_ascii=False, sort_keys=True)}")
    print(f"[{TEST_ID}] broken_prompt_links={len(broken_prompt_links)} decorative_remnants={len(decorative_remnants)}")
    print(f"[{TEST_ID}] primitive_functional_preserved={len(report['primitive_functional_preserved'])}")
    return 0 if status in {"ok", "inspection_found_issues"} else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Wan2.2 S2V ComfyUI prompt payload locally. No RunPod/R2.")
    parser.add_argument("--workflow-local", default="", help="Optional local workflow JSON path.")
    parser.add_argument("--object-info-local", default="", help="Optional saved ComfyUI /object_info JSON for exact widget mapping.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(inspect(parse_args()))
