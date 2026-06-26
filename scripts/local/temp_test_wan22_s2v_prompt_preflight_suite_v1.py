import copy
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


TEST_ID = "TEST_LOCAL_WAN22_S2V_PROMPT_PREFLIGHT_SUITE_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PROBE_PATH = REPO_ROOT / "docker" / "wan22-s2v-runpod-v1" / "runtime_probe.py"
REPORT_PATH = REPO_ROOT / "review" / "wan22_s2v_offline_preflight_suite_v1.md"
LOG_PATH = REPO_ROOT / "logs" / "wan22_s2v_offline_preflight_suite_v1.json"
PAYLOAD_CANDIDATES = (
    REPO_ROOT / "logs" / "wan22_s2v_prompt_payload_local_debug_v1.json",
    REPO_ROOT / "logs" / "wan22_s2v_prompt_payload_debug_v1.json",
)
FINAL_REPORT_PATH = REPO_ROOT / "logs" / "wan22_s2v_probe_final_report_v1.json"
WORKFLOW_CANDIDATES = (
    REPO_ROOT / "docker" / "wan22-s2v-runpod-v1" / "wanvideo2_2_S2V_context_window_testing.json",
    REPO_ROOT / "docker" / "wan22-s2v-runpod-v1" / "workflows" / "wanvideo2_2_S2V_context_window_testing.json",
    REPO_ROOT / "workflows" / "wanvideo2_2_S2V_context_window_testing.json",
    REPO_ROOT / "data" / "workflows" / "wanvideo2_2_S2V_context_window_testing.json",
)
BANNED_NODE_PREFIXES = ("MelBandRoFormer", "GIMMVFI")
BANNED_NODE_CLASSES = {"MarkdownNote", "PrimitiveNode", "DownloadAndLoadGIMMVFIModel"}
STRUCTURAL_OBJECT_TYPES = {
    "LATENT",
    "EMBEDS",
    "WANVIDIMAGE_EMBEDS",
    "WANVIDAUDIO_EMBEDS",
    "WANVIDEMBEDS",
    "FETAARGS",
    "CACHEARGS",
    "FLOWEDITARGS",
    "SLGARGS",
    "SELECTEDBLOCKS",
    "WANVIDLORA",
    "MASK",
    "IMAGE",
    "AUDIO",
}
ACCEPTED_LITERAL_ALLOWLIST = {
    "WanVideoEmptyEmbeds.width",
    "WanVideoEmptyEmbeds.height",
    "WanVideoEmptyEmbeds.num_frames",
    "WanVideoAddS2VEmbeds.audio_scale",
    "WanVideoAddS2VEmbeds.pose_start_percent",
    "WanVideoAddS2VEmbeds.pose_end_percent",
    "WanVideoSampler.seed",
    "WanVideoSampler.steps",
    "WanVideoSampler.cfg",
    "WanVideoSampler.shift",
    "WanVideoSampler.denoise_strength",
    "WanVideoSampler.scheduler",
    "WanVideoSampler.riflex_freq_index",
    "WanVideoModelLoader.base_precision",
    "WanVideoModelLoader.quantization",
    "WanVideoModelLoader.attention_mode",
    "WanVideoModelLoader.model",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_runtime_probe():
    spec = importlib.util.spec_from_file_location("wan22_runtime_probe", RUNTIME_PROBE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import runtime_probe from {RUNTIME_PROBE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_if_exists(path: Path):
    return load_json(path) if path.is_file() else None


def looks_like_prompt(value) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    return all(isinstance(item, dict) and "class_type" in item for item in list(value.values())[:5])


def extract_prompt(document: dict | None) -> tuple[dict, str]:
    if not document:
        return {}, "missing"
    if isinstance(document.get("prompt"), dict):
        return document["prompt"], "prompt"
    for key in ("payload", "workflow", "prompt_payload", "api_prompt"):
        value = document.get(key)
        if isinstance(value, dict) and isinstance(value.get("prompt"), dict):
            return value["prompt"], f"{key}.prompt"
        if looks_like_prompt(value):
            return value, key
    if looks_like_prompt(document):
        return document, "document"
    return {}, "not_found"


def first_payload() -> tuple[Path | None, dict | None, dict, str]:
    fallback_path = None
    fallback_doc = None
    for path in PAYLOAD_CANDIDATES:
        if not path.is_file():
            continue
        doc = load_json(path)
        if fallback_path is None:
            fallback_path = path
            fallback_doc = doc
        prompt, source = extract_prompt(doc)
        if prompt:
            return path, doc, prompt, source
    return fallback_path, fallback_doc, {}, "not_found"


def recursive_find_dict(data, wanted_keys: tuple[str, ...]):
    if isinstance(data, dict):
        if any(key in data for key in wanted_keys):
            return data
        for value in data.values():
            found = recursive_find_dict(value, wanted_keys)
            if found:
                return found
    elif isinstance(data, list):
        for value in data:
            found = recursive_find_dict(value, wanted_keys)
            if found:
                return found
    return None


def extract_object_info(*documents) -> tuple[dict, str]:
    for label, document in documents:
        if not isinstance(document, dict):
            continue
        direct = document.get("object_info") or document.get("comfyui_object_info")
        if isinstance(direct, dict):
            return direct, label
        found = recursive_find_dict(document, ("WanVideoSampler", "WanVideoAddS2VEmbeds"))
        if isinstance(found, dict) and "WanVideoSampler" in found:
            return found, label
    return {}, "missing"


def first_workflow_path() -> Path | None:
    for path in WORKFLOW_CANDIDATES:
        if path.is_file():
            return path
    return None


def value_is_link(value) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[1], int)


def value_kind(value) -> str:
    if value_is_link(value):
        return "link"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def preview(value):
    if isinstance(value, dict):
        return list(value.keys())[:10]
    if isinstance(value, list):
        return value[:6]
    return value


def add(results: dict, status: str, check: str, detail: str, items=None) -> None:
    results[status].append({"check": check, "detail": detail, "items": items or []})


def class_counts(prompt: dict) -> dict:
    counts = {}
    for node in prompt.values():
        class_type = str(node.get("class_type", ""))
        counts[class_type] = counts.get(class_type, 0) + 1
    return dict(sorted(counts.items()))


def prompt_link_errors(prompt: dict) -> list[dict]:
    node_ids = set(prompt.keys())
    errors = []
    for node_id, node in prompt.items():
        for input_name, value in node.get("inputs", {}).items():
            if value_is_link(value) and str(value[0]) not in node_ids:
                errors.append({"node_id": node_id, "class_type": node.get("class_type"), "input_name": input_name, "value": value})
    return errors


def banned_node_errors(prompt: dict) -> list[dict]:
    errors = []
    for node_id, node in prompt.items():
        class_type = str(node.get("class_type", ""))
        if class_type in BANNED_NODE_CLASSES or any(class_type.startswith(prefix) for prefix in BANNED_NODE_PREFIXES):
            errors.append({"node_id": node_id, "class_type": class_type})
    return errors


def spec_type(spec) -> str:
    if isinstance(spec, (list, tuple)) and spec:
        first = spec[0]
        if isinstance(first, str):
            return first.upper()
        if isinstance(first, list):
            return "OPTIONS"
    return ""


def object_spec(object_info: dict, class_type: str, input_name: str):
    class_info = object_info.get(class_type, {})
    for section in ("required", "optional"):
        values = class_info.get("input", {}).get(section, {}) if isinstance(class_info, dict) else {}
        if isinstance(values, dict) and input_name in values:
            return values[input_name]
    return None


def object_info_literal_errors(prompt: dict, object_info: dict) -> list[dict]:
    if not object_info:
        return []
    errors = []
    for node_id, node in prompt.items():
        class_type = str(node.get("class_type", ""))
        for input_name, value in node.get("inputs", {}).items():
            spec = object_spec(object_info, class_type, input_name)
            if spec_type(spec) not in STRUCTURAL_OBJECT_TYPES:
                continue
            if isinstance(value, (int, str, bool)):
                errors.append(
                    {
                        "node_id": node_id,
                        "class_type": class_type,
                        "input_name": input_name,
                        "value": value,
                        "value_type": type(value).__name__,
                        "object_info_spec": spec,
                    }
                )
    return errors


def specific_rule_errors(prompt: dict) -> list[dict]:
    errors = []
    for node_id, node in prompt.items():
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})
        if class_type == "WanVideoEmptyEmbeds":
            for input_name, value in inputs.items():
                lower = input_name.lower()
                if input_name in {"control_embeds", "extra_latents"} and isinstance(value, (int, str, bool)):
                    errors.append({"node_id": node_id, "class_type": class_type, "input_name": input_name, "value": value, "reason": "empty_embeds_object_literal"})
                elif ("embed" in lower or "latent" in lower) and input_name not in {"width", "height", "num_frames"} and isinstance(value, (int, str, bool)):
                    errors.append({"node_id": node_id, "class_type": class_type, "input_name": input_name, "value": value, "reason": "empty_embeds_structural_literal"})
        if class_type == "WanVideoAddS2VEmbeds":
            for input_name, value in inputs.items():
                lower = input_name.lower()
                if input_name in {"pose_latent", "image_embeds", "audio_encoder_output", "embeds", "clip_embeds", "control_embeds"} and isinstance(value, (int, str, bool)):
                    errors.append({"node_id": node_id, "class_type": class_type, "input_name": input_name, "value": value, "reason": "add_s2v_structural_literal"})
                elif ("latent" in lower or "embed" in lower) and input_name not in {"audio_scale", "pose_start_percent", "pose_end_percent"} and isinstance(value, (int, str, bool)):
                    errors.append({"node_id": node_id, "class_type": class_type, "input_name": input_name, "value": value, "reason": "add_s2v_wildcard_structural_literal"})
        if class_type == "WanVideoSampler":
            for input_name in ("samples", "cache_args", "flowedit_args", "feta_args", "slg_args"):
                value = inputs.get(input_name)
                if isinstance(value, (int, str, bool)):
                    errors.append({"node_id": node_id, "class_type": class_type, "input_name": input_name, "value": value, "reason": "sampler_args_structural_literal"})
            if "batched_cfg" in inputs and not isinstance(inputs.get("batched_cfg"), bool):
                errors.append({"node_id": node_id, "class_type": class_type, "input_name": "batched_cfg", "value": inputs.get("batched_cfg"), "reason": "batched_cfg_not_bool"})
    return errors


def corrected_rule_errors(prompt: dict) -> list[dict]:
    errors = []
    for node_id, node in prompt.items():
        class_type = node.get("class_type", "")
        for input_name, value in node.get("inputs", {}).items():
            lower_value = str(value).lower() if isinstance(value, str) else ""
            if input_name == "attention_mode" and "sage" in lower_value:
                errors.append({"node_id": node_id, "class_type": class_type, "input_name": input_name, "value": value, "reason": "sageattention_enabled"})
            if input_name == "base_precision" and "fast" in lower_value:
                errors.append({"node_id": node_id, "class_type": class_type, "input_name": input_name, "value": value, "reason": "fast_precision"})
            if isinstance(value, str) and ("<tr" in value.lower() or "<td" in value.lower() or "</" in value.lower()):
                errors.append({"node_id": node_id, "class_type": class_type, "input_name": input_name, "value": value[:200], "reason": "html_string"})
        if class_type == "ImageResizeKJv2":
            inputs = node.get("inputs", {})
            if isinstance(inputs.get("mask"), str):
                errors.append({"node_id": node_id, "class_type": class_type, "input_name": "mask", "value": inputs.get("mask"), "reason": "mask_string"})
            if inputs.get("upscale_method") == "lanczos" and inputs.get("device") == "gpu":
                errors.append({"node_id": node_id, "class_type": class_type, "input_name": "device", "value": "gpu", "reason": "lanczos_gpu"})
    return errors


def suspicious_nodes(prompt: dict) -> list[dict]:
    items = []
    for node_id, node in prompt.items():
        class_type = node.get("class_type", "")
        for input_name, value in node.get("inputs", {}).items():
            key = f"{class_type}.{input_name}"
            lower = input_name.lower()
            if key in ACCEPTED_LITERAL_ALLOWLIST:
                continue
            if ("latent" in lower or "embed" in lower or "args" in lower) and isinstance(value, (int, str, bool)):
                items.append({"node_id": node_id, "class_type": class_type, "input_name": input_name, "value": value, "value_kind": value_kind(value)})
    return items


def table(items: list[dict], columns: list[str], limit: int = 60) -> str:
    if not items:
        return "Nenhum item.\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for item in items[:limit]:
        cells = []
        for column in columns:
            value = item.get(column, "")
            text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
            cells.append(text[:220])
        lines.append("| " + " | ".join(cells) + " |")
    if len(items) > limit:
        lines.append(f"\nItens omitidos: {len(items) - limit}\n")
    return "\n".join(lines) + "\n"


def render_report(report: dict) -> str:
    lines = [
        "# Wan2.2 S2V Offline Preflight Suite V1",
        "",
        f"Criado em: `{report['created_at']}`",
        "",
        "## Escopo",
        "",
        "- Bateria local/offline.",
        "- RunPod pausado.",
        "- Sem build/push, sem upload R2, sem alteracoes em LatentSync ou WAN 2.7.",
        "",
        "## Fontes",
        "",
        f"- payload: `{report.get('payload_path') or 'nao encontrado'}`",
        f"- final_report: `{report.get('final_report_path') or 'nao encontrado'}`",
        f"- workflow: `{report.get('workflow_path') or 'nao encontrado'}`",
        f"- object_info: `{report.get('object_info_source')}`",
        f"- prompt_source: `{report.get('prompt_source')}`",
        "",
        "## PASS Checks",
        "",
        table(report["pass_checks"], ["check", "detail"]),
        "## FAIL Checks",
        "",
        table(report["fail_checks"], ["check", "detail"]),
        "## WARN Checks",
        "",
        table(report["warn_checks"], ["check", "detail"]),
        "## Nodes Still Suspicious",
        "",
        table(report["nodes_still_suspicious"], ["node_id", "class_type", "input_name", "value_kind", "value"]),
        "## Proposed Fixes For 0.1.16",
        "",
    ]
    lines.extend(f"- {item}" for item in report["proposed_fixes_0_1_16"])
    lines.extend(
        [
            "",
            "## Allowlist De Literais Aceitos",
            "",
            table([{"input": item} for item in sorted(ACCEPTED_LITERAL_ALLOWLIST)], ["input"], 120),
            "## Sanitized Runtime Reports",
            "",
            "```json",
            json.dumps(report["runtime_reports"], ensure_ascii=False, indent=2)[:6000],
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    runtime = load_runtime_probe()
    payload_path, payload_doc, prompt, prompt_source = first_payload()
    final_report = load_json_if_exists(FINAL_REPORT_PATH) or {}
    object_info, object_info_source = extract_object_info(("payload", payload_doc), ("final_report", final_report))
    workflow_path = first_workflow_path()
    results = {"pass": [], "fail": [], "warn": []}

    add(results, "pass" if payload_doc else "fail", "json_loadable", f"payload_path={payload_path or ''}")
    add(results, "pass" if prompt else "fail", "prompt_found", f"source={prompt_source} nodes={len(prompt)}")
    if prompt:
        missing_class = [{"node_id": node_id} for node_id, node in prompt.items() if not node.get("class_type")]
        add(results, "pass" if not missing_class else "fail", "class_type_present", f"missing={len(missing_class)}", missing_class)
        bad_inputs = [{"node_id": node_id, "class_type": node.get("class_type")} for node_id, node in prompt.items() if not isinstance(node.get("inputs", {}), dict)]
        add(results, "pass" if not bad_inputs else "fail", "inputs_dict_present", f"bad={len(bad_inputs)}", bad_inputs)
        link_errors = prompt_link_errors(prompt)
        add(results, "pass" if not link_errors else "fail", "prompt_links_valid", f"errors={len(link_errors)}", link_errors)
        banned = banned_node_errors(prompt)
        add(results, "pass" if not banned else "fail", "known_removed_nodes_absent", f"errors={len(banned)}", banned)
    else:
        link_errors = banned = []

    raw_object_errors = object_info_literal_errors(prompt, object_info)
    add(results, "pass" if not raw_object_errors else "fail", "object_info_structural_literals_raw", f"errors={len(raw_object_errors)}", raw_object_errors)
    raw_specific_errors = specific_rule_errors(prompt)
    add(results, "pass" if not raw_specific_errors else "fail", "specific_wanvideo_rules_raw", f"errors={len(raw_specific_errors)}", raw_specific_errors)
    raw_corrected_errors = corrected_rule_errors(prompt)
    add(results, "pass" if not raw_corrected_errors else "fail", "previously_fixed_rules_raw", f"errors={len(raw_corrected_errors)}", raw_corrected_errors)
    if not workflow_path:
        add(results, "warn", "workflow_original_available", "workflow original not found locally")
    if not object_info:
        add(results, "warn", "object_info_available", "object_info not found in payload/final_report; object_info checks are heuristic-limited")

    final_prompt = copy.deepcopy(prompt)
    sanitize_report = {}
    preflight_report = {"prompt_semantics_preflight_status": "not_run"}
    if final_prompt:
        final_prompt, sanitize_report = runtime.sanitize_prompt_values(final_prompt, object_info)
        preflight_report = runtime.preflight_prompt_semantics(final_prompt, object_info)
        add(results, "pass" if sanitize_report["prompt_sanitize_status"] == "ok" else "fail", "runtime_sanitize_final", sanitize_report["prompt_sanitize_status"], sanitize_report.get("prompt_sanitize_errors"))
        add(results, "pass" if preflight_report["prompt_semantics_preflight_status"] == "ok" else "fail", "runtime_preflight_final", preflight_report["prompt_semantics_preflight_status"], preflight_report.get("prompt_semantics_preflight_errors"))

    final_suspicious = suspicious_nodes(final_prompt)
    add(results, "pass" if not final_suspicious else "fail", "final_structural_literals_after_sanitize", f"suspicious={len(final_suspicious)}", final_suspicious)

    proposed_fixes = [
        "Keep RunPod paused until this suite shows PASS for runtime_sanitize_final, runtime_preflight_final, and final_structural_literals_after_sanitize.",
        "Use sanitize_wanvideo_structural_literals to neutralize WanVideoAddS2VEmbeds.pose_latent=1 and any remaining WanVideo structural literal.",
        "Preserve scalar allowlist only for known scalar controls such as width, height, num_frames, seed, steps, cfg, shift, scheduler, and timing/audio scale controls.",
        "When object_info becomes available locally, rerun this suite with exact ComfyUI type validation before tagging 0.1.16.",
    ]

    report = {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "payload_path": str(payload_path) if payload_path else "",
        "final_report_path": str(FINAL_REPORT_PATH) if FINAL_REPORT_PATH.is_file() else "",
        "workflow_path": str(workflow_path) if workflow_path else "",
        "object_info_source": object_info_source,
        "prompt_source": prompt_source,
        "prompt_node_count": len(prompt),
        "class_type_counts": class_counts(prompt),
        "pass_checks": results["pass"],
        "fail_checks": results["fail"],
        "warn_checks": results["warn"],
        "nodes_still_suspicious": final_suspicious,
        "proposed_fixes_0_1_16": proposed_fixes,
        "runtime_reports": {"sanitize": sanitize_report, "preflight": preflight_report},
        "no_runpod": True,
        "no_build_push": True,
        "not_latentsync": True,
        "not_wan27": True,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(report), encoding="utf-8")
    print(f"[{TEST_ID}] prompt_nodes={len(prompt)} pass={len(results['pass'])} fail={len(results['fail'])} warn={len(results['warn'])}")
    print(f"[{TEST_ID}] report={REPORT_PATH}")
    print(f"[{TEST_ID}] log={LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
