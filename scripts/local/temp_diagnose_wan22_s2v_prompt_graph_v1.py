import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


TEST_ID = "TEST_LOCAL_WAN22_S2V_PROMPT_GRAPH_DIAGNOSTIC_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PROBE_PATH = REPO_ROOT / "docker" / "wan22-s2v-runpod-v1" / "runtime_probe.py"
REPORT_PATH = REPO_ROOT / "review" / "wan22_s2v_prompt_graph_diagnostic_v1.md"
LOG_PATH = REPO_ROOT / "logs" / "wan22_s2v_prompt_graph_diagnostic_v1.json"
FINAL_REPORT_CANDIDATES = (
    REPO_ROOT / "logs" / "wan22_s2v_probe_final_report_v1.json",
    REPO_ROOT / "logs" / "wan22_s2v_probe_final_report_download_v1_log.json",
)
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
EMBED_INPUT_NAMES = {
    "control_embeds",
    "pose_embeds",
    "image_embeds",
    "audio_embeds",
    "clip_embeds",
}
STRUCTURAL_INPUT_NAME_TOKENS = ("latent", "latents", "embed", "embeds", "args", "mask", "image", "audio")
SCALAR_INPUT_ALLOWLIST = {
    "width",
    "height",
    "num_frames",
    "frame_window_size",
    "steps",
    "cfg",
    "shift",
    "seed",
    "denoise_strength",
    "audio_scale",
    "pose_start_percent",
    "pose_end_percent",
    "precision",
    "base_precision",
    "quantization",
    "attention_mode",
    "device",
    "scheduler",
    "riflex_freq_index",
    "normalization",
    "noise_aug_strength",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_if_exists(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def first_existing(candidates: tuple[Path, ...]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def first_payload_with_prompt(candidates: tuple[Path, ...]) -> tuple[Path | None, dict | None, str]:
    fallback_path = None
    fallback_doc = None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        doc = load_json_if_exists(candidate)
        if fallback_path is None:
            fallback_path = candidate
            fallback_doc = doc
        prompt, source = extract_prompt(doc)
        if prompt:
            return candidate, doc, source
    return fallback_path, fallback_doc, "not_found"


def load_runtime_probe():
    spec = importlib.util.spec_from_file_location("wan22_runtime_probe", RUNTIME_PROBE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import runtime_probe from {RUNTIME_PROBE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_prompt(payload_doc: dict | None) -> tuple[dict, str]:
    if not payload_doc:
        return {}, "missing"
    if isinstance(payload_doc.get("prompt"), dict):
        return payload_doc["prompt"], "payload.prompt"
    for key in ("workflow", "prompt_payload", "api_prompt"):
        value = payload_doc.get(key)
        if isinstance(value, dict) and isinstance(value.get("prompt"), dict):
            return value["prompt"], f"payload.{key}.prompt"
        if looks_like_api_prompt(value):
            return value, f"payload.{key}"
    payload = payload_doc.get("payload")
    if isinstance(payload, dict) and isinstance(payload.get("prompt"), dict):
        return payload["prompt"], "payload.payload.prompt"
    if looks_like_api_prompt(payload_doc):
        return payload_doc, "payload_direct"
    return {}, "not_found"


def looks_like_api_prompt(value) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    sample = list(value.values())[:5]
    return all(isinstance(item, dict) and "class_type" in item for item in sample)


def prompt_value_is_link(value) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[1], int)


def prompt_value_is_literal(value) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def input_name_is_structural(input_name: str) -> bool:
    lower_name = input_name.lower()
    return any(token in lower_name for token in STRUCTURAL_INPUT_NAME_TOKENS)


def local_wanvideo_structural_literal_errors(prompt: dict, object_info: dict | None = None) -> list[dict]:
    errors = []
    object_info = object_info or {}
    for node_id, node in prompt.items():
        class_type = str(node.get("class_type", ""))
        if not class_type.startswith("WanVideo"):
            continue
        for input_name, value in node.get("inputs", {}).items():
            if not isinstance(value, (int, str, bool)):
                continue
            if not input_name_is_structural(input_name):
                continue
            if input_name in SCALAR_INPUT_ALLOWLIST:
                continue
            spec = None
            class_info = object_info.get(class_type, {})
            for section in ("required", "optional"):
                values = class_info.get("input", {}).get(section, {}) if isinstance(class_info, dict) else {}
                if isinstance(values, dict) and input_name in values:
                    spec = values[input_name]
                    break
            if input_name == "latent_strength" and isinstance(spec, (list, tuple)) and spec and spec[0] == "FLOAT":
                continue
            errors.append(
                {
                    "node_id": node_id,
                    "class_type": class_type,
                    "input_name": input_name,
                    "value": value,
                    "value_type": type(value).__name__,
                    "object_info_spec": spec,
                    "reason": "wanvideo_structural_literal_error",
                }
            )
    return errors


def safe_repr(value, limit: int = 220) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    if len(text) <= limit:
        return text
    return text[:limit] + f"... ({len(text) - limit} chars truncated)"


def value_preview(value) -> dict:
    if prompt_value_is_link(value):
        return {"value_kind": "link", "preview": value}
    if isinstance(value, list):
        return {"value_kind": "list", "preview": value[:6], "length": len(value)}
    if isinstance(value, dict):
        return {"value_kind": "dict", "preview": list(value.keys())[:12], "length": len(value)}
    return {"value_kind": type(value).__name__, "preview": value}


def class_type_counts(prompt: dict) -> dict:
    counts: dict[str, int] = {}
    for node in prompt.values():
        class_type = str(node.get("class_type", ""))
        if class_type:
            counts[class_type] = counts.get(class_type, 0) + 1
    return dict(sorted(counts.items()))


def workflow_primitive_link_diagnostics(workflow: dict | None) -> list[dict]:
    if not workflow:
        return []
    primitive_ids = {
        str(node.get("id")): node
        for node in workflow.get("nodes", [])
        if node.get("type") == "PrimitiveNode"
    }
    results = []
    for link in workflow.get("links", []):
        if len(link) < 6 or str(link[1]) not in primitive_ids:
            continue
        results.append(
            {
                "link_id": str(link[0]),
                "primitive_node_id": str(link[1]),
                "primitive_title": primitive_ids[str(link[1])].get("title", ""),
                "source_output_index": link[2],
                "target_node_id": str(link[3]),
                "target_input_index": link[4],
                "target_input_type": link[5],
            }
        )
    return results


def prompt_primitive_link_diagnostics(prompt: dict) -> list[dict]:
    primitive_ids = {
        node_id
        for node_id, node in prompt.items()
        if node.get("class_type") == "PrimitiveNode"
    }
    results = []
    for node_id, node in prompt.items():
        for input_name, value in node.get("inputs", {}).items():
            if prompt_value_is_link(value) and str(value[0]) in primitive_ids:
                results.append(
                    {
                        "node_id": node_id,
                        "class_type": node.get("class_type", ""),
                        "input_name": input_name,
                        "primitive_source_node_id": str(value[0]),
                        "source_output_index": value[1],
                    }
                )
    return results


def diagnose_prompt_inputs(prompt: dict, object_info: dict | None = None) -> dict:
    object_info = object_info or {}
    structural_errors = []
    literal_where_link_expected = []
    wanvideo_optional_misaligned = []
    special_s2v_inputs = []
    suspicious_values = []

    for node_id, node in prompt.items():
        class_type = str(node.get("class_type", ""))
        inputs = node.get("inputs", {})
        for input_name, value in inputs.items():
            lower_name = input_name.lower()
            item = {
                "node_id": node_id,
                "class_type": class_type,
                "input_name": input_name,
                "value": value,
                **value_preview(value),
            }
            if (
                class_type == "WanVideoEmptyEmbeds"
                and input_name in {"control_embeds", "extra_latents"}
                and isinstance(value, (int, str, bool))
            ):
                structural_errors.append(
                    {
                        **item,
                        "reason": f"wanvideo_empty_embeds_invalid_{input_name}",
                    }
                )
            expects_link = any(token in lower_name for token in ("embed", "image", "audio", "mask", "latent", "model"))
            if expects_link and prompt_value_is_literal(value):
                literal_where_link_expected.append({**item, "reason": "literal_in_link_or_object_like_input"})
            if input_name in EMBED_INPUT_NAMES:
                special_s2v_inputs.append(
                    {
                        **item,
                        "value_kind": "link" if prompt_value_is_link(value) else type(value).__name__,
                        "is_suspicious_literal": prompt_value_is_literal(value),
                    }
                )
            if class_type.startswith("WanVideo") and prompt_value_is_literal(value):
                class_info = object_info.get(class_type, {})
                optional_inputs = class_info.get("input", {}).get("optional", {})
                if isinstance(optional_inputs, dict) and input_name in optional_inputs:
                    wanvideo_optional_misaligned.append(
                        {
                            **item,
                            "object_info_spec": optional_inputs.get(input_name),
                            "reason": "optional_wanvideo_literal_review_needed",
                        }
                    )
            if isinstance(value, str) and ("<tr" in value.lower() or "<td" in value.lower() or "</" in value.lower()):
                suspicious_values.append({**item, "reason": "html_string", "value": value[:1000]})
            elif ("tensor" in lower_name or "mask" in lower_name or "embed" in lower_name) and isinstance(value, str):
                suspicious_values.append({**item, "reason": "string_in_tensor_mask_or_embed_input"})
            elif ("embed" in lower_name or "dict" in lower_name or "control" in lower_name) and isinstance(value, int):
                suspicious_values.append({**item, "reason": "int_in_embed_dict_or_control_input"})
            elif ("scheduler" in lower_name or "mode" in lower_name) and isinstance(value, bool):
                suspicious_values.append({**item, "reason": "bool_in_scheduler_or_mode_input"})
            elif input_name == "device" and isinstance(value, str) and value not in {"cpu", "gpu"}:
                suspicious_values.append({**item, "reason": "invalid_device_value"})
            elif lower_name != "device" and isinstance(value, str) and value in {"cpu", "gpu"}:
                suspicious_values.append({**item, "reason": "device_string_in_non_device_input"})

    return {
        "structural_errors": structural_errors + local_wanvideo_structural_literal_errors(prompt, object_info),
        "literal_where_link_expected": literal_where_link_expected,
        "wanvideo_optional_misaligned": wanvideo_optional_misaligned,
        "s2v_embed_inputs": special_s2v_inputs,
        "suspicious_values": suspicious_values,
    }


def object_info_summary(object_info: dict | None, prompt: dict) -> list[dict]:
    if not object_info:
        return []
    results = []
    for node_id, node in prompt.items():
        class_type = str(node.get("class_type", ""))
        class_info = object_info.get(class_type)
        if not isinstance(class_info, dict):
            continue
        known_inputs = set()
        for section in ("required", "optional"):
            values = class_info.get("input", {}).get(section, {})
            if isinstance(values, dict):
                known_inputs.update(values.keys())
        unknown = sorted(set(node.get("inputs", {}).keys()) - known_inputs)
        if unknown:
            results.append({"node_id": node_id, "class_type": class_type, "unknown_inputs": unknown})
    return results


def format_table(items: list[dict], columns: list[str], limit: int = 40) -> str:
    if not items:
        return "Nenhum item encontrado.\n"
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for item in items[:limit]:
        rows.append("| " + " | ".join(safe_repr(item.get(column, "")) for column in columns) + " |")
    if len(items) > limit:
        rows.append(f"\nItens adicionais omitidos: {len(items) - limit}\n")
    return "\n".join(rows) + "\n"


def render_markdown(report: dict) -> str:
    preflight = report.get("runtime_preflight", {})
    prompt_diag = report.get("prompt_diagnostics", {})
    final_report = report.get("final_report_summary", {})
    lines = [
        "# Wan2.2 S2V Prompt Graph Diagnostic V1",
        "",
        f"Criado em: `{report['created_at']}`",
        "",
        "## Escopo",
        "",
        "- Diagnostico local/offline apenas.",
        "- RunPod pausado para o probe Wan2.2 S2V ate consolidar os fixes.",
        "- Sem upload R2, sem build/push, sem download de pesos.",
        "- Sem alteracoes em LatentSync ou WAN 2.7.",
        "",
        "## Fontes Locais",
        "",
        f"- workflow: `{report.get('workflow_path') or 'nao encontrado'}`",
        f"- payload debug: `{report.get('payload_path') or 'nao encontrado'}`",
        f"- final_report: `{report.get('final_report_path') or 'nao encontrado'}`",
        f"- prompt_source: `{report.get('prompt_source')}`",
        "",
        "## Estado Do Probe 0.1.15",
        "",
        "- contexto informado: `0.1.15 passou WanVideoEmptyEmbeds control_embeds e extra_latents`",
        "- novo erro informado: `TypeError: 'int' object is not subscriptable`",
        "- ponto informado: `s2v/nodes.py line 114 pose_latent[\"samples\"]`",
        "- interpretacao: `WanVideoAddS2VEmbeds.pose_latent=1 e outros literais estruturais precisam de saneamento em lote`",
        "",
        "## Final Report Local Disponivel",
        "",
        f"- runtime_probe_status: `{final_report.get('runtime_probe_status', '')}`",
        f"- output_upload_status: `{final_report.get('output_upload_status', '')}`",
        f"- foco local: `{final_report.get('error_focus', '')}`",
        "",
        "## Preflight Semantico",
        "",
        f"- status: `{preflight.get('prompt_semantics_preflight_status', 'not_run')}`",
        f"- erros: `{preflight.get('prompt_semantics_preflight_errors', [])}`",
        "",
        "### control/embed",
        "",
        format_table(
            preflight.get("prompt_semantics_primitive_embed_inputs", []),
            ["node_id", "class_type", "input_name", "value"],
        ),
        "### Inputs S2V Principais",
        "",
        format_table(
            prompt_diag.get("s2v_embed_inputs", []),
            ["node_id", "class_type", "input_name", "value_kind", "value"],
        ),
        "### Valores Suspeitos",
        "",
        format_table(
            prompt_diag.get("suspicious_values", []),
            ["node_id", "class_type", "input_name", "reason", "value"],
        ),
        "### Erros Estruturais",
        "",
        format_table(
            prompt_diag.get("structural_errors", []),
            ["node_id", "class_type", "input_name", "reason", "value"],
        ),
        "### Literais Onde Link/Objeto Era Esperado",
        "",
        format_table(
            prompt_diag.get("literal_where_link_expected", []),
            ["node_id", "class_type", "input_name", "reason", "value"],
        ),
        "### Links Para PrimitiveNode",
        "",
        format_table(
            report.get("prompt_primitive_links", []),
            ["node_id", "class_type", "input_name", "primitive_source_node_id"],
        ),
        "### PrimitiveNode No Workflow Original",
        "",
        format_table(
            report.get("workflow_primitive_links", []),
            ["link_id", "primitive_node_id", "primitive_title", "target_node_id", "target_input_type"],
        ),
        "### WanVideo Opcionais Para Revisao",
        "",
        format_table(
            prompt_diag.get("wanvideo_optional_misaligned", []),
            ["node_id", "class_type", "input_name", "reason", "value"],
        ),
        "## Fixes Propostos Para Tag 0.1.16",
        "",
        "1. `0.1.15` corrigiu `control_embeds=832` e `extra_latents=480` no `WanVideoEmptyEmbeds` node `37`.",
        "2. O novo bloqueio confirmou `WanVideoAddS2VEmbeds.pose_latent=1` como literal `int`.",
        "3. Decisao V1: sanitizar genericamente literais estruturais em `WanVideo*` quando o input parecer `latent`, `embed`, `args`, `mask`, `image` ou `audio`, preservando apenas allowlist escalar explicita.",
        "4. Manter o preflight `preflight_prompt_semantics(prompt, object_info)` antes do payload debug e antes do POST `/prompt`.",
        "5. Rodar `temp_test_wan22_s2v_prompt_preflight_suite_v1.py` antes de qualquer nova tag RunPod.",
        "",
        "## Proxima Tag Sugerida",
        "",
        "```text",
        "0.1.16",
        "```",
        "",
        "## Observacoes",
        "",
    ]
    if report.get("limitations"):
        lines.extend(f"- {item}" for item in report["limitations"])
    else:
        lines.append("- Nenhuma limitacao adicional registrada.")
    lines.append("")
    return "\n".join(lines)


def diagnose(args: argparse.Namespace) -> int:
    runtime_probe = load_runtime_probe()
    workflow_path = Path(args.workflow).expanduser().resolve() if args.workflow else first_existing(WORKFLOW_CANDIDATES)
    if args.payload:
        payload_path = Path(args.payload).expanduser().resolve()
        payload_doc = load_json_if_exists(payload_path)
        prompt, prompt_source = extract_prompt(payload_doc)
    else:
        payload_path, payload_doc, prompt_source = first_payload_with_prompt(PAYLOAD_CANDIDATES)
        prompt, prompt_source = extract_prompt(payload_doc)
    final_report_path = Path(args.final_report).expanduser().resolve() if args.final_report else first_existing(FINAL_REPORT_CANDIDATES)
    object_info_path = Path(args.object_info).expanduser().resolve() if args.object_info else None

    workflow = load_json_if_exists(workflow_path) if workflow_path else None
    final_report = load_json_if_exists(final_report_path) if final_report_path else {}
    object_info = load_json_if_exists(object_info_path) if object_info_path else {}

    preflight = runtime_probe.preflight_prompt_semantics(prompt, object_info) if prompt else {
        "prompt_semantics_preflight_status": "not_run",
        "prompt_semantics_preflight_errors": ["No prompt payload available locally."],
    }
    prompt_diagnostics = diagnose_prompt_inputs(prompt, object_info)
    error_focus = final_report.get("error_truncated") or final_report.get("traceback") or ""
    if not error_focus and final_report.get("runtime_probe_status"):
        error_focus = "0.1.13 reported TypeError in WanVideoAddS2VEmbeds/control_embeds from probe context."

    limitations = []
    if not workflow:
        limitations.append("Workflow original nao foi encontrado nos caminhos locais padrao.")
    if not prompt:
        limitations.append("Payload debug mais recente nao foi encontrado em logs; diagnostico ficou limitado ao final_report/local metadata.")
    if not object_info:
        limitations.append("object_info nao foi fornecido; comparacao exata de tipos/opcoes do ComfyUI ficou heuristica.")
    if final_report.get("runtime_probe_status") and final_report.get("runtime_probe_status") != "failed":
        limitations.append("final_report local disponivel pode estar defasado em relacao ao erro 0.1.13 informado nesta tarefa.")

    report = {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "workflow_path": str(workflow_path) if workflow_path else "",
        "payload_path": str(payload_path) if payload_path else "",
        "final_report_path": str(final_report_path) if final_report_path else "",
        "object_info_path": str(object_info_path) if object_info_path else "",
        "prompt_source": prompt_source,
        "prompt_node_count": len(prompt),
        "class_type_counts": class_type_counts(prompt),
        "runtime_preflight": preflight,
        "prompt_diagnostics": prompt_diagnostics,
        "prompt_primitive_links": prompt_primitive_link_diagnostics(prompt),
        "workflow_primitive_links": workflow_primitive_link_diagnostics(workflow),
        "object_info_unknown_inputs": object_info_summary(object_info, prompt),
        "final_report_summary": {
            "runtime_probe_status": final_report.get("runtime_probe_status", ""),
            "output_upload_status": final_report.get("output_upload_status", ""),
            "prompt_sanitize_status": final_report.get("prompt_sanitize_status", ""),
            "error_focus": error_focus[:2000],
        },
        "recommended_tag": "0.1.14",
        "limitations": limitations,
        "no_runpod": True,
        "no_r2": True,
        "no_download": True,
        "no_build_push": True,
        "not_latentsync": True,
        "not_wan27": True,
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_markdown(report), encoding="utf-8")

    print(f"[{TEST_ID}] status={preflight.get('prompt_semantics_preflight_status')}")
    print(f"[{TEST_ID}] prompt_nodes={len(prompt)} report={REPORT_PATH}")
    print(f"[{TEST_ID}] log={LOG_PATH}")
    if limitations:
        print(f"[{TEST_ID}] limitations={len(limitations)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Wan2.2 S2V prompt graph diagnostic. No RunPod/R2.")
    parser.add_argument("--workflow", default="", help="Optional original workflow JSON path.")
    parser.add_argument("--payload", default="", help="Optional prompt payload debug JSON path.")
    parser.add_argument("--final-report", default="", help="Optional final_report JSON path.")
    parser.add_argument("--object-info", default="", help="Optional saved ComfyUI /object_info JSON path.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(diagnose(parse_args()))
