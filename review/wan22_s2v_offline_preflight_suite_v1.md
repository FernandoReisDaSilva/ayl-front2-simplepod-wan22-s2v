# Wan2.2 S2V Offline Preflight Suite V1

Criado em: `2026-06-26T19:50:08.488351+00:00`

## Escopo

- Bateria local/offline.
- RunPod pausado.
- Sem build/push, sem upload R2, sem alteracoes em LatentSync ou WAN 2.7.

## Fontes

- payload: `/Users/fernandoreisdasilva/Projects/ayl-front2-voice-character-lipsync/logs/wan22_s2v_prompt_payload_debug_v1.json`
- final_report: `/Users/fernandoreisdasilva/Projects/ayl-front2-voice-character-lipsync/logs/wan22_s2v_probe_final_report_v1.json`
- workflow: `nao encontrado`
- object_info: `missing`
- prompt_source: `prompt`

## PASS Checks

| check | detail |
| --- | --- |
| json_loadable | payload_path=/Users/fernandoreisdasilva/Projects/ayl-front2-voice-character-lipsync/logs/wan22_s2v_prompt_payload_debug_v1.json |
| prompt_found | source=prompt nodes=26 |
| class_type_present | missing=0 |
| inputs_dict_present | bad=0 |
| prompt_links_valid | errors=0 |
| known_removed_nodes_absent | errors=0 |
| object_info_structural_literals_raw | errors=0 |
| previously_fixed_rules_raw | errors=0 |
| runtime_sanitize_final | ok |
| runtime_preflight_final | ok |
| final_structural_literals_after_sanitize | suspicious=0 |
| final_torch_compile_disabled_after_sanitize | errors=0 |

## FAIL Checks

| check | detail |
| --- | --- |
| specific_wanvideo_rules_raw | errors=9 |
| torch_compile_disabled_raw | errors=3 |

## WARN Checks

| check | detail |
| --- | --- |
| workflow_original_available | workflow original not found locally |
| object_info_available | object_info not found in payload/final_report; object_info checks are heuristic-limited |

## Nodes Still Suspicious

Nenhum item.

## Proposed Fixes For 0.1.18

- Keep RunPod paused until this suite shows PASS for runtime_sanitize_final, runtime_preflight_final, and final_structural_literals_after_sanitize.
- Use sanitize_wanvideo_structural_literals to neutralize WanVideoSampler.samples=0, WanVideoAddS2VEmbeds.pose_latent=1, and any remaining WanVideo structural literal.
- Force WanVideoSampler.batched_cfg to a real boolean when the workflow export provides -1.
- Disable WanVideoTorchCompileSettings and remove compile_args links so Torch Dynamo/Inductor does not run without a C compiler.
- Preserve scalar allowlist only for known scalar controls such as width, height, num_frames, seed, steps, cfg, shift, scheduler, and timing/audio scale controls.
- When object_info becomes available locally, rerun this suite with exact ComfyUI type validation before tagging 0.1.18.

## Allowlist De Literais Aceitos

| input |
| --- |
| WanVideoAddS2VEmbeds.audio_scale |
| WanVideoAddS2VEmbeds.pose_end_percent |
| WanVideoAddS2VEmbeds.pose_start_percent |
| WanVideoEmptyEmbeds.height |
| WanVideoEmptyEmbeds.num_frames |
| WanVideoEmptyEmbeds.width |
| WanVideoModelLoader.attention_mode |
| WanVideoModelLoader.base_precision |
| WanVideoModelLoader.model |
| WanVideoModelLoader.quantization |
| WanVideoSampler.cfg |
| WanVideoSampler.denoise_strength |
| WanVideoSampler.riflex_freq_index |
| WanVideoSampler.scheduler |
| WanVideoSampler.seed |
| WanVideoSampler.shift |
| WanVideoSampler.steps |

## Sanitized Runtime Reports

```json
{
  "sanitize": {
    "prompt_sanitize_status": "ok",
    "prompt_sanitize_changes": [
      {
        "node_id": "27",
        "class_type": "WanVideoSampler",
        "input_name": "batched_cfg",
        "old_value": -1,
        "new_value": false,
        "reason": "force boolean batched_cfg after UI widget alignment"
      },
      {
        "node_id": "37",
        "class_type": "WanVideoEmptyEmbeds",
        "input_name": "control_embeds",
        "old_value": 832,
        "new_value": null,
        "reason": "set invalid literal control_embeds to None for V1 minimum probe"
      },
      {
        "node_id": "37",
        "class_type": "WanVideoEmptyEmbeds",
        "input_name": "extra_latents",
        "old_value": 480,
        "new_value": null,
        "reason": "set invalid literal extra_latents to None for V1 minimum probe"
      },
      {
        "node_id": "72",
        "class_type": "WanVideoEncode",
        "input_name": "latent_strength",
        "old_value": 1,
        "new_value": null,
        "reason": "set invalid WanVideo structural literal to None for V1 minimum probe"
      },
      {
        "node_id": "27",
        "class_type": "WanVideoSampler",
        "input_name": "samples",
        "old_value": 0,
        "new_value": null,
        "reason": "set invalid WanVideo structural literal to None for V1 minimum probe"
      },
      {
        "node_id": "27",
        "class_type": "WanVideoSampler",
        "input_name": "feta_args",
        "old_value": false,
        "new_value": null,
        "reason": "set invalid WanVideo structural literal to None for V1 minimum probe"
      },
      {
        "node_id": "27",
        "class_type": "WanVideoSampler",
        "input_name": "cache_args",
        "old_value": "comfy",
        "new_value": null,
        "reason": "set invalid WanVideo structural literal to None for V1 minimum probe"
      },
      {
        "node_id": "27",
        "class_type": "WanVideoSampler",
        "input_name": "flowedit_args",
        "old_value": 0,
        "new_value": null,
        "reason": "set invalid WanVideo structural literal to None for V1 minimum probe"
      },
      {
        "node_id": "27",
        "class_type": "WanVideoSampler",
        "input_name": "slg_args",
        "old_value": false,
        "new_value": null,
        "reason": "set invalid WanVideo structural literal to None for V1 minimum probe"
      },
      {
        "node_id": "101",
        "class_type": "WanVideoAddS2VEmbeds",
        "input_name": "pose_latent",
        "old_value": 1,
        "new_value": null,
        "reason": "set invalid WanVideo structural literal to None for V1 minimum probe"
      },
      {
        "node_id": "35",
        "class_type": "WanVideoTorchCompileSettings",
        "input_name": "backend",
        "old_value": "inductor",
        "new_value": "eager",
        "reason": "disable WanVideo torch compile / inductor for V1 minimum probe"
      },
      {
        "node_id": "35",
        "class_type": "WanVideoTorchCompileSettings",
        "input_name": "compile_transformer_blocks_only",
        "old_value": true,
        "new_value": false,
        "reason": "disable WanVideo torch compile / inductor for V1 minimum probe"
      },
      {
        "node_id": "22",
        "class_type": "WanVideoModelLoader",
        "input_name": "compile_args",
        "old_value": [
          "35",
          0
        ],
        "new_value": null,
        "reason": "neutralize link to WanVideoTorchCompileSettings for V1 minimum probe"
      },
      {
        "node_id": "35",
        "class_type": "WanVideoTorchCompileSettings",
        "input_name": "__node__",
        "old_value": {
          "class_type": "WanVideoTorchCompileSettings",
          "inputs": {
            "backend": "eager",
            "fullgraph": false,
            "mode": "default",
            "dynamic": false,
            "dynamo_cache_size_limit": 64,
            "compile_transformer_blocks_only": false,
            "dynamo_recompile_limit": 128
          }
        },
        "new_value": "<removed>",
        "reason": "remove WanVideoTorchCompileSettings node for V1 minimum probe"
      }
    ],
    "prompt_sanitize_errors": [],
    "prompt_sanitize_remaining_suspect_values": [],
    "image_resize_policy": "image_resize_detected_no_change_needed",
    "image_resize_detected_nodes": [
      {
        "node_id": "74",
        "class_type": "ImageResizeKJv2"
      }
    ],
    "image_resize_sanitize_changes": [],
    "image_resize_remaining_invalid_mask_values": [],
    "image_resize_remaining_invalid_combinations": [],
    "wanvideo_empty_embeds_policy": "payload_control_applied",
    "wanvideo_empty_embeds_detected_nodes": [
      {
        "node_id": "37",
        "class_type": "WanVideoEmptyEmbeds"
      }
    ],
    "wanvideo_empty_embeds_sanitize_changes": [
      {
        "node_id": "37",
        "class_type": "WanVideoEmptyEmbeds",
        "input_name": "control_embeds",
        "old_value": 832,
        "new_value": null,
        "reason": "set invalid literal control_embeds to None for V1 minimum probe"
      },
      {
        "node_id": "37",
        "class_type": "WanVideoEmptyEmbeds",
        "input_name": "extra_latents",
        "old_value": 480,
        "new_value": null,
        "reason": "set invalid literal extra_latents to None for V1 minimum probe"
      }
    ],
    "wanvideo_empty_embeds_remaining_invalid_values": [],
    "wanvideo_structural_literal_policy": "payload_control_applied",
    "wanvideo_structural_literal_detected": [
      {
        "node_id": "72",
        "class_type": "WanVideoEncode",
        "input_name": "latent_strength",
        "value": 1,
        "value_type": "int",
        "object_info_spec": null,
        "reason": "wanvideo_structural_literal_error"
      },
      {
        "node_id": "27",
        "class_type": "WanVideoSampler",
        "input_name": "samples",
        "value": 0,
        "value_type": "int"
```
