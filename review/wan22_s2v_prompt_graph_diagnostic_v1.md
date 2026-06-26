# Wan2.2 S2V Prompt Graph Diagnostic V1

Criado em: `2026-06-26T12:49:12.924121+00:00`

## Escopo

- Diagnostico local/offline apenas.
- RunPod pausado para o probe Wan2.2 S2V ate consolidar os fixes.
- Sem upload R2, sem build/push, sem download de pesos.
- Sem alteracoes em LatentSync ou WAN 2.7.

## Fontes Locais

- workflow: `nao encontrado`
- payload debug: `/Users/fernandoreisdasilva/Projects/ayl-front2-voice-character-lipsync/logs/wan22_s2v_prompt_payload_debug_v1.json`
- final_report: `/Users/fernandoreisdasilva/Projects/ayl-front2-voice-character-lipsync/logs/wan22_s2v_probe_final_report_v1.json`
- prompt_source: `payload.prompt`

## Estado Do Probe 0.1.13

- contexto informado: `0.1.13 passou missing nodes, MelBand, GIMMVFI, PrimitiveNode, validation, SageAttention, fp16_fast, ImageResize mask e Lanczos GPU`
- novo erro informado: `TypeError: 'int' object is not subscriptable`
- ponto informado: `ComfyUI-WanVideoWrapper/nodes.py line 1438 control_embeds["control_embeds"]`
- interpretacao: `ha desalinhamentos remanescentes de links/inputs no payload API`

## Final Report Local Disponivel

- runtime_probe_status: `video_output_missing`
- output_upload_status: `not_attempted`
- foco local: `0.1.13 reported TypeError in WanVideoAddS2VEmbeds/control_embeds from probe context.`

## Preflight Semantico

- status: `error`
- erros: `['Primitive literal values remain in embed inputs.', 'control_embeds contains a literal int.', 'wanvideo_empty_embeds_invalid_control_embeds']`

### control/embed

| node_id | class_type | input_name | value |
| --- | --- | --- | --- |
| 37 | WanVideoEmptyEmbeds | control_embeds | 832 |

### Inputs S2V Principais

| node_id | class_type | input_name | value_kind | value |
| --- | --- | --- | --- | --- |
| 37 | WanVideoEmptyEmbeds | control_embeds | int | 832 |
| 27 | WanVideoSampler | image_embeds | link | ["101", 0] |

### Valores Suspeitos

| node_id | class_type | input_name | reason | value |
| --- | --- | --- | --- | --- |
| 37 | WanVideoEmptyEmbeds | control_embeds | int_in_embed_dict_or_control_input | 832 |

### Literais Onde Link/Objeto Era Esperado

| node_id | class_type | input_name | reason | value |
| --- | --- | --- | --- | --- |
| 38 | WanVideoVAELoader | model_name | literal_in_link_or_object_like_input | wanvideo/Wan2_1_VAE_bf16.safetensors |
| 73 | LoadImage | image | literal_in_link_or_object_like_input | mae_reference.png |
| 67 | WanVideoTextEncodeCached | model_name | literal_in_link_or_object_like_input | umt5-xxl-enc-bf16.safetensors |
| 72 | WanVideoEncode | latent_strength | literal_in_link_or_object_like_input | 1 |
| 22 | WanVideoModelLoader | model | literal_in_link_or_object_like_input | WanVideo/S2V/Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors |
| 94 | VHS_LoadAudio | audio_file | literal_in_link_or_object_like_input | input/mae_audio_5s.wav |
| 66 | LoadAudio | audio | literal_in_link_or_object_like_input | NieR_ Automata - _Weight of the World_ ENG VER. by Lizz Robinett [CyOSTbel3AM].mp3 |
| 65 | AudioEncoderLoader | audio_encoder_name | literal_in_link_or_object_like_input | wav2vec_xlsr_53_english_fp32.safetensors |
| 37 | WanVideoEmptyEmbeds | control_embeds | literal_in_link_or_object_like_input | 832 |
| 37 | WanVideoEmptyEmbeds | extra_latents | literal_in_link_or_object_like_input | 480 |
| 101 | WanVideoAddS2VEmbeds | audio_scale | literal_in_link_or_object_like_input | 1.0 |
| 101 | WanVideoAddS2VEmbeds | pose_latent | literal_in_link_or_object_like_input | 1 |
| 97 | VHS_VideoCombine | trim_to_audio | literal_in_link_or_object_like_input | true |

### Links Para PrimitiveNode

Nenhum item encontrado.

### PrimitiveNode No Workflow Original

Nenhum item encontrado.

### WanVideo Opcionais Para Revisao

Nenhum item encontrado.

## Fixes Propostos Para Tag 0.1.14

1. `0.1.13` passou pelos ajustes de `ImageResizeKJv2` e chegou ao bloqueio de embed.
2. O payload real confirmou `WanVideoEmptyEmbeds` node `37` com `control_embeds=832` como literal `int`.
3. Decisao V1: no probe minimo, remover `control_embeds` quando o input for opcional; caso contrario, setar `None`.
4. Manter o preflight `preflight_prompt_semantics(prompt, object_info)` antes do payload debug e antes do POST `/prompt`.
5. Falhar localmente quando encontrar `WanVideoEmptyEmbeds.control_embeds` como `int`, `str` ou `bool`, alem dos bloqueios ja existentes para `mask` string, HTML string, `lanczos+gpu`, `base_precision` fast ou `attention_mode` sage.

## Proxima Tag Sugerida

```text
0.1.14
```

## Observacoes

- Workflow original nao foi encontrado nos caminhos locais padrao.
- object_info nao foi fornecido; comparacao exata de tipos/opcoes do ComfyUI ficou heuristica.
- final_report local disponivel pode estar defasado em relacao ao erro 0.1.13 informado nesta tarefa.
