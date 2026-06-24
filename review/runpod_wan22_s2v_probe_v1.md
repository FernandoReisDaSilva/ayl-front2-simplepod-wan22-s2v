# RunPod Wan2.2 S2V Probe V1

## Objetivo

Criar um probe separado de LatentSync e WAN 2.7 para validar Wan2.2-S2V-14B via ComfyUI headless/API em RunPod Community, usando R2 para inputs, outputs e reports.

O alvo do primeiro teste é gerar um MP4 480p curto da Maé a partir de:

```text
mae_reference.png
mae_audio_5s.wav
```

## Arquivos Criados

```text
docker/wan22-s2v-runpod-v1/Dockerfile
docker/wan22-s2v-runpod-v1/entrypoint.sh
docker/wan22-s2v-runpod-v1/runtime_probe.py
scripts/r2/temp_upload_wan22_s2v_probe_inputs_v1.py
scripts/r2/temp_check_wan22_s2v_probe_inputs_v1.py
scripts/runpod/temp_test_runpod_wan22_s2v_probe_v1.py
review/runpod_wan22_s2v_probe_v1.md
```

## Fonte Técnica Escolhida

Implementação ComfyUI:

```text
kijai/ComfyUI-WanVideoWrapper
```

Workflow real usado como base:

```text
s2v/wanvideo2_2_S2V_context_window_testing.json
```

Nós S2V confirmados no workflow real:

```text
LoadImage
VHS_LoadAudio
WanVideoModelLoader
WanVideoVAELoader
WanVideoTextEncodeCached
WanVideoSampler
WanVideoAddS2VEmbeds
VHS_VideoCombine
```

## Controles Confirmados

No `WanVideoSampler`, os campos existem no código do node:

```text
steps
cfg
shift
seed
denoise_strength
```

No `WanVideoAddS2VEmbeds`, os campos existem no código do node:

```text
audio_scale
pose_start_percent
pose_end_percent
```

O runtime valida `/object_info` do ComfyUI antes de executar. Se um node ou campo esperado não existir na versão instalada, o probe falha explicitamente em `final_report.json`.

## Node IDs Validados no Workflow

```text
73  LoadImage
94  VHS_LoadAudio
27  WanVideoSampler
101 WanVideoAddS2VEmbeds
30  VHS_VideoCombine
97  VHS_VideoCombine
```

## Imagem Planejada

```text
ghcr.io/fernandoreisdasilva/ayl-wan22-s2v-runpod:0.1.0
```

Esta imagem não é LatentSync e não é WAN 2.7.

## Pesos

Pesos não entram no Git.

Decisão V1:

```text
R2 prefix: checkpoints/wan22_s2v/comfyui_models/
Container: /opt/ComfyUI/models/
```

O prefixo deve espelhar a estrutura esperada pelo ComfyUI, por exemplo:

```text
checkpoints/wan22_s2v/comfyui_models/diffusion_models/WanVideo/S2V/Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors
checkpoints/wan22_s2v/comfyui_models/vae/wanvideo/Wan2_1_VAE_bf16.safetensors
checkpoints/wan22_s2v/comfyui_models/text_encoders/umt5-xxl-enc-bf16.safetensors
checkpoints/wan22_s2v/comfyui_models/audio_encoders/wav2vec_xlsr_53_english_fp32.safetensors
```

A estrutura exata precisa ser validada contra o ComfyUI instalado antes do teste pago.

## R2 Keys

```text
tests/runpod_wan22_s2v_probe_v1/input/mae_reference.png
tests/runpod_wan22_s2v_probe_v1/input/mae_audio_5s.wav
tests/runpod_wan22_s2v_probe_v1/output/video_out.mp4
tests/runpod_wan22_s2v_probe_v1/output/final_report.json
tests/runpod_wan22_s2v_probe_v1/progress/container_started.json
```

## Runtime

Modo:

```text
AYL_RUN_MODE=wan22_s2v_probe
```

Fases:

- escreve `container_started` no R2;
- valida torch/CUDA/GPU;
- valida ffmpeg;
- baixa inputs do R2;
- baixa prefixo de modelos do R2 para `/opt/ComfyUI/models`;
- inicia ComfyUI headless;
- valida nodes reais via `/object_info`;
- converte o workflow UI real para prompt API;
- aplica paths e controles;
- executa `/prompt`;
- localiza MP4 gerado;
- sobe `video_out.mp4`;
- escreve `final_report.json`.

## Comandos Dry-Run

```bash
python3 scripts/r2/temp_upload_wan22_s2v_probe_inputs_v1.py
python3 scripts/r2/temp_check_wan22_s2v_probe_inputs_v1.py
python3 scripts/runpod/temp_test_runpod_wan22_s2v_probe_v1.py
```

## Upload Real Futuro

NÃO executar ainda:

```bash
python3 scripts/r2/temp_upload_wan22_s2v_probe_inputs_v1.py --execute --confirm-upload --overwrite
```

## Comando Pago Futuro

NÃO executar ainda:

```bash
python3 scripts/runpod/temp_test_runpod_wan22_s2v_probe_v1.py --execute --confirm-cost-risk
```

## Critério Técnico

O probe técnico só passa se:

- `pod_created=true`;
- `r2_progress_detected=true`;
- `r2_final_detected=true`;
- `final_report_verified=true`;
- `pod_terminated=true`;
- `manual_cleanup_required=false`;
- `runtime_probe_status=ok`;
- `output_upload_status=ok`;
- `video_out.mp4` existir no R2.

## Critério Editorial

O teste editorial só passa depois de revisão humana ou métrica externa:

```text
phonetic_mouth_score >= 8
head_motion_artifact <= aceitável
```

O runtime registra `editorial_gate.status=manual_review_required`.

## Riscos

- RTX 3090 pode ser insuficiente para Wan2.2-S2V-14B, mesmo com fp8 e offload.
- O workflow real é WIP e pode mudar com updates do wrapper.
- O conversor UI workflow -> API prompt pode exigir ajuste se ComfyUI alterar o formato.
- Os pesos em R2 precisam espelhar exatamente os nomes esperados pelos loaders.
- `VHS_VideoCombine` pode salvar em subpasta diferente; o runtime procura no history e em `/opt/ComfyUI/output/**/*.mp4`.

## Fontes

- https://github.com/kijai/ComfyUI-WanVideoWrapper
- https://github.com/kijai/ComfyUI-WanVideoWrapper/tree/main/s2v
- https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/main/s2v/nodes.py
- https://github.com/kijai/ComfyUI-WanVideoWrapper/blob/main/nodes_sampler.py
- https://huggingface.co/Wan-AI/Wan2.2-S2V-14B
