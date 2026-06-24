# Wan2.2 S2V Model Weights Inventory V1

## Objetivo

Mapear quais arquivos precisam existir em R2 sob:

```text
checkpoints/wan22_s2v/comfyui_models/
```

para que o workflow:

```text
wanvideo2_2_S2V_context_window_testing.json
```

tenha chance de rodar no container:

```text
docker/wan22-s2v-runpod-v1
```

Nenhum peso foi baixado, enviado ou alterado nesta etapa.

## Espelhamento ComfyUI

O prefixo R2 deve espelhar diretamente:

```text
/opt/ComfyUI/models/
```

Exemplo:

```text
R2:
checkpoints/wan22_s2v/comfyui_models/diffusion_models/WanVideo/S2V/model.safetensors

Container:
/opt/ComfyUI/models/diffusion_models/WanVideo/S2V/model.safetensors
```

## Inventário

| Status | R2 key | Container path | Fonte provável | Observação |
|---|---|---|---|---|
| required_confirmed | `checkpoints/wan22_s2v/comfyui_models/diffusion_models/WanVideo/S2V/Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors` | `/opt/ComfyUI/models/diffusion_models/WanVideo/S2V/Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors` | `Kijai/WanVideo_comfy_fp8_scaled/S2V/Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors` | Workflow node `22`, `WanVideoModelLoader`. |
| required_confirmed | `checkpoints/wan22_s2v/comfyui_models/vae/wanvideo/Wan2_1_VAE_bf16.safetensors` | `/opt/ComfyUI/models/vae/wanvideo/Wan2_1_VAE_bf16.safetensors` | `Kijai/WanVideo_comfy/Wan2_1_VAE_bf16.safetensors` | Workflow node `38`, `WanVideoVAELoader`. |
| required_confirmed | `checkpoints/wan22_s2v/comfyui_models/text_encoders/umt5-xxl-enc-bf16.safetensors` | `/opt/ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors` | `Kijai/WanVideo_comfy/umt5-xxl-enc-bf16.safetensors` | Workflow node `67`, `WanVideoTextEncodeCached`. |
| required_workflow_unresolved_source | `checkpoints/wan22_s2v/comfyui_models/text_encoders/wav2vec_xlsr_53_english_fp32.safetensors` | `/opt/ComfyUI/models/text_encoders/wav2vec_xlsr_53_english_fp32.safetensors` | `Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/model.safetensors` | Workflow node `65`, `AudioEncoderLoader`; precisa confirmar conversão/rename para o filename ComfyUI. |
| required_workflow_unresolved_loader | `checkpoints/wan22_s2v/comfyui_models/audio_encoders/MelBandRoFormer/MelBandRoformer_fp16.safetensors` | `/opt/ComfyUI/models/audio_encoders/MelBandRoFormer/MelBandRoformer_fp16.safetensors` | a confirmar | Workflow node `81`, `MelBandRoFormerModelLoader`; loader não foi localizado no custom-node set atual do Docker. |
| optional_or_workflow_aux | `checkpoints/wan22_s2v/comfyui_models/loras/WanVideo/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors` | `/opt/ComfyUI/models/loras/WanVideo/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors` | `Kijai/WanVideo_comfy/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors` | O workflow tem widget de LoRA, mas o caminho parece desconectado no grafo inspecionado; manter como auxiliar. |
| optional_or_workflow_aux | `checkpoints/wan22_s2v/comfyui_models/upscale_models/gimmvfi_r_arb_lpips_fp32.safetensors` | `/opt/ComfyUI/models/upscale_models/gimmvfi_r_arb_lpips_fp32.safetensors` | a confirmar | Workflow node `95`, `DownloadAndLoadGIMMVFIModel`; node sugere auto-download, mas V1 deve evitar download em runtime. |

## Tamanhos

Os tamanhos esperados ainda não foram fixados no inventário porque as consultas públicas usadas retornaram nomes, mas não `Content-Length` confiável para os arquivos Xet/LFS.

O script registra `expected_size_bytes=null` para não inventar valores. Quando os arquivos forem preparados localmente, o próximo passo deve registrar `size_bytes` e `sha256` antes do upload ao R2.

## Script

Criado:

```text
scripts/r2/temp_check_wan22_s2v_model_weights_v1.py
```

Dry-run:

```bash
python3 scripts/r2/temp_check_wan22_s2v_model_weights_v1.py
```

HEAD real no R2, sem download/upload/delete:

```bash
python3 scripts/r2/temp_check_wan22_s2v_model_weights_v1.py --execute
```

## Pendências Antes Do Teste Pago

- Confirmar o custom node que fornece `MelBandRoFormerModelLoader`.
- Confirmar o custom node que fornece `DownloadAndLoadGIMMVFIModel`, ou remover essa etapa do workflow se não for essencial para o primeiro probe.
- Confirmar se `wav2vec_xlsr_53_english_fp32.safetensors` é conversão direta do `Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/model.safetensors`.
- Preparar script separado de download/preparo local com size e sha256.
- Criar uploader específico dos pesos somente depois dessas confirmações.

## Fontes

- https://github.com/kijai/ComfyUI-WanVideoWrapper
- https://github.com/kijai/ComfyUI-WanVideoWrapper/tree/main/s2v
- https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/tree/main
- https://huggingface.co/Kijai/WanVideo_comfy/tree/main
- https://huggingface.co/Wan-AI/Wan2.2-S2V-14B/tree/main
