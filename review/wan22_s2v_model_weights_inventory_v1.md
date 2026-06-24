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
| required_confirmed_rename_only | `checkpoints/wan22_s2v/comfyui_models/text_encoders/wav2vec_xlsr_53_english_fp32.safetensors` | `/opt/ComfyUI/models/text_encoders/wav2vec_xlsr_53_english_fp32.safetensors` | `Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/model.safetensors` | Workflow node `65`, `AudioEncoderLoader`; fonte é safetensors, então V1 exige copy/rename para o nome ComfyUI, não conversão de formato. |
| remove_from_v1_clean_audio_optional_original_workflow | `checkpoints/wan22_s2v/comfyui_models/audio_encoders/MelBandRoFormer/MelBandRoformer_fp16.safetensors` | `/opt/ComfyUI/models/audio_encoders/MelBandRoFormer/MelBandRoformer_fp16.safetensors` | não usado no V1 mínimo | Workflow original usa nodes `81`, `82`, `98` para separação/normalização de vocais. Como `mae_audio_5s.wav` já é fala isolada, V1 deve bypassar esse ramo. |
| optional_or_workflow_aux | `checkpoints/wan22_s2v/comfyui_models/loras/WanVideo/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors` | `/opt/ComfyUI/models/loras/WanVideo/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors` | `Kijai/WanVideo_comfy/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors` | O workflow tem widget de LoRA, mas o caminho parece desconectado no grafo inspecionado; manter como auxiliar. |
| remove_from_v1_interpolation_optional | `checkpoints/wan22_s2v/comfyui_models/upscale_models/gimmvfi_r_arb_lpips_fp32.safetensors` | `/opt/ComfyUI/models/upscale_models/gimmvfi_r_arb_lpips_fp32.safetensors` | não usado no V1 mínimo | Workflow original usa nodes `95`, `96`, `102`, `30` para interpolação e combine final. V1 deve usar o combine não interpolado do node `97`. |

## Resolução Das Pendências

### 1. wav2vec_xlsr_53_english_fp32.safetensors

Status: resolvido como `required_confirmed_rename_only`.

Fonte:

```text
Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/model.safetensors
```

Destino ComfyUI:

```text
/opt/ComfyUI/models/text_encoders/wav2vec_xlsr_53_english_fp32.safetensors
```

R2 key:

```text
checkpoints/wan22_s2v/comfyui_models/text_encoders/wav2vec_xlsr_53_english_fp32.safetensors
```

Comando documentado para preparo local, **não executar ainda**:

```bash
huggingface-cli download Wan-AI/Wan2.2-S2V-14B \
  wav2vec2-large-xlsr-53-english/model.safetensors \
  --local-dir data/checkpoints/wan22_s2v/raw

mkdir -p data/checkpoints/wan22_s2v/comfyui_models/text_encoders

cp data/checkpoints/wan22_s2v/raw/wav2vec2-large-xlsr-53-english/model.safetensors \
  data/checkpoints/wan22_s2v/comfyui_models/text_encoders/wav2vec_xlsr_53_english_fp32.safetensors
```

### 2. MelBandRoFormerModelLoader

Status: removível do V1 mínimo.

No workflow original, o caminho é:

```text
VHS_LoadAudio node 94
  -> MelBandRoFormerSampler node 82
  -> NormalizeAudioLoudness node 98
  -> AudioEncoderEncode node 64
```

Para o probe V1, `mae_audio_5s.wav` é fala isolada. A recomendação é remover/bypassar vocal separation:

```text
VHS_LoadAudio node 94
  -> AudioEncoderEncode node 64
```

Assim, `MelBandRoFormerModelLoader`, `MelBandRoFormerSampler` e o peso `MelBandRoformer_fp16.safetensors` deixam de ser obrigatórios para o V1.

### 3. GIMMVFI

Status: removível do V1 mínimo.

No workflow original, GIMMVFI é interpolação, não geração S2V principal:

```text
DownloadAndLoadGIMMVFIModel node 95
GIMMVFI_interpolate node 96
VHS_SelectEveryNthImage node 102
VHS_VideoCombine node 30
```

O workflow também possui `VHS_VideoCombine` node `97`, ligado ao stream não interpolado. A recomendação para V1 é salvar o output do node `97` e não usar GIMMVFI.

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

## Próxima Alteração Recomendada No Runtime

Antes de build/push, ajustar o runtime/template do workflow V1 para:

- conectar node `94` diretamente ao input `audio` do node `64`;
- desativar/remover nodes `81`, `82`, `98`;
- usar `VHS_VideoCombine` node `97` como output V1;
- desativar/remover nodes `95`, `96`, `102`, `30`.

Depois disso, os pesos bloqueadores do V1 mínimo ficam:

```text
Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors
Wan2_1_VAE_bf16.safetensors
umt5-xxl-enc-bf16.safetensors
wav2vec_xlsr_53_english_fp32.safetensors
```

## Fontes

- https://github.com/kijai/ComfyUI-WanVideoWrapper
- https://github.com/kijai/ComfyUI-WanVideoWrapper/tree/main/s2v
- https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/tree/main
- https://huggingface.co/Kijai/WanVideo_comfy/tree/main
- https://huggingface.co/Wan-AI/Wan2.2-S2V-14B/tree/main
