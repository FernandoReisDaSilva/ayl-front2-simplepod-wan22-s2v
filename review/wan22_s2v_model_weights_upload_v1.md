# Wan2.2 S2V Model Weights Upload V1

## Objetivo

Criar um uploader controlado para os quatro pesos mínimos do Wan2.2 S2V V1 em Cloudflare R2, sem baixar pesos, sem RunPod, sem build/push, sem LatentSync e sem alterar o scaffold WAN 2.7.

## Escopo V1

R2 prefix:

```text
checkpoints/wan22_s2v/comfyui_models/
```

Keys esperadas:

```text
checkpoints/wan22_s2v/comfyui_models/diffusion_models/WanVideo/S2V/Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors
checkpoints/wan22_s2v/comfyui_models/vae/wanvideo/Wan2_1_VAE_bf16.safetensors
checkpoints/wan22_s2v/comfyui_models/text_encoders/umt5-xxl-enc-bf16.safetensors
checkpoints/wan22_s2v/comfyui_models/audio_encoders/wav2vec_xlsr_53_english_fp32.safetensors
```

Essas keys espelham o destino dentro do container:

```text
/opt/ComfyUI/models/
```

## Script

Criado:

```text
scripts/r2/temp_upload_wan22_s2v_model_weights_v1.py
```

Comportamento:

- dry-run por padrão;
- upload real somente com `--execute --confirm-upload`;
- seletores `--only-transformer`, `--only-vae`, `--only-umt5`, `--only-wav2vec`, `--only-all`;
- `--overwrite` para substituir objetos remotos já existentes;
- dry-run relata existência e tamanho local, com exit `0` mesmo quando os pesos ainda não estão preparados;
- upload real valida existência e tamanho local antes de qualquer envio e bloqueia com exit não-zero se algo estiver ausente;
- depois do upload, executa `HEAD` real e compara `ContentLength` com o tamanho local;
- grava log incremental em `logs/r2_wan22_s2v_model_weights_upload_v1_log.json`.

## Paths Locais Padrão

O script assume pesos já preparados localmente em:

```text
data/checkpoints/wan22_s2v/comfyui_models/
```

Ele não baixa, não converte e não copia pesos.

## Comandos

Dry-run:

```bash
python3 scripts/r2/temp_upload_wan22_s2v_model_weights_v1.py
```

Upload real de todos os pesos:

```bash
python3 scripts/r2/temp_upload_wan22_s2v_model_weights_v1.py --execute --confirm-upload
```

Upload real de um peso:

```bash
python3 scripts/r2/temp_upload_wan22_s2v_model_weights_v1.py --only-wav2vec --execute --confirm-upload
```

Sobrescrever objeto existente:

```bash
python3 scripts/r2/temp_upload_wan22_s2v_model_weights_v1.py --execute --confirm-upload --overwrite
```

HEAD real pós-upload:

```bash
python3 scripts/r2/temp_check_wan22_s2v_model_weights_v1.py --execute
```

## Validações

```bash
python3 -m py_compile scripts/r2/temp_upload_wan22_s2v_model_weights_v1.py
python3 scripts/r2/temp_upload_wan22_s2v_model_weights_v1.py
git diff --check
```
