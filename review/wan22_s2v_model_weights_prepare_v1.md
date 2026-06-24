# Wan2.2 S2V Model Weights Prepare V1

## Objetivo

Preparar localmente os quatro pesos mínimos do Wan2.2 S2V V1 usados pelo uploader R2, sem upload automático, sem RunPod e sem build/push.

## Paths Locais Confirmados

O uploader espera os arquivos sob:

```text
data/checkpoints/wan22_s2v/comfyui_models/
```

Paths finais:

```text
data/checkpoints/wan22_s2v/comfyui_models/diffusion_models/WanVideo/S2V/Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors
data/checkpoints/wan22_s2v/comfyui_models/vae/wanvideo/Wan2_1_VAE_bf16.safetensors
data/checkpoints/wan22_s2v/comfyui_models/text_encoders/umt5-xxl-enc-bf16.safetensors
data/checkpoints/wan22_s2v/comfyui_models/audio_encoders/wav2vec_xlsr_53_english_fp32.safetensors
```

O arquivo bruto do wav2vec fica em:

```text
data/checkpoints/wan22_s2v/raw/wav2vec2-large-xlsr-53-english/model.safetensors
```

e o preparador aplica copy/rename para:

```text
data/checkpoints/wan22_s2v/comfyui_models/audio_encoders/wav2vec_xlsr_53_english_fp32.safetensors
```

## Fontes

| Peso | Fonte | Arquivo |
|---|---|---|
| transformer | `Kijai/WanVideo_comfy_fp8_scaled` | `S2V/Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors` |
| VAE | `Kijai/WanVideo_comfy` | `Wan2_1_VAE_bf16.safetensors` |
| UMT5 | `Kijai/WanVideo_comfy` | `umt5-xxl-enc-bf16.safetensors` |
| wav2vec | `Wan-AI/Wan2.2-S2V-14B` | `wav2vec2-large-xlsr-53-english/model.safetensors` |

## Script

Criado:

```text
scripts/models/temp_prepare_wan22_s2v_model_weights_v1.py
```

Comportamento:

- cria os diretórios locais necessários;
- dry-run por padrão;
- download real somente com `--execute --confirm-download`;
- download real usa `hf download`, não `huggingface-cli download`;
- antes do primeiro download real, valida se o comando `hf` existe e falha com pedido claro de instalação/upgrade se estiver ausente;
- lista fonte, destino, existência local e tamanho atual;
- valida tamanho mínimo esperado depois de execução real;
- aplica copy/rename do wav2vec;
- não faz upload R2;
- não executa RunPod;
- não faz build/push.

## Comandos

Dry-run:

```bash
python3 scripts/models/temp_prepare_wan22_s2v_model_weights_v1.py
```

Download real local:

```bash
python3 scripts/models/temp_prepare_wan22_s2v_model_weights_v1.py --execute --confirm-download
```

Dependência para execução real:

```bash
python3 -m pip install --upgrade huggingface_hub
```

Forma de download usada pelo script:

```bash
hf download <repo> <file> --local-dir <dir>
```

## Validações

```bash
python3 -m py_compile scripts/models/temp_prepare_wan22_s2v_model_weights_v1.py
python3 scripts/models/temp_prepare_wan22_s2v_model_weights_v1.py
git diff --check
```
