# SimplePod Wan2.2 S2V Runtime V2 Bootstrap Plan

Data: 2026-06-29

## Objetivo

Preparar a imagem/runtime V2 para validar bootstrap de Wan2.2 S2V no SimplePod sem baixar pesos, sem executar inferencia e sem gerar video.

## Contexto Validado

- Template V1 funciona.
- Porta publica foi resolvida via `ports.direct` com `srcPort=8000`.
- FastAPI V1 respondeu `/health`, `/gpu` e `/models`.
- V1 nao tem `torch`.
- `/mnt/ayl_models` nao apareceu montado no primeiro smoke.
- R2 env nao apareceu dentro do container.

## Imagem V2

Imagem alvo:

```text
ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.1.0
```

Docker root:

```text
docker/simplepod-wan22-s2v-fastapi-v2/
```

Base escolhida:

```text
pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
```

Motivo: validar import de `torch` e visibilidade CUDA no SimplePod antes de instalar dependencias maiores de Wan2.2 S2V.

## Incluido

- FastAPI atual.
- Endpoints preservados: `/health`, `/gpu`, `/models`, `POST /jobs/wan22-s2v`.
- `torch`/CUDA via imagem base PyTorch.
- `boto3`.
- `python-dotenv`.

## Fora do Escopo

- baixar pesos;
- carregar Wan2.2 S2V;
- rodar inferencia;
- criar output de video;
- embutir segredos SimplePod/R2 na imagem;
- resolver montagem do Network Drive automaticamente via payload nao documentado.

## Workflow

```text
.github/workflows/build-simplepod-wan22-s2v-fastapi-v2.yml
```

Disparo manual:

```text
GitHub Actions -> Build SimplePod Wan2.2 S2V FastAPI V2 -> Run workflow -> image_tag=0.1.1
```

## Check GHCR

```bash
python3 scripts/simplepod/temp_check_ghcr_image_manifest_v2.py
```

Report:

```text
logs/simplepod_ghcr_image_manifest_v2.json
```

## Runtime Smoke V2

```bash
python3 scripts/simplepod/temp_simplepod_runtime_smoke_v2.py
```

Execucao real futura exige template V2 criado e confirmacoes:

```bash
python3 scripts/simplepod/temp_simplepod_runtime_smoke_v2.py --template-id <TEMPLATE_ID_V2> --execute --confirm-start --confirm-delete
```

Valida:

- `/health`;
- `/gpu` com `torch_import_status=ok`;
- `/models`;
- presenca ou ausencia de `/mnt/ayl_models`;
- presenca ou ausencia de env R2 redigida.

## Criterios de Sucesso do Gate V2

- imagem V2 publicada no GHCR;
- template SimplePod V2 criado em gate separado;
- runtime smoke V2 retorna `/health=200`;
- `/gpu` importa torch;
- report registra se CUDA esta visivel;
- report registra se `/mnt/ayl_models` existe;
- report registra env R2 como booleans/redigido, sem segredos;
- instancia deletada ao final.

## Wan2.2 S2V Weights Download Gate

Status: preparado em dry-run. Execucao real ainda pendente de imagem V2 `0.1.1` publicada e template `25114` apontando para essa tag.

Motivo para nova tag:

- a tag V2 `0.1.0` validou FastAPI, torch/CUDA e Network Drive;
- a tag V2 `0.1.0` nao tinha endpoint interno para download controlado de pesos;
- a tag V2 `0.1.1` adiciona `huggingface_hub[cli]` e `POST /admin/download-wan22-s2v-weights`.

Endpoint novo:

```text
POST /admin/download-wan22-s2v-weights
```

Guardas:

- exige env `AYL_ENABLE_ADMIN_DOWNLOADS=1` no container;
- exige payload `confirm_download=DOWNLOAD_WAN22_S2V_WEIGHTS`;
- baixa somente para `WAN22_S2V_MODEL_DIR`;
- nao executa inferencia;
- nao gera video;
- nao imprime segredos.

Plano de download:

```text
repo_id=Wan-AI/Wan2.2-S2V-14B
target_dir=/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B
hf_home=/mnt/ayl_models/caches/huggingface
```

Comando equivalente dentro do container:

```bash
huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir /mnt/ayl_models/wan2.2/Wan2.2-S2V-14B
```

Script do gate:

```bash
python3 scripts/simplepod/temp_simplepod_download_wan22_s2v_weights_v1.py
```

Execucao real futura:

```bash
python3 scripts/simplepod/temp_simplepod_download_wan22_s2v_weights_v1.py --execute --confirm-start --confirm-download --confirm-delete
```

Report:

```text
logs/simplepod_wan22_s2v_weights_download_v1.json
```
