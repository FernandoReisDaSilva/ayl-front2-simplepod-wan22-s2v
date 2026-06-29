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
ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.1.3
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
GitHub Actions -> Build SimplePod Wan2.2 S2V FastAPI V2 -> Run workflow -> image_tag=0.1.3
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

Status: preparado em dry-run. Execucao real agora deve usar a imagem V2 mais recente publicada e template `25114` apontando para essa tag.

Motivo para nova tag:

- a tag V2 `0.1.0` validou FastAPI, torch/CUDA e Network Drive;
- a tag V2 `0.1.0` nao tinha endpoint interno para download controlado de pesos;
- a tag V2 `0.1.1` adicionou `huggingface_hub[cli]` e `POST /admin/download-wan22-s2v-weights`;
- a tag V2 `0.1.2` corrige travamento provavel do download, adiciona timeout por subprocesso e cria `GET /admin/verify-wan22-s2v-weights`.

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
- retorna JSON final mesmo em erro/timeout do subprocesso.

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

### Correcao de Travamento

Causa provavel: o endpoint anterior chamava `snapshot_download()` diretamente dentro da request HTTP, sem timeout proprio e sem subprocesso isolado. Se o Hub/cache/lock finalizasse o download mas prendesse a chamada, o cliente ficava aguardando indefinidamente e nao gravava report final.

Correcao em `0.1.2`:

- `POST /admin/download-wan22-s2v-weights` usa `subprocess.run()` com timeout explicito;
- captura `returncode`, `stdout` e `stderr` truncados;
- retorna `status=timeout` se o comando exceder o limite;
- sempre calcula inventario do diretorio apos o comando;
- o cliente tenta `GET /admin/verify-wan22-s2v-weights` se o request de download falhar/expirar;
- o cliente grava report final e tenta delete em `finally`.

Status possiveis do gate de download:

- `succeeded`;
- `succeeded_but_client_timeout`;
- `failed_download`;
- `failed_verify_after_download`;
- `timeout`;
- `interrupted_delete_attempted`;
- `delete_failed_manual_required`.

## Wan2.2 S2V Weights Verify Gate

Endpoint:

```text
GET /admin/verify-wan22-s2v-weights
```

Guardas:

- exige `AYL_ENABLE_ADMIN_DOWNLOADS=1` ou `AYL_ENABLE_ADMIN_VERIFY=1`;
- nao baixa nada;
- nao roda inferencia;
- nao gera video;
- nao imprime segredos.

Verificacoes:

- `path=/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B`;
- `exists`;
- `is_dir`;
- `recursive_file_count`;
- `recursive_total_size_bytes`;
- `recursive_total_size_gb`;
- principais `.safetensors` com tamanho;
- arquivos de config/tokenizer/model index quando existirem.

Script:

```bash
python3 scripts/simplepod/temp_simplepod_verify_wan22_s2v_weights_v1.py
```

Execucao real futura:

```bash
python3 scripts/simplepod/temp_simplepod_verify_wan22_s2v_weights_v1.py --execute --confirm-start --confirm-verify --confirm-delete
```

Report:

```text
logs/simplepod_wan22_s2v_weights_verify_v1.json
```

## Phase Timing

Utilitario:

```text
scripts/simplepod/simplepod_phase_timing.py
```

Formato compacto de stdout:

```text
[00:00] START phase=market_selection
[00:07] DONE phase=market_selection elapsed=00:07
[02:14] START phase=wait_health
[03:02] DONE phase=wait_health elapsed=00:48
```

Cada report registra por fase:

- `phase_name`;
- `started_at`;
- `ended_at`;
- `elapsed_seconds`;
- `elapsed_hhmmss`;
- `elapsed_mmss`.

## SimplePod GPU Selection Policies

Camada reutilizavel:

```text
scripts/simplepod/simplepod_gpu_policies.py
```

Policies:

- `smoke_gpu_policy`: `rentalStatus=active`, `datacenter=EU-PL-01`, `gpuCount=1`, `order[pricePerGpu]=asc`, pick first.
- `download_gpu_policy`: igual `smoke_gpu_policy`.
- `first_inference_gpu_policy`: `rentalStatus=active`, `datacenter=EU-PL-01`, `gpuCount=1`, `gpuMemorySize>=24000`, `order[pricePerGpu]=asc`, pick first.
- `production_single_job_policy`: `rentalStatus=active`, `datacenter=EU-PL-01`, `gpuCount=1`, `gpuMemorySize>=48000`, prefer `A6000`, `L40S`, `RTX 6000 Ada`, `order[pricePerGpu]=asc`, pick first.
- `production_parallel_policy`: `rentalStatus=active`, `datacenter=EU-PL-01`, `gpuCount=1`, `gpuMemorySize>=90000`, prefer `RTX PRO 6000 Blackwell` ou `RTX 6000 Blackwell`, `order[pricePerGpu]=asc`, pick first.
- `premium_batch_policy`: `rentalStatus=active`, `datacenter=EU-PL-01`, require `gpuModel` contendo `H100` ou `H200`, `order[pricePerGpu]=asc`, pick first.

Nenhum script deve hardcodar um market id unico. O market pode ser passado manualmente apenas como override explicito; nesse caso o report marca o motivo como override e nao inventa GPU/preco.

Todo report de selecao deve registrar:

- selected policy;
- selected market id;
- `gpuModel`;
- `gpuMemorySize`;
- `pricePerGpu`;
- datacenter;
- reason selected;
- rejected candidates summary.

## Resolution Policy

- `target_production_resolution`: `1080x1080`.
- `fallback_resolution`: `960x960`.
- smoke/download/check nao definem resolucao editorial.
- `first_inference_gpu_policy` com `gpuMemorySize>=24000`: tentar `1080x1080` para clipe curto Maé 14.8s; se OOM, registrar fallback para `960x960`.
- `production_single_job_policy` com `gpuMemorySize>=48000`: usar `1080x1080` como padrao.
- `production_parallel_policy` com `gpuMemorySize>=90000`: usar `1080x1080` como padrao e permitir paralelismo controlado por fila/job slots.
- Nao fazer upscale `960x960 -> 1080x1080` como padrao se `1080x1080` direto for estavel.

Todo report de runtime/inferencia deve registrar:

- `requested_resolution`;
- `actual_generation_resolution`;
- `fallback_used`;
- `gpu_policy`;
- `gpuModel`;
- `gpuMemorySize`;
- `runtime_seconds`;
- `estimated_cost`;
- `oom_or_error_status`.

## First Maé FR 14.8s 1080 Inference Gate

Plano:

```text
review/simplepod_mae_fr_14_8s_1080_inference_plan_v1.md
```

Imagem alvo:

```text
ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.1.3
```

Motivo da tag `0.1.3`:

- adiciona `POST /jobs/wan22-s2v/run`;
- valida payload Maé FR 14.8s 1080;
- exige `confirm_inference=RUN_WAN22_S2V_MAE_14_8S_1080`;
- ainda nao integra o runner Wan2.2 S2V real;
- nao gera placeholder.

Script:

```bash
python3 scripts/simplepod/temp_simplepod_run_mae_wan22_s2v_14_8s_1080_v1.py
```

Execucao real futura:

```bash
python3 scripts/simplepod/temp_simplepod_run_mae_wan22_s2v_14_8s_1080_v1.py --execute --confirm-start --confirm-inference --confirm-delete
```

Policy:

```text
first_inference_gpu_policy
```

Regras:

- `gpuMemorySize>=24000`;
- `target_resolution=1080x1080`;
- `fallback_resolution=960x960` somente se OOM;
- RTX 3060 nao deve ser usada para inferencia real.

Report:

```text
logs/simplepod_mae_wan22_s2v_14_8s_1080_inference_v1.json
```
