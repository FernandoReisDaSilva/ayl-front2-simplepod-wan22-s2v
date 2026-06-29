# SimplePod Wan2.2 S2V FastAPI Migration Plan V1

Data: 2026-06-28

## Objetivo

Preparar uma trilha V1 minima, segura e testavel para migrar o AYL Front 2 de RunPod Community Pods + ComfyUI/WanVideoWrapper para SimplePod + Network Drive persistente + FastAPI propria para inferencia direta Wan2.2 S2V.

Esta proposta nao executa custo, nao baixa pesos, nao cria SimplePod, nao faz build/push e nao altera o pipeline Wan2.2 S2V atual.

## Contexto Atual

### Arquitetura Atual: RunPod / ComfyUI / R2

Fluxo atual do probe Wan2.2 S2V:

1. `scripts/runpod/temp_test_runpod_wan22_s2v_probe_v1.py` cria Pod RunPod Community via GraphQL.
2. O container `docker/wan22-s2v-runpod-v1/` inicia ComfyUI headless.
3. O runtime baixa inputs do Cloudflare R2:
   - imagem de referencia;
   - audio;
   - pesos/modelos, quando necessario pelo fluxo.
4. O runtime converte workflow UI para prompt API ComfyUI.
5. O runtime aplica filtros/bypasses/sanitizers para lidar com:
   - nos decorativos;
   - `PrimitiveNode`;
   - MelBandRoFormer;
   - GIMMVFI;
   - valores desalinhados no payload;
   - Torch compile/Inductor;
   - `ImageResizeKJv2`;
   - literais estruturais em `WanVideo*`.
6. O runtime chama `POST /prompt` no ComfyUI.
7. O runtime aguarda history, localiza MP4, sobe output e `final_report.json` para R2.
8. O script local faz polling de R2 e termina o Pod.

### Problemas observados

- Cold start caro e lento.
- RunPod Community Pods variam em disponibilidade e estabilidade.
- R2 usado indevidamente como cache de pesos grandes.
- ComfyUI/WanVideoWrapper introduz instabilidade por:
  - nos customizados ausentes;
  - workflow UI exportado com campos desalinhados;
  - mudancas em custom nodes;
  - necessidade de sanitizacao defensiva extensa antes do `/prompt`.
- Debug fica indireto: workflow -> prompt ComfyUI -> history -> report.
- Cada novo erro custa uma rodada de Pod.

## Arquitetura Proposta

### SimplePod / Network Drive / FastAPI / R2 I/O

Fluxo proposto V1:

1. Criar uma imagem/container controlado `simplepod-wan22-s2v-fastapi-v1`.
2. Montar Network Drive persistente em caminho fixo: `/mnt/ayl_models`.
3. Pesos e caches ficam persistentes no Network Drive, nao em R2.
4. Container sobe uma API FastAPI propria.
5. A API recebe um job S2V com R2 keys de input e parametros de inferencia.
6. Worker baixa apenas inputs leves do R2 para disco local temporario.
7. Worker carrega modelos diretamente do Network Drive.
8. Worker executa inferencia direta Wan2.2 S2V, sem ComfyUI e sem WanVideoWrapper.
9. Worker salva MP4 local, sobe output para R2 e publica `final_report.json`.
10. Script local apenas chama FastAPI, acompanha status e baixa report/output se necessario.

### Principios V1

- Network Drive e a fonte unica de pesos/cache.
- R2 e somente para input/output/report.
- FastAPI deve ter uma superficie pequena e auditavel.
- Nada de workflow UI, ComfyUI custom nodes ou `/prompt`.
- Relatorios devem preservar o contrato mental atual de `final_report.json`.
- A primeira versao deve privilegiar reproducibilidade e diagnostico, nao performance maxima.

## Estrutura Proposta no Repo

Nao implementar ainda. Estrutura sugerida:

```text
docker/simplepod-wan22-s2v-fastapi-v1/
  Dockerfile
  entrypoint.sh
  app/
    main.py
    config.py
    r2_client.py
    reports.py
    redact.py
    schemas.py
    wan22_s2v_infer.py
    model_paths.py
    health.py
  README.md

scripts/simplepod/
  temp_check_simplepod_wan22_s2v_fastapi_health_v1.py
  temp_submit_simplepod_wan22_s2v_job_v1.py
  temp_poll_simplepod_wan22_s2v_job_v1.py
  temp_download_simplepod_wan22_s2v_outputs_v1.py
  temp_prepare_simplepod_network_drive_manifest_v1.py

scripts/r2/
  temp_upload_simplepod_wan22_s2v_inputs_v1.py
  temp_download_simplepod_wan22_s2v_final_report_v1.py
  temp_download_simplepod_wan22_s2v_output_video_v1.py

review/
  simplepod_wan22_s2v_fastapi_migration_plan_v1.md
  simplepod_wan22_s2v_network_drive_inventory_v1.md
  simplepod_wan22_s2v_fastapi_probe_v1.md
```

## Network Drive

### Mount

Padrao operacional V1:

```text
SIMPLEPOD_VOLUME_NAME=ayl_models_wan22_s2v_v1
SIMPLEPOD_DATACENTER=EU-PL-01
SIMPLEPOD_MODELS_ROOT=/mnt/ayl_models
WAN22_S2V_MODEL_DIR=/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B
```

Env configuravel:

```text
SIMPLEPOD_MODELS_ROOT=/mnt/ayl_models
WAN22_S2V_MODEL_DIR=/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B
```

### Volume persistente

Status registrado em 2026-06-28:

- `SIMPLEPOD_VOLUME_NAME`: `ayl_models_wan22_s2v_v1`
- `SIMPLEPOD_DATACENTER`: `EU-PL-01`
- tamanho: 100GB
- status: criado e ativo
- finalidade: armazenar pesos/cache Wan2.2 S2V fora do Git e fora do R2

### Estrutura

```text
/mnt/ayl_models/
  manifests/
    wan22_s2v_minimal_v1_manifest.json
    wan22_s2v_minimal_v1_checksums.json
  wan2.2/
    Wan2.2-S2V-14B/
      config.json
      model_index.json
      diffusion_pytorch_model.safetensors.index.json
      shards/
  caches/
    huggingface/
    modelscope/
    torch/
    triton/
  tmp/
```

### Regra V1

Manter dois layouts possiveis:

- `comfyui_compat`: reaproveita os pesos ja validados para ComfyUI/Kijai.
- `native`: reservado para inference direta usando codigo oficial/diffusers.

A V1 deve escolher explicitamente um deles no manifest. Nao misturar automaticamente.

## Programmatic Control Gate

Status registrado em 2026-06-28 via `scripts/simplepod/temp_probe_simplepod_api_v1.py`.

### Volume

- `SIMPLEPOD_VOLUME_NAME`: `ayl_models_wan22_s2v_v1`
- `SIMPLEPOD_DATACENTER`: `EU-PL-01`
- tamanho: 100GB
- status: criado e ativo
- mount esperado: `/mnt/ayl_models`
- modelo esperado: `/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B`

### API key

- env esperada: `SIMPLEPOD_API_KEY`
- status atual do probe: presente no `.env`/ambiente local
- metodo de autenticacao documentado: API key no header `X-AUTH-TOKEN`
- `SIMPLEPOD_API_LOGIN`/`SIMPLEPOD_API_PASSWORD`: nomes presentes no `.env`, valores locais vazios no momento do probe; nao usados pelo probe porque a documentacao REST consultada define `apiKey` via header, nao Basic auth
- valor da chave: nunca imprimir, nunca salvar em report

### Base URL e endpoints read-only

- env de base URL esperada: `SIMPLEPOD_API_BASE_URL`
- base URL configurada: `https://api.simplepod.ai`
- documentacao REST usada: `https://api.simplepod.ai/docs`
- endpoints read-only testados:
  - `GET /instances/list`: HTTP `200`
  - `GET /instances/global-templates/list`: HTTP `200`
  - `GET /instances/templates/list`: HTTP `200`
  - `GET /instances/summary`: HTTP `200`
- status do probe: `succeeded`
- resultado: chamadas read-only chegaram na API usando o header documentado `X-AUTH-TOKEN`
- templates globais observados: 30 itens reportados; amostras seguras incluem `pytorch`, `TensorFlow GPU`, `vllm-openai Phi-3-small-8k-instruct`, `ubuntu nvidia cuda 12.8.1 for RTX5000/6000 series`, `ollama/ollama` e `stable-diffusion:web-ui`
- templates privados observados: 0 itens
- instancias listadas: 0 itens
- endpoint de volumes/storage: nao foi chamado porque nenhum path REST read-only correspondente foi confirmado na documentacao consultada
- report local: `logs/simplepod_api_probe_v1.json`

### Pendencias para start/stop programatico

1. Confirmar endpoint REST documentado para volumes/storage, se existir em versao posterior da documentacao.
2. Escolher template base para o futuro container FastAPI, provavelmente via imagem propria em vez de template oficial generico.
3. Somente depois criar scripts separados para start/stop com guards explicitos de custo.

### Proximo gate recomendado

Gate recomendado: desenhar dry-run de configuracao de instancia/template SimplePod sem criar pod, mantendo guards explicitos contra custo.

## Private Template Dry-Run Gate

Status registrado em 2026-06-28 via `scripts/simplepod/temp_prepare_simplepod_template_create_dryrun_v1.py`.

### Endpoints identificados

- criar template privado: `POST /instances/templates`
- listar templates privados: `GET /instances/templates/list`
- autenticacao documentada: API key no header `X-AUTH-TOKEN`
- endpoint de escrita chamado: nao
- template real criado: nao

### Campos documentados para template privado

O OpenAPI em `https://api.simplepod.ai/docs` lista propriedades para o body de `POST /instances/templates`, mas nao declara um array `required` explicito.

Campos usados no dry-run AYL:

- `name`
- `imageName`
- `categoryName`
- `defaultTag`
- `diskSize`
- `exposePorts`
- `startScript`
- `argOptions`
- `envVariables`
- `notes`
- `isPasswordProtected`
- `isRunSshServerOn`
- `isRunJupyterOn`

Campos documentados mas nao usados no dry-run por envolverem credenciais de registry privado:

- `host`
- `username`
- `password`

### Payload dry-run

```json
{
  "name": "ayl-wan22-s2v-fastapi-v1",
  "imageName": "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v1",
  "categoryName": "ayl-wan22-s2v",
  "defaultTag": "0.1.0",
  "diskSize": 32,
  "exposePorts": "8000",
  "startScript": "uvicorn app.main:app --host 0.0.0.0 --port 8000",
  "argOptions": "",
  "envVariables": [
    {
      "name": "SIMPLEPOD_MODELS_ROOT",
      "value": "/mnt/ayl_models"
    },
    {
      "name": "WAN22_S2V_MODEL_DIR",
      "value": "/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B"
    },
    {
      "name": "AYL_IMAGE_TAG",
      "value": "0.1.0"
    },
    {
      "name": "PYTHONUNBUFFERED",
      "value": "1"
    }
  ],
  "notes": "Dry-run payload for AYL Wan2.2 S2V FastAPI skeleton. No inference, no model download, no secrets.",
  "isPasswordProtected": false,
  "isRunSshServerOn": false,
  "isRunJupyterOn": false
}
```

Imagem Docker alvo quando existir publicacao GHCR:

```text
ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v1:0.1.0
```

### Volume e datacenter

Contexto operacional planejado, nao incluido no body documentado de `POST /instances/templates`:

- datacenter: `EU-PL-01`
- volume: `ayl_models_wan22_s2v_v1`
- mount path: `/mnt/ayl_models`
- modelo esperado: `/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B`

Pendencia: confirmar no SimplePod se o volume persistente e o datacenter entram no template, no create de instancia, ou apenas pela UI/selecionador de volume.

### Report

- report local: `logs/simplepod_template_create_dryrun_v1.json`
- guardrails: sem criar template, sem iniciar pod, sem deletar recursos, sem custo, sem baixar pesos, sem build/push

## Private Template Create Gate

Status preparado para o gate de criacao real do template privado SimplePod, ainda em dry-run por padrao.

Script:

```bash
python3 scripts/simplepod/temp_create_simplepod_template_v1.py
```

Criacao real futura exige confirmacao explicita:

```bash
python3 scripts/simplepod/temp_create_simplepod_template_v1.py --execute --confirm-create
```

Endpoint planejado:

```text
POST /instances/templates
```

Autenticacao:

```text
X-AUTH-TOKEN
```

Payload base reutiliza o dry-run validado:

```json
{
  "name": "ayl-wan22-s2v-fastapi-v1",
  "imageName": "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v1",
  "categoryName": "ayl-wan22-s2v",
  "defaultTag": "0.1.0",
  "diskSize": 32,
  "exposePorts": "8000",
  "startScript": "uvicorn app.main:app --host 0.0.0.0 --port 8000",
  "argOptions": "",
  "envVariables": [
    {
      "name": "SIMPLEPOD_MODELS_ROOT",
      "value": "/mnt/ayl_models"
    },
    {
      "name": "WAN22_S2V_MODEL_DIR",
      "value": "/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B"
    },
    {
      "name": "AYL_IMAGE_TAG",
      "value": "0.1.0"
    },
    {
      "name": "PYTHONUNBUFFERED",
      "value": "1"
    }
  ],
  "notes": "AYL Wan2.2 S2V FastAPI template. No inference, no model download, no secrets in template payload.",
  "isPasswordProtected": false,
  "isRunSshServerOn": false,
  "isRunJupyterOn": false
}
```

Report local:

```text
logs/simplepod_template_create_v1.json
```

Guardrails:

- dry-run por padrao: sim
- criacao real exige `--execute --confirm-create`
- iniciar instancia: nao
- criar pod: nao
- baixar pesos: nao
- rodar inferencia: nao
- imprimir segredos: nao

## GHCR image publish gate

Proximo bloco consolidado: publicar a imagem FastAPI no GHCR via GitHub Actions, porque o Mac local nao tem Docker CLI disponivel.

Workflow criado:

```text
.github/workflows/build-simplepod-wan22-s2v-fastapi-v1.yml
```

Imagem/tag alvo:

```text
ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v1:0.1.0
```

Contexto de build:

```text
docker/simplepod-wan22-s2v-fastapi-v1
```

Disparo manual:

1. Abrir GitHub Actions.
2. Selecionar `Build SimplePod Wan2.2 S2V FastAPI V1`.
3. Usar `Run workflow`.
4. Manter `image_tag=0.1.0`.

O workflow usa apenas o `GITHUB_TOKEN` padrao com `packages: write`. Nao inclui segredos SimplePod, R2, Hugging Face ou ModelScope.

Verificacao local de manifest GHCR, sem baixar camadas da imagem:

```bash
python3 scripts/simplepod/temp_check_ghcr_image_manifest_v1.py
```

O checker implementa o fluxo anonimo correto do registry:

1. solicita o manifest;
2. se receber `401`, le o header `WWW-Authenticate`;
3. extrai `realm`, `service` e `scope`;
4. solicita token anonimo ao `realm`;
5. repete o request do manifest com `Authorization: Bearer <token>`.

O token anonimo nao e salvo no report.

Report local:

```text
logs/simplepod_ghcr_image_manifest_v1.json
```

Status finais possiveis:

- `image_tag_found`
- `image_tag_not_found`
- `image_tag_private_or_auth_required`
- `ghcr_auth_challenge_parse_failed`
- `ghcr_token_request_failed`

Antes do publish, o resultado esperado e `image_tag_not_found` ou `image_tag_private_or_auth_required`, dependendo de como o GHCR expuser o pacote/tag. Depois do publish publico/acessivel, o resultado esperado e `image_tag_found`; o report deve incluir `Docker-Content-Digest` quando o GHCR retornar esse header.

Guardrails:

- nao executar SimplePod;
- nao iniciar pod;
- nao baixar pesos;
- nao rodar inferencia;
- nao incluir segredos SimplePod/R2 no workflow;
- manter `scripts/simplepod/temp_create_simplepod_template_v1.py` pronto, mas sem executar ate a imagem existir.

## Runtime smoke gate

Status preparado para iniciar a primeira instancia SimplePod somente quando houver confirmacao explicita. Objetivo: subir a imagem FastAPI, validar endpoints leves e encerrar com seguranca, sem baixar pesos e sem inferencia.

Template real criado:

```text
template_id=25114
template_name=ayl-wan22-s2v-fastapi-v1
image=ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v1:0.1.0
```

Volume ativo planejado:

```text
name=ayl_models_wan22_s2v_v1
datacenter=EU-PL-01
mount_path=/mnt/ayl_models
```

Endpoints REST identificados na documentacao SimplePod:

```text
GET /instances/market/list?rentalStatus=active
POST /instances
GET /instances/{id}
DELETE /instances/{id}
```

Port mapping: a documentacao informa que `GET /instances/{id}` e `GET /instances/list` retornam o campo `ports`, incluindo `proxyUrl`, usado para chegar na porta publica.

Body documentado para `POST /instances`:

- `gpuCount`
- `instanceMarket`
- `instanceTemplate`
- `startScript`
- `envVariables`

Nao foi encontrado campo documentado no body de `POST /instances` para anexar volume/network drive. O report do script registra `api_attach_status=not_documented_in_POST_/instances_body`; se o SimplePod exigir selecao explicita do Network Drive, isso deve ser feito no painel/template/UI antes do start real.

Script:

```bash
python3 scripts/simplepod/temp_simplepod_runtime_smoke_v1.py
```

Execucao real futura exige as duas confirmacoes:

```bash
python3 scripts/simplepod/temp_simplepod_runtime_smoke_v1.py --execute --confirm-start --confirm-delete
```

Modo curto de investigacao de porta/proxy, sem chamar `/health` quando nao houver URL publica:

```bash
python3 scripts/simplepod/temp_simplepod_runtime_smoke_v1.py --execute --confirm-start --inspect-only --confirm-delete
```

Payload dry-run:

```json
{
  "gpuCount": 1,
  "instanceMarket": "<selected_from_GET_/instances/market/list>",
  "instanceTemplate": "/instances/templates/25114",
  "startScript": "uvicorn app.main:app --host 0.0.0.0 --port 8000",
  "envVariables": [
    {
      "name": "SIMPLEPOD_MODELS_ROOT",
      "value": "/mnt/ayl_models"
    },
    {
      "name": "WAN22_S2V_MODEL_DIR",
      "value": "/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B"
    },
    {
      "name": "AYL_RUNTIME_SMOKE_ONLY",
      "value": "1"
    },
    {
      "name": "PYTHONUNBUFFERED",
      "value": "1"
    }
  ]
}
```

FastAPI smoke endpoints:

```text
GET /health
GET /gpu
GET /models
```

Report local:

```text
logs/simplepod_runtime_smoke_v1.json
```

Diagnostico apos primeiro runtime smoke real:

- instancia criada: `108208`
- delete: `DELETE /instances/108208` retornou `204`
- `GET /instances/{id}` retornou `200` repetidamente
- `public_api_base_url` ficou vazio
- `api_readiness`: `blocked_no_proxy_url_for_port_8000`

Hipotese mais provavel ate a proxima inspecao: o runtime criou a instancia, mas a API nao retornou `proxyUrl` para a porta `8000` no formato esperado. As causas candidatas sao, nesta ordem:

1. port mapping nao gerado para `8000`;
2. `proxyUrl` existe em outro campo/nivel e o extractor antigo nao capturou;
3. template precisa de `exposePortMappings` em vez de somente `exposePorts`;
4. container ainda nao estava pronto, embora `GET /instances/{id}` tenha respondido `200` em todas as leituras;
5. necessidade de flag de servico HTTP/Jupyter no template, se o SimplePod usar isso para publicar URLs.

O script foi atualizado para salvar no report uma versao redigida de `GET /instances/{id}` com:

- top-level keys;
- campos seguros relacionados a `ports`, `proxy`, `expose`, `network`, `status` e `state`;
- candidatos de URL publica;
- resumo seguro de `GET /instances/list` quando necessario.

Diagnostico do modo `inspect-only`:

```text
ports.direct[2].srcPort = 8000
ports.direct[2].destPort = 20008
ports.direct[2].ip = 194.93.49.14
ports.direct[2].url = ""
ports.proxy[2].url = "closed"
```

Correcao aplicada ao extractor:

- procurar primeiro `ports.direct` onde `srcPort == 8000`;
- ignorar `srcPort="0console"`;
- se `url` vier vazio, montar `http://{ip}:{destPort}`;
- ignorar `ports.proxy` quando `url == "closed"`;
- registrar `selected_api_port_mapping` no report.

Mapping esperado para a API FastAPI observado no inspect:

```json
{
  "source": "ports.direct.ip_destPort",
  "srcPort": 8000,
  "destPort": "20008",
  "ip": "194.93.49.14",
  "service": "PORT-8000",
  "protocol": "unknown",
  "url": "",
  "selected_url": "http://194.93.49.14:20008"
}
```

Guardrails:

- dry-run por padrao: sim
- start real exige `--execute --confirm-start --confirm-delete`
- delete/encerramento ao final exige `--confirm-delete`
- escolher GPU de menor custo observado pela API somente em execucao real
- custo estimado so deve ser registrado se a API retornar preco/oferta
- baixar pesos: nao
- rodar inferencia: nao
- imprimir segredos: nao

## Runtime V2 bootstrap gate

Novo bloco preparado em:

```text
review/simplepod_wan22_s2v_runtime_v2_bootstrap_plan.md
```

Imagem/tag alvo V2:

```text
ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.1.0
```

Arquivos criados:

```text
docker/simplepod-wan22-s2v-fastapi-v2/
.github/workflows/build-simplepod-wan22-s2v-fastapi-v2.yml
scripts/simplepod/temp_check_ghcr_image_manifest_v2.py
scripts/simplepod/temp_simplepod_runtime_smoke_v2.py
```

Objetivo do V2: validar runtime com `torch`/CUDA, `/mnt/ayl_models` e env R2 redigida, sem baixar pesos e sem inferencia.

Execucao real do smoke V2 fica bloqueada ate existir template V2:

```bash
python3 scripts/simplepod/temp_simplepod_runtime_smoke_v2.py --template-id <TEMPLATE_ID_V2> --execute --confirm-start --confirm-delete
```

## R2 I/O

## First Real Test Candidate — Maé FR 14.8s 1080x1080

Status registrado para preparacao local dry-run, sem upload R2, sem chamada SimplePod, sem template real, sem iniciar pod, sem baixar pesos e sem inferencia.

### Objetivo

Preparar o primeiro payload real de teste Wan2.2 S2V para SimplePod/FastAPI:

- personagem: Maé
- idioma base ensinado: FR
- objetivo: lip sync Wan2.2 S2V
- duracao alvo: 14.8s
- formato: 1:1
- resolucao alvo: 1080x1080
- fps: 16

### Inputs locais esperados

```text
~/Downloads/Maé para Wan V3.png
~/Downloads/mae_fr_14_8s_cut_for_wan.wav
```

O script de preparacao localiza os arquivos em `~/Downloads`, valida existencia, extensao `.png`/`.wav` e registra tamanhos em bytes. A busca do nome da imagem tolera normalizacao Unicode para evitar divergencia entre acento composto/decomposto no macOS.

### Script dry-run

```bash
python3 scripts/simplepod/temp_prepare_mae_wan22_s2v_14_8s_job_dryrun_v1.py
```

Report local:

```text
logs/simplepod_mae_wan22_s2v_14_8s_job_dryrun_v1.json
```

### Futuro endpoint

```text
POST /jobs/wan22-s2v
```

Payload dry-run planejado:

```json
{
  "job_id": "mae_fr_wan22_s2v_14_8s_1080_v1",
  "character_id": "mae",
  "base_taught_language": "FR",
  "reference_image_local_path": "~/Downloads/Maé para Wan V3.png",
  "audio_local_path": "~/Downloads/mae_fr_14_8s_cut_for_wan.wav",
  "target_width": 1080,
  "target_height": 1080,
  "target_duration_seconds": 14.8,
  "fps": 16,
  "output_video_key": "tests/simplepod_wan22_s2v/mae_fr_wan22_s2v_14_8s_1080_v1.mp4",
  "output_report_key": "tests/simplepod_wan22_s2v/mae_fr_wan22_s2v_14_8s_1080_v1_final_report.json"
}
```

### Guardrails confirmados

- upload R2 executado: nao
- chamada SimplePod executada: nao
- template real criado: nao
- pod iniciado: nao
- pesos baixados: nao
- inferencia executada: nao
- segredos impressos: nao

## R2 input staging for first Maé test

Gate preparado para staging dos inputs locais da Maé no R2, com dry-run como padrao. Nesta etapa o script valida arquivos locais e monta as chaves finais, mas nao executa upload real sem `--execute`.

### Script

```bash
python3 scripts/r2/temp_prepare_simplepod_mae_inputs_upload_v1.py
```

Upload real futuro, em etapa separada e explicita:

```bash
python3 scripts/r2/temp_prepare_simplepod_mae_inputs_upload_v1.py --execute
```

### Inputs locais

```text
/Users/fernandoreisdasilva/Downloads/Maé para Wan V3.png
/Users/fernandoreisdasilva/Downloads/mae_fr_14_8s_cut_for_wan.wav
```

O script valida existencia, arquivo regular, extensoes `.png` e `.wav`, tamanho em bytes e sha256 local. A localizacao do arquivo de imagem tolera normalizacao Unicode no nome.

### Chaves R2 planejadas

```text
tests/simplepod_wan22_s2v/inputs/mae_fr_wan22_s2v_14_8s_1080_v1/reference/Mae_para_Wan_V3.png
tests/simplepod_wan22_s2v/inputs/mae_fr_wan22_s2v_14_8s_1080_v1/audio/mae_fr_14_8s_cut_for_wan.wav
```

Report local:

```text
logs/simplepod_mae_inputs_r2_upload_v1.json
```

### Guardrails

- dry-run por padrao: sim
- upload R2 sem `--execute`: nao
- chamada SimplePod: nao
- pod criado: nao
- instancia iniciada: nao
- pesos baixados: nao
- inferencia executada: nao
- segredos impressos: nao

### Prefixos

```text
tests/simplepod_wan22_s2v_fastapi_v1/input/
tests/simplepod_wan22_s2v_fastapi_v1/output/
tests/simplepod_wan22_s2v_fastapi_v1/progress/
tests/simplepod_wan22_s2v_fastapi_v1/debug/
```

### Inputs minimos

```text
tests/simplepod_wan22_s2v_fastapi_v1/input/mae_reference.png
tests/simplepod_wan22_s2v_fastapi_v1/input/mae_audio.wav
tests/simplepod_wan22_s2v_fastapi_v1/input/job_request.json
```

### Outputs minimos

```text
tests/simplepod_wan22_s2v_fastapi_v1/output/video_out.mp4
tests/simplepod_wan22_s2v_fastapi_v1/output/final_report.json
tests/simplepod_wan22_s2v_fastapi_v1/debug/request_redacted.json
tests/simplepod_wan22_s2v_fastapi_v1/debug/env_presence_redacted.json
```

## FastAPI V1

### Endpoints minimos

```text
GET /health
GET /ready
GET /models/wan22-s2v/status
POST /jobs/wan22-s2v
GET /jobs/{job_id}
GET /jobs/{job_id}/report
```

### `GET /health`

Objetivo: responder se o processo HTTP esta vivo.

Resposta:

```json
{
  "status": "ok",
  "service": "ayl-simplepod-wan22-s2v-fastapi-v1",
  "timestamp": "...",
  "version": "0.1.0"
}
```

### `GET /ready`

Objetivo: validar readiness operacional sem rodar inferencia.

Checa:

- CUDA visivel;
- Python/imports basicos;
- Network Drive montado;
- manifest de pesos existe;
- arquivos obrigatorios existem e tem tamanho minimo;
- R2 env presente, sem imprimir segredo.

### `GET /models/wan22-s2v/status`

Objetivo: devolver inventario de modelo sem carregar tudo.

Campos:

- manifest path;
- model layout selecionado;
- arquivos esperados;
- existencia;
- tamanhos;
- checksums se manifest tiver;
- cache roots.

### `POST /jobs/wan22-s2v`

Request V1:

```json
{
  "job_id": "mae_test_0001",
  "input_image_key": "tests/simplepod_wan22_s2v_fastapi_v1/input/mae_reference.png",
  "input_audio_key": "tests/simplepod_wan22_s2v_fastapi_v1/input/mae_audio.wav",
  "output_video_key": "tests/simplepod_wan22_s2v_fastapi_v1/output/video_out.mp4",
  "final_report_key": "tests/simplepod_wan22_s2v_fastapi_v1/output/final_report.json",
  "width": 960,
  "height": 960,
  "num_frames": 81,
  "seed": 42,
  "steps": 4,
  "cfg": 1.0,
  "shift": 4.0,
  "denoise_strength": 0.85,
  "audio_scale": 1.35,
  "prompt": "stable talking head...",
  "negative_prompt": "excessive head movement...",
  "dry_run": true
}
```

V1 deve aceitar `dry_run=true` por padrao. Execucao real so com:

```json
{
  "dry_run": false,
  "confirm_inference": true
}
```

### `GET /jobs/{job_id}`

Resposta curta:

```json
{
  "job_id": "mae_test_0001",
  "status": "queued|running|succeeded|failed",
  "current_step": "download_inputs|load_models|inference|upload_outputs|done",
  "created_at": "...",
  "updated_at": "...",
  "output_video_key": "...",
  "final_report_key": "..."
}
```

### `GET /jobs/{job_id}/report`

Retorna o `final_report.json` local, se existir, ou busca do R2 se configurado.

## Variaveis Env Necessarias

### API/runtime

```text
AYL_RUN_MODE=simplepod_wan22_s2v_fastapi_v1
AYL_SERVICE_VERSION=0.1.0
SIMPLEPOD_VOLUME_NAME=ayl_models_wan22_s2v_v1
SIMPLEPOD_DATACENTER=EU-PL-01
SIMPLEPOD_MODELS_ROOT=/mnt/ayl_models
WAN22_S2V_MODEL_DIR=/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B
AYL_NETWORK_DRIVE_ROOT=/mnt/ayl_models
AYL_WAN22_S2V_MODEL_LAYOUT=comfyui_compat|native
AYL_WAN22_S2V_MANIFEST=/mnt/ayl_models/manifests/wan22_s2v_minimal_v1_manifest.json
AYL_WORKSPACE=/workspace
AYL_JOBS_DIR=/workspace/jobs
AYL_OUTPUT_DIR=/workspace/output
AYL_MAX_CONCURRENT_JOBS=1
AYL_REQUIRE_CONFIRM_INFERENCE=true
```

### R2

Reaproveitar contrato existente:

```text
R2_ENDPOINT
R2_ACCESS_KEY_ID
R2_SECRET_ACCESS_KEY
R2_BUCKET
R2_REGION
```

### Cache/modelos

```text
HF_HOME=/mnt/ayl_models/caches/huggingface
MODELSCOPE_CACHE=/mnt/ayl_models/caches/modelscope
TORCH_HOME=/mnt/ayl_models/caches/torch
TRITON_CACHE_DIR=/mnt/ayl_models/caches/triton
```

### Segurança/logs

```text
AYL_LOG_LEVEL=INFO
AYL_REDACT_ENV=true
AYL_REPORT_MAX_TRACE_CHARS=20000
```

## `final_report.json` V1

Reaproveitar o padrao mental dos reports atuais:

```json
{
  "test_id": "TEST_SIMPLEPOD_WAN22_S2V_FASTAPI_V1",
  "runtime_status": "ok|failed",
  "success": true,
  "current_step": "done",
  "created_at": "...",
  "finished_at": "...",
  "hostname": "...",
  "python_version": "...",
  "cuda_available": true,
  "gpu_name": "...",
  "env_present_redacted": {
    "R2_ENDPOINT": true,
    "R2_ACCESS_KEY_ID": true,
    "R2_SECRET_ACCESS_KEY": true,
    "R2_BUCKET": true,
    "R2_REGION": true
  },
  "network_drive": {
    "volume_name": "ayl_models_wan22_s2v_v1",
    "datacenter": "EU-PL-01",
    "root": "/mnt/ayl_models",
    "model_dir": "/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B",
    "exists": true,
    "model_manifest_exists": true
  },
  "input_files": {
    "image": {"r2_key": "...", "local_path": "...", "size_bytes": 0},
    "audio": {"r2_key": "...", "local_path": "...", "size_bytes": 0}
  },
  "model_files": [],
  "controls": {},
  "timings": {},
  "output_video_key": "...",
  "output_video_exists": true,
  "output_upload_status": "ok",
  "error_messages": [],
  "traceback": ""
}
```

## Reaproveitamento do Repo Atual

### Clientes R2

Reaproveitar padrao de:

- `scripts/r2/temp_download_wan22_s2v_probe_final_report_v1.py`
- `scripts/r2/temp_upload_wan22_s2v_probe_inputs_v1.py`
- `docker/wan22-s2v-runpod-v1/runtime_probe.py`

Padroes uteis:

- `load_repo_dotenv()`;
- `env_config()`;
- `r2_client()`;
- `head_object()`;
- `download_file()`;
- `upload_file()`;
- `endpoint_host_only()`;
- logs JSON locais em `logs/`.

### `final_report.json`

Reaproveitar ideias de:

- `docker/wan22-s2v-runpod-v1/runtime_probe.py`;
- `scripts/r2/temp_download_wan22_s2v_probe_final_report_v1.py`;
- `scripts/runpod/temp_test_runpod_wan22_s2v_probe_v1.py`.

Campos importantes a manter:

- status final;
- etapa atual;
- R2 keys;
- input/output facts;
- env presence redacted;
- GPU/torch probe;
- erros truncados;
- traceback limitado;
- flags `not_latentsync`, `not_wan27`, `no_runpod`.

### Redaction

Reaproveitar `sanitize_string()` de `scripts/runpod/temp_test_runpod_wan22_s2v_probe_v1.py`.

Tokens a redigir:

- `Bearer ...`;
- `api_key`;
- `access_key_id`;
- `secret_access_key`;
- qualquer env com `KEY`, `SECRET`, `TOKEN`, `PASSWORD`.

### Scripts RunPod que podem virar SimplePod

Nao copiar a camada GraphQL/Pod lifecycle. Reaproveitar apenas:

- dry-run default;
- `--execute`/confirmações explicitas;
- `intended_payload.json`;
- log JSON incremental;
- stdout curto e progressivo;
- verificacao de report final.

Candidatos:

- `scripts/runpod/temp_test_runpod_wan22_s2v_probe_v1.py` -> virar `scripts/simplepod/temp_submit_simplepod_wan22_s2v_job_v1.py`;
- `scripts/r2/temp_download_wan22_s2v_probe_final_report_v1.py` -> virar downloader de report SimplePod;
- `scripts/r2/temp_upload_wan22_s2v_probe_inputs_v1.py` -> virar uploader de input SimplePod;
- `docker/wan22-s2v-runpod-v1/runtime_probe.py` -> reaproveitar apenas utilitarios R2/report, nao ComfyUI/prompt.

## Etapas V1

### Fase 0: Inventario sem custo

1. Definir layout final do Network Drive.
2. Criar manifest de pesos esperado, sem baixar.
3. Mapear API direta Wan2.2 S2V oficial/diffusers versus pesos ComfyUI/Kijai.
4. Decidir se V1 usa:
   - caminho oficial `Wan-AI/Wan2.2-S2V-14B`;
   - ou pesos `safetensors` ComfyUI ja validados.

Saida:

- `review/simplepod_wan22_s2v_network_drive_inventory_v1.md`.

### Fase 1: Skeleton local sem inferencia

1. Criar `docker/simplepod-wan22-s2v-fastapi-v1/`.
2. Criar FastAPI com `/health`, `/ready`, `/models/wan22-s2v/status`.
3. Criar cliente R2 reutilizavel.
4. Criar report builder.
5. Criar script local de health check.

Validacao local:

```bash
python3 -m py_compile docker/simplepod-wan22-s2v-fastapi-v1/app/*.py
python3 scripts/simplepod/temp_check_simplepod_wan22_s2v_fastapi_health_v1.py
git diff --check
```

### Fase 2: Network Drive readiness sem inferencia

1. Subir container apenas para health/readiness.
2. Montar Network Drive.
3. Validar paths, manifest, tamanhos e checksums.
4. Nao carregar modelo ainda.
5. Nao gerar video ainda.

### Fase 3: R2 I/O smoke sem inferencia

1. `POST /jobs/wan22-s2v` com `dry_run=true`.
2. Baixar input pequeno do R2 para workspace.
3. Gerar `final_report.json` dry-run.
4. Subir report para R2.
5. Baixar report localmente e validar campos.

### Fase 4: Import/model load probe

1. Instalar dependencias definitivas.
2. Importar pipeline direto.
3. Carregar tokenizer/text encoder/model metadata.
4. Opcionalmente carregar pesos com `low_cpu_mem_usage`/offload, sem inferencia.
5. Reportar VRAM/RAM.

### Fase 5: Inferencia curta controlada

1. Rodar clipe minimo 1-2s.
2. Output MP4 em R2.
3. `final_report.json` completo.
4. Comparar qualidade e tempo contra RunPod/ComfyUI.

## Riscos

### Tecnicos

- Codigo oficial/diffusers pode nao aceitar exatamente os pesos ComfyUI/Kijai.
- Wan2.2 S2V 14B pode exigir GPU/VRAM maior do que SimplePod selecionado.
- Network Drive pode ter I/O insuficiente para carregar shards grandes.
- Dependencias CUDA/Torch podem divergir do container atual.
- FastAPI em processo unico pode bloquear durante inferencia longa se nao houver job queue simples.
- Falta de ComfyUI remove conveniencia de graph UI, mas tambem remove grande fonte de instabilidade.

### Operacionais

- SimplePod pode ter semantica propria para Network Drive e lifecycle.
- Permissoes de escrita no mount podem falhar.
- Persistencia do cache pode mascarar problemas de manifest.
- R2 continua sendo dependencia para I/O.

### Licenca/modelos

- Nao migrar para Wan2.7 sem fonte oficial validada.
- Manter Wan2.2 S2V como escopo inicial.
- Registrar origem, licenca e checksums dos pesos no manifest.

## Rollback para RunPod

Manter intactos:

- `docker/wan22-s2v-runpod-v1/`;
- `scripts/runpod/temp_test_runpod_wan22_s2v_probe_v1.py`;
- scripts R2 existentes do probe;
- prefixos R2 atuais:
  - `tests/runpod_wan22_s2v_probe_v1/...`;
  - `checkpoints/wan22_s2v/...`.

Rollback operacional:

1. Usar imagem RunPod ultima conhecida boa.
2. Usar scripts atuais de upload/download R2.
3. Rodar probe RunPod apenas se SimplePod V1 bloquear.
4. Nao apagar outputs/reports RunPod.

## Criterios de Sucesso

### V1 dry-run

- `/health` responde.
- `/ready` valida Network Drive e env R2 sem segredos.
- `/models/wan22-s2v/status` lista manifest e arquivos esperados.
- `POST /jobs/wan22-s2v` com `dry_run=true` gera report e nao executa inferencia.
- Report final sobe para R2.

### V1 inference

- Job real exige confirmacao explicita.
- Inputs baixam do R2.
- Pesos carregam do Network Drive.
- Output MP4 e `final_report.json` sobem para R2.
- Nenhum peso e baixado para R2 durante execucao.
- Reexecucao no mesmo SimplePod reaproveita Network Drive/cache.
- Logs nao imprimem segredos.

## Criterios para Abandonar ComfyUI

Abandonar ComfyUI para Wan2.2 S2V quando todos forem verdadeiros:

1. FastAPI gera pelo menos 3 videos S2V validos em cenarios A/B/C.
2. Tempo total sem cold weight download e melhor ou previsivel.
3. Nao ha dependencia de custom nodes.
4. Relatorio/debug e suficiente para diagnostico sem ComfyUI history.
5. Qualidade visual/lip sync e equivalente ou melhor.
6. Rollback RunPod continua disponivel por uma release.

Manter ComfyUI temporariamente se:

- pipeline direto nao carregar os pesos atuais;
- qualidade cair de forma significativa;
- SimplePod/Network Drive for instavel;
- dependencias oficiais exigirem reempacotamento maior.

## Proximos Comandos Sugeridos

Somente planejamento/dry-run, sem custo:

```bash
git status --short
git diff --check
```

Depois, em tarefa separada:

```bash
mkdir -p docker/simplepod-wan22-s2v-fastapi-v1/app scripts/simplepod
```

E entao criar apenas skeleton FastAPI sem inferencia:

```bash
python3 -m py_compile docker/simplepod-wan22-s2v-fastapi-v1/app/*.py
git diff --check
```

## Decisoes V1

- Nao executar SimplePod nesta etapa.
- Nao baixar pesos nesta etapa.
- Nao alterar Wan2.2 S2V atual.
- Nao alterar LatentSync.
- Nao alterar scripts RunPod existentes.
- R2 fica apenas como I/O.
- Network Drive passa a ser o local de pesos/cache.
- FastAPI propria substitui ComfyUI somente apos validacao incremental.

## SimplePod V2 Weights Download Gate

Status: gate preparado em dry-run para baixar pesos Wan2.2 S2V no Network Drive, sem inferencia.

Contexto validado:

- SimplePod V2 smoke aprovado;
- CUDA visivel com `torch_import_status=ok`;
- `/mnt/ayl_models` existe no container;
- modelo ainda ausente em `/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B`.

Decisao de imagem:

- nova imagem V2 tag `0.1.1` e necessaria;
- motivo: a tag `0.1.0` nao possui endpoint interno de download;
- imagem alvo: `ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.1.1`.

Endpoint administrativo adicionado:

```text
POST /admin/download-wan22-s2v-weights
```

Guardas:

- exige `AYL_ENABLE_ADMIN_DOWNLOADS=1`;
- exige `confirm_download=DOWNLOAD_WAN22_S2V_WEIGHTS`;
- restringe destino a `WAN22_S2V_MODEL_DIR`;
- registra inventario leve antes/depois;
- nao roda inferencia;
- nao gera video;
- nao imprime segredos.

Comando equivalente dentro do container:

```bash
huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir /mnt/ayl_models/wan2.2/Wan2.2-S2V-14B
```

Script local:

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
