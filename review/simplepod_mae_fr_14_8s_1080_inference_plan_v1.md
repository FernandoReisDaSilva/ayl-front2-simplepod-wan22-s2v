# SimplePod Maé FR 14.8s 1080 Inference Plan V1

Data: 2026-06-29

## Objetivo

Preparar o primeiro gate de inferencia Wan2.2 S2V para Maé FR, 14.8s, 1080x1080, fps 16, com dry-run por padrao e sem gerar placeholder.

## Contexto Validado

- SimplePod V2 `0.1.2` OK.
- Template SimplePod: `25114`.
- CUDA OK.
- Network Drive OK.
- Pesos Wan2.2 S2V verificados em `/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B`.
- Peso verificado: `recursive_total_size_gb=45.773`.
- Arquivos verificados: `recursive_file_count=148`.
- Inputs R2 Maé existem.

## Decisao de Imagem

Nova imagem necessaria: sim.

```text
ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.1.5
```

Motivo:

- substitui o bloqueio controlado por runner real single job Wan2.2 S2V;
- baixa input image/audio do R2 para diretorio temporario da instancia;
- chama o `generate.py` oficial Wan2.2 via wrapper;
- sobe MP4 real para R2 quando a inferencia conclui;
- sobe `final_report.json` para R2 em sucesso ou falha;
- corrige injecao de env R2 usando os nomes reais do `.env` local;
- adiciona preflight R2 antes da inferencia;
- nao gera placeholder;
- nao cria video falso.

## Endpoint

```text
POST /jobs/wan22-s2v/run
```

Confirmacao obrigatoria:

```text
confirm_inference=RUN_WAN22_S2V_MAE_14_8S_1080
```

Modo atual:

```text
real_single_job_no_scheduler
```

O endpoint valida payload, GPU, R2 env, HEAD dos inputs R2 e inventario dos pesos, baixa os inputs do R2, roda inferencia real Wan2.2 S2V e envia MP4/report para R2. Scheduler e paralelismo continuam fora do escopo.

## R2 Env

Nomes locais usados como fonte:

- `R2_ENDPOINT`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `R2_BUCKET`
- `R2_REGION`

Aliases aceitos pelo runtime:

- endpoint: `R2_ENDPOINT` ou `R2_ENDPOINT_URL`
- bucket: `R2_BUCKET` ou `R2_BUCKET_NAME`
- access key: `R2_ACCESS_KEY_ID`
- secret key: `R2_SECRET_ACCESS_KEY`
- region: `R2_REGION`, com fallback `auto`

O script cliente carrega `.env` explicitamente e injeta as cinco variaveis reais no payload SimplePod. Reports registram apenas `PRESENT/MISSING` e `value=<present_redacted>`.

Preflight antes da inferencia:

- env check;
- HEAD `reference_image_key`;
- HEAD `audio_key`;
- write permission check seguro em chave temporaria;
- abortar antes de Wan2.2 se env ou HEAD falhar.

## Payload

```json
{
  "job_id": "mae_fr_wan22_s2v_14_8s_1080_v1",
  "character_id": "mae",
  "base_taught_language": "FR",
  "reference_image_key": "tests/simplepod_wan22_s2v/inputs/mae_fr_wan22_s2v_14_8s_1080_v1/reference/Mae_para_Wan_V3.png",
  "audio_key": "tests/simplepod_wan22_s2v/inputs/mae_fr_wan22_s2v_14_8s_1080_v1/audio/mae_fr_14_8s_cut_for_wan.wav",
  "target_width": 1080,
  "target_height": 1080,
  "fps": 16,
  "target_duration_seconds": 14.8,
  "output_video_key": "tests/simplepod_wan22_s2v/outputs/mae_fr_wan22_s2v_14_8s_1080_v1.mp4",
  "output_report_key": "tests/simplepod_wan22_s2v/outputs/mae_fr_wan22_s2v_14_8s_1080_v1_final_report.json",
  "confirm_inference": "RUN_WAN22_S2V_MAE_14_8S_1080",
  "allow_oom_fallback": false
}
```

## Runner Real

Arquivo:

```text
docker/simplepod-wan22-s2v-fastapi-v2/app/wan22_s2v_runner.py
```

Wrapper:

```text
docker/simplepod-wan22-s2v-fastapi-v2/app/wan22_s2v_generate_wrapper.py
```

Comando base dentro do container:

```bash
python -m app.wan22_s2v_generate_wrapper --task s2v-14B --size 1080*1080 --ckpt_dir /mnt/ayl_models/wan2.2/Wan2.2-S2V-14B --offload_model True --convert_model_dtype --prompt "A natural, stable talking-head lip sync video of Maé speaking French." --image /tmp/ayl_wan22_s2v_jobs/mae_fr_wan22_s2v_14_8s_1080_v1/reference.png --audio /tmp/ayl_wan22_s2v_jobs/mae_fr_wan22_s2v_14_8s_1080_v1/audio.wav --save_file /tmp/ayl_wan22_s2v_jobs/mae_fr_wan22_s2v_14_8s_1080_v1/mae_fr_wan22_s2v_14_8s_1080_v1_1080x1080.mp4
```

O wrapper adiciona `1080*1080` ao mapa de tamanhos do Wan2.2 antes de chamar o `generate.py` oficial.

## GPU Policy

Policy:

```text
production_single_job_policy
```

Regras:

- `rentalStatus=active`;
- `datacenter=EU-PL-01`;
- `gpuCount=1`;
- `gpuMemorySize>=48000`;
- `order[pricePerGpu]=asc`;
- pick first.

RTX 3060 e GPUs de 24GB nao devem ser usadas neste primeiro teste Maé 14.8s 1080x1080. O gate usa MB do marketplace como criterio de selecao e exige 48GB nominal.

Normalizacao VRAM:

- `marketplace_gpuMemorySize_mb`: valor do SimplePod marketplace, usado para selecao;
- `nominal_vram_gb`: `marketplace_gpuMemorySize_mb / 1000`;
- `runtime_vram_total_gib`: valor reportado por `/gpu` em GiB;
- `policy_min_vram_mb`: `48000`;
- `policy_min_runtime_vram_gib`: `46.0`.

Nota: nao comparar GiB runtime contra GB decimal como falha automatica. Para 48GB nominal, aceitar runtime `>=46 GiB` como sanity check.

## Resolution Policy

- requested resolution: `1080x1080`;
- actual generation resolution planejada: `1080x1080`;
- fallback: `960x960` somente em caso de OOM;
- nao fazer upscale `960x960 -> 1080x1080` como padrao.

## Script

```bash
python3 scripts/simplepod/temp_simplepod_run_mae_wan22_s2v_14_8s_1080_v1.py
```

Execucao real futura:

```bash
python3 scripts/simplepod/temp_simplepod_run_mae_wan22_s2v_14_8s_1080_v1.py --execute --confirm-start --confirm-inference --confirm-delete
```

## Report

```text
logs/simplepod_mae_wan22_s2v_14_8s_1080_inference_v1.json
```

Campos obrigatorios:

- `requested_resolution`;
- `actual_generation_resolution`;
- `fallback_used`;
- `gpu_policy`;
- `gpuModel`;
- `gpuMemorySize`;
- `runtime_seconds`;
- `estimated_cost`;
- `oom_or_error_status`;
- phase timings;
- selected market id;
- rejected candidates summary.

## Guardas

- dry-run por padrao;
- exige `--execute --confirm-start --confirm-inference --confirm-delete`;
- verifica `/health`;
- verifica `/gpu`;
- verifica `GET /admin/verify-wan22-s2v-weights`;
- chama `POST /jobs/wan22-s2v/run`;
- tenta deletar instancia em `finally`;
- nao imprime segredos;
- nao baixa pesos;
- nao gera placeholder.
- nao implementa scheduler/paralelismo.
