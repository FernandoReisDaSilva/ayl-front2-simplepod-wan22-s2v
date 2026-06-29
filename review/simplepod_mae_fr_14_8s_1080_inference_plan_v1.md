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
ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.1.3
```

Motivo:

- adiciona `POST /jobs/wan22-s2v/run`;
- valida payload e confirmacao de inferencia;
- ainda nao integra o runner real Wan2.2 S2V;
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
controlled_not_implemented_no_placeholder
```

O endpoint valida payload, GPU, R2 env e inventario dos pesos, mas retorna `blocked_real_inference_not_integrated` ate que o runner Wan2.2 S2V real seja incorporado. Nenhum output MP4 e gerado nesse modo.

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
  "confirm_inference": "RUN_WAN22_S2V_MAE_14_8S_1080"
}
```

## GPU Policy

Policy:

```text
first_inference_gpu_policy
```

Regras:

- `rentalStatus=active`;
- `datacenter=EU-PL-01`;
- `gpuCount=1`;
- `gpuMemorySize>=24000`;
- `order[pricePerGpu]=asc`;
- pick first.

RTX 3060 nao deve ser usada para inferencia real porque fica abaixo do minimo de 24GB VRAM.

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

## Proximo Bloqueio Real

Para produzir MP4 real, a imagem precisa incorporar o runner Wan2.2 S2V oficial/validado, com download de inputs R2, inferencia, upload do MP4 e upload do report final. Ate la, o gate deve retornar bloqueio controlado em vez de gerar output falso.
