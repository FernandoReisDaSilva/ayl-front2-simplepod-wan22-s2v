# RunPod WAN 2.7 Probe V1

## Objetivo

Criar o caminho mínimo para validar WAN 2.7 no RunPod Community usando Cloudflare R2 como transporte técnico e uma imagem separada da linha LatentSync.

Este probe não aprova pipeline production, não usa Network Volume, não usa LatentSync e não executa GPU paga no dry-run.

## Arquivos Criados

```text
scripts/r2/temp_upload_wan27_probe_inputs_v1.py
scripts/r2/temp_check_wan27_probe_inputs_v1.py
scripts/runpod/temp_test_runpod_wan27_probe_v1.py
docker/wan27-runpod-v1/Dockerfile
docker/wan27-runpod-v1/entrypoint.sh
docker/wan27-runpod-v1/runtime_probe.py
data/wan27/inputs/mae_reference.png
```

## Imagem Planejada

```text
ghcr.io/fernandoreisdasilva/ayl-wan27-runpod:0.1.0
```

Esta imagem é separada de:

```text
ghcr.io/fernandoreisdasilva/ayl-latentsync-runpod
```

## R2 Keys

```text
tests/runpod_wan27_probe_v1/input/mae_reference.png
tests/runpod_wan27_probe_v1/input/audio.wav
tests/runpod_wan27_probe_v1/input/mae_5s.mp4
tests/runpod_wan27_probe_v1/output/video_out.mp4
tests/runpod_wan27_probe_v1/output/final_report.json
tests/runpod_wan27_probe_v1/progress/container_started.json
```

## Assets Locais

```text
~/Downloads/mae_5s.mp4
~/Downloads/mae_audio_5s.wav
data/wan27/inputs/mae_reference.png
```

`mae_reference.png` foi extraído do vídeo aprovado para suportar o caminho image-to-video.

## Runtime Contract

O modo:

```text
AYL_RUN_MODE=wan27_probe
```

faz:

- upload imediato de progress `container_started`;
- validação de torch/CUDA/GPU;
- validação de ffmpeg;
- download de imagem, áudio e vídeo fonte do R2;
- execução de `WAN27_COMMAND`, se configurado;
- upload de `video_out.mp4`;
- upload de `final_report.json`.

Se `WAN27_COMMAND` não estiver configurado na imagem/env, o report final usa:

```text
runtime_probe_status=wan27_command_not_configured
```

e o script RunPod não deve considerar o probe aprovado.

## Comandos Dry-Run

```bash
python3 scripts/r2/temp_upload_wan27_probe_inputs_v1.py
python3 scripts/r2/temp_check_wan27_probe_inputs_v1.py
python3 scripts/runpod/temp_test_runpod_wan27_probe_v1.py
```

## Upload Real Futuro

NÃO executar ainda:

```bash
python3 scripts/r2/temp_upload_wan27_probe_inputs_v1.py --execute --confirm-upload --overwrite
```

## Comando Pago Futuro

NÃO executar ainda:

```bash
python3 scripts/runpod/temp_test_runpod_wan27_probe_v1.py --execute --confirm-cost-risk
```

## Riscos Conhecidos

- A imagem `0.1.0` ainda precisa receber o comando/modelo WAN 2.7 real.
- RTX 3090 Community pode ser insuficiente para algumas variantes WAN 2.7; o V1 tenta manter 5s/480p para reduzir custo e risco.
- Sem Network Volume, pesos grandes precisam estar na imagem ou baixados por lógica futura controlada.
- Se a imagem não configurar `WAN27_COMMAND`, o Pod deve iniciar, escrever progress/final e falhar de forma explícita.

## Próximos Passos

1. Fazer upload dos inputs no R2.
2. Definir a fonte/comando WAN 2.7 real para a imagem `ayl-wan27-runpod`.
3. Buildar a imagem em GHCR com `push_image=false` primeiro, se houver workflow.
4. Publicar `ghcr.io/fernandoreisdasilva/ayl-wan27-runpod:0.1.0`.
5. Rodar o probe pago somente após confirmar inputs R2 e imagem publicada.
