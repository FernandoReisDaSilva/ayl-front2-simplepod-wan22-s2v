# Wan2.2 S2V ComfyUI Prompt 400 Diagnostic V1

## Problema

O probe Wan2.2 S2V chegou ate:

```text
POST http://127.0.0.1:8188/prompt
```

mas o `final_report.json` so preservava o erro generico:

```text
400 Client Error: Bad Request for url: http://127.0.0.1:8188/prompt
```

Isso escondia a resposta real do ComfyUI, que normalmente contem o motivo da rejeicao do prompt.

## Alteracao

Arquivo alterado:

```text
docker/wan22-s2v-runpod-v1/runtime_probe.py
```

Antes de chamar `raise_for_status`, o worker agora captura:

- `comfyui_prompt_status_code`;
- `comfyui_prompt_response_text`;
- `comfyui_prompt_response_json`;
- `comfyui_prompt_response_headers`;
- `comfyui_prompt_payload_summary`.

Quando o status do `/prompt` for `>= 400`, o worker retorna um `final_report.json` com:

```text
runtime_probe_status=comfyui_prompt_http_error
output_upload_status=not_attempted
```

e ainda faz upload normal do `final_report.json` para R2.

## Payload Debug

O prompt enviado ao ComfyUI e salvo localmente antes do POST:

```text
/workspace/wan22_s2v_prompt_payload_debug.json
```

O worker tambem tenta subir esse arquivo para:

```text
tests/runpod_wan22_s2v_probe_v1/debug/prompt_payload_debug.json
```

Se o upload do debug falhar, o erro fica resumido em:

```text
comfyui_prompt_payload_summary.payload_debug_upload_status
```

## Truncamento

`comfyui_prompt_response_text` e truncado apenas se for muito grande, mantendo ate `50.000` caracteres, acima do minimo operacional de `20.000` caracteres.

## Escopo Negativo

Esta alteracao:

- nao altera LatentSync;
- nao altera WAN 2.7;
- nao executa RunPod;
- nao faz build/push;
- nao baixa pesos.

## Validacoes

```bash
python3 -m py_compile docker/wan22-s2v-runpod-v1/runtime_probe.py
git diff --check
```
