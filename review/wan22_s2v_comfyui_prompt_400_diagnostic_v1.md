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

## Causa Encontrada

No teste com imagem `0.1.1`, o ComfyUI retornou:

```text
missing_node_type
Node 'MarkdownNote' not found
Node ID '#61'
class_type: MarkdownNote
```

`MarkdownNote` e um no decorativo/anotacao do workflow UI. Ele nao tem funcao computacional no grafo API e nao deve ser enviado para `POST /prompt`.

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

## Correcao Aplicada

Antes de converter/enviar o workflow para `/prompt`, o worker agora remove nos decorativos destes tipos:

```text
MarkdownNote
Note
AnythingEverywhere
Reroute
```

O `final_report.json` passa a incluir:

```text
workflow_filter_removed_nodes
workflow_filter_removed_class_type_counts
workflow_filter_status
workflow_filter_preserved_non_decorative_node_classes
```

Se algum link depender de um no removido, o worker falha antes do POST com:

```text
runtime_probe_status=workflow_filter_error
output_upload_status=not_attempted
```

Para `MarkdownNote`, a expectativa e remocao sem impacto. O payload debug salvo/subido e o payload final ja filtrado.

## Ajuste Pos 0.1.2

No probe `0.1.2`, o filtro detectou que `PrimitiveNode` id `71`, title `num_frames`, era funcional:

```text
link 79: PrimitiveNode 71 -> WanVideoEmptyEmbeds 37 input num_frames
link 161: PrimitiveNode 71 -> WanVideoAddS2VEmbeds 101 input frame_window_size
```

Por isso, `PrimitiveNode` deixou de ser tratado como decorativo/removivel. Se ele voltar a gerar erro no `/prompt`, a proxima correcao deve converter `PrimitiveNode` para literal no payload API, nao remove-lo.

## Ajuste Pos 0.1.3

No probe `0.1.3`, o ComfyUI confirmou a ausencia de:

```text
missing_node_type
Node 'MelBandRoFormerModelLoader' not found
Node ID '#81'
class_type: MelBandRoFormerModelLoader
```

Decisao V1: `MelBandRoFormer` nao entra no conjunto minimo de pesos/nos porque o audio Mae usado no probe ja esta isolado.

O worker agora tenta aplicar um bypass especifico da cadeia de separacao de audio antes do POST:

```text
VHS_LoadAudio -> AudioEncoderEncode
```

Quando seguro, o `final_report.json` inclui:

```text
melband_bypass_status=ok
melband_bypass_removed_nodes
melband_bypass_removed_links
melband_bypass_new_links
melband_bypass_audio_source_node
melband_bypass_audio_target_node
```

Se a topologia nao for segura, o worker falha antes do POST com:

```text
runtime_probe_status=melband_bypass_error
output_upload_status=not_attempted
melband_bypass_status=error
melband_bypass_error
melband_bypass_detected_nodes
melband_bypass_detected_links
```

## Ajuste Pos 0.1.4

No probe `0.1.4`, o bypass detectou a topologia suficiente, mas falhou porque tentava identificar o input de audio por contagem generica de links.

Topologia confirmada:

```text
94 VHS_LoadAudio
64 AudioEncoderEncode
81 MelBandRoFormerModelLoader
82 MelBandRoFormerSampler
98 NormalizeAudioLoudness
82 output 0 -> 98 input audio
98 output 0 -> 64 input audio
```

Correcao aplicada: o alvo do bypass agora e explicitamente:

```text
AudioEncoderEncode 64 input audio
```

Se esse input nao existir, o worker falha antes do POST com:

```text
runtime_probe_status=melband_bypass_error
melband_bypass_error=AudioEncoderEncode input audio not found
```

Quando aplicado, o `final_report.json` tambem registra:

```text
melband_bypass_audio_target_input_name=audio
```

## Ajuste Pos 0.1.5

No probe `0.1.5`, o bypass MelBand funcionou:

```text
workflow_filter_status=ok
melband_bypass_status=ok
```

O proximo erro do ComfyUI foi:

```text
missing_node_type
Node 'DownloadAndLoadGIMMVFIModel' not found
Node ID '#95'
class_type: DownloadAndLoadGIMMVFIModel
```

Decisao V1: GIMMVFI nao entra no conjunto minimo. Ele e interpolacao/pos-processamento, nao S2V principal.

O worker agora tenta aplicar bypass GIMMVFI antes do POST:

- remove `DownloadAndLoadGIMMVFIModel` e demais nos diretamente dependentes da interpolacao;
- remove o `VHS_VideoCombine` ligado ao caminho interpolado;
- preserva o caminho S2V direto;
- prefere `VHS_VideoCombine` node `97` quando ele nao estiver no caminho interpolado;
- se nao for seguro identificar o combine direto, falha antes do POST com `runtime_probe_status=gimmvfi_bypass_error`.

Quando aplicado, o `final_report.json` registra:

```text
gimmvfi_bypass_status=ok
gimmvfi_bypass_removed_nodes
gimmvfi_bypass_removed_links
gimmvfi_bypass_preserved_video_path
gimmvfi_bypass_selected_video_combine_node
```

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

## Proxima Tag Sugerida

```text
0.1.6
```

## Validacoes

```bash
python3 -m py_compile docker/wan22-s2v-runpod-v1/runtime_probe.py
git diff --check
```
