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

## Ajuste Pos 0.1.6

No probe `0.1.6`, os bypasses anteriores funcionaram:

```text
workflow_filter_status=ok
melband_bypass_status=ok
gimmvfi_bypass_status=ok
```

O novo bloqueio foi:

```text
missing_node_type
Node 'PrimitiveNode' not found
Node ID '#71'
class_type: PrimitiveNode
```

O `PrimitiveNode` id `71`, title `num_frames`, e funcional porque alimenta:

```text
node 37 WanVideoEmptyEmbeds input num_frames
node 101 WanVideoAddS2VEmbeds input frame_window_size
```

Decisao: `PrimitiveNode` funcional nao deve ser removido como decorativo, mas tambem nao deve ir para `/prompt`. O worker agora resolve o valor literal, substitui os links nos inputs de destino e remove o `PrimitiveNode` do payload final.

Para o caso `num_frames`, se nao houver valor seguro em `widgets_values`, o worker usa `WAN22_S2V_NUM_FRAMES` com fallback controlado `81`.

Quando aplicado, o `final_report.json` registra:

```text
primitive_resolve_status=ok
primitive_resolve_node_id=71
primitive_resolve_title=num_frames
primitive_resolve_targets
primitive_resolve_resolved_nodes
primitive_resolve_replaced_inputs
primitive_resolve_removed_links
```

Se nao conseguir resolver com seguranca, o worker falha antes do POST com:

```text
runtime_probe_status=primitive_resolve_error
output_upload_status=not_attempted
primitive_resolve_status=error
primitive_resolve_error
primitive_resolve_detected_nodes
primitive_resolve_detected_links
```

## Ajuste Pos 0.1.7

No probe `0.1.7`, o erro `PrimitiveNode` continuou e o `final_report.json` veio sem campos `primitive_resolve_*`, indicando que o resolver nao executou no fluxo real antes do `POST /prompt` ou nao entrou na tag testada.

Correcao aplicada: a etapa `resolve_primitive_nodes` fica obrigatoriamente depois de:

```text
workflow_filter
melband_bypass
gimmvfi_bypass
```

e antes de:

```text
payload debug
POST /prompt
```

O `final_report.json` agora sempre inclui, mesmo quando nao houver `PrimitiveNode`:

```text
primitive_resolve_status
primitive_resolve_detected_nodes
primitive_resolve_resolved_nodes
primitive_resolve_replaced_inputs
primitive_resolve_removed_links
primitive_resolve_remaining_primitive_nodes
```

Para `PrimitiveNode` `71` / `num_frames`, quando nao houver valor explicito no payload/API, o fallback V1 fica registrado:

```text
primitive_resolve_fallback_used=true
primitive_resolve_fallback_reason=PrimitiveNode num_frames value not found in API payload; using V1 probe fallback 81
```

Se ainda restar `PrimitiveNode` no payload final, o worker falha antes do POST com:

```text
runtime_probe_status=primitive_resolve_error
output_upload_status=not_attempted
primitive_resolve_status=error
primitive_resolve_error=PrimitiveNode remained in final payload
```

## Revisao Pre 0.1.8

O grep ainda mostrava `PrimitiveNode` perto da lista de filtros, mas ele estava na lista de preservacao/relatorio, nao em `DECORATIVE_NODE_TYPES`.

Para evitar ambiguidade antes da tag, a constante foi renomeada para:

```text
REPORT_ONLY_NON_DECORATIVE_NODE_TYPES
```

`DECORATIVE_NODE_TYPES` continua contendo somente:

```text
MarkdownNote
Note
AnythingEverywhere
Reroute
```

`PrimitiveNode` continua sendo tratado apenas por `resolve_primitive_nodes()`, depois de `workflow_filter`, `melband_bypass` e `gimmvfi_bypass`, antes do payload debug e antes do `POST /prompt`.

## Ajuste Pos 0.1.8

No probe `0.1.8`, a resolucao de `PrimitiveNode` funcionou:

```text
workflow_filter_status=ok
melband_bypass_status=ok
gimmvfi_bypass_status=ok
primitive_resolve_status=ok
```

O novo estagio e validacao de valores do prompt. O ComfyUI retornou `prompt_outputs_failed_validation` por valores desalinhados do workflow UI:

- caminhos de modelo com `\` em vez de `/`;
- LoRA nao incluida no V1 minimo;
- `ImageResizeKJv2.device` contendo HTML em vez de `cpu`/`gpu`;
- `WanVideoSampler.scheduler` e `riflex_freq_index` desalinhados.

Foi adicionada a etapa:

```text
sanitize_prompt_values(prompt, object_info)
```

Ela roda depois de `resolve_primitive_nodes()` e antes do payload debug / `POST /prompt`.

O `final_report.json` passa a incluir:

```text
prompt_sanitize_status
prompt_sanitize_changes
prompt_sanitize_errors
prompt_sanitize_remaining_suspect_values
```

Se ainda houver string HTML em qualquer input apos a sanitizacao, o worker falha antes do POST com:

```text
runtime_probe_status=prompt_sanitize_error
output_upload_status=not_attempted
```

## Ajuste Pos 0.1.9

No probe `0.1.9`, o payload foi aceito pelo `/prompt`, a validacao passou e a execucao ComfyUI iniciou.

Novo bloqueio:

```text
ValueError: Can't import SageAttention: No module named 'sageattention'
ComfyUI-WanVideoWrapper/nodes_model_loading.py loadmodel
```

Decisao V1: antes de instalar `sageattention`, tentar desabilitar SageAttention pelo payload quando houver controle seguro.

`sanitize_prompt_values(prompt, object_info)` agora procura inputs relacionados a:

```text
sage
sage_attention
use_sage
use_sage_attention
attention
attention_mode
attention_backend
```

Quando encontrar controle, tenta escolher valor seguro aceito pelo `object_info`, preferindo:

```text
false
disabled
sdpa
pytorch
torch
flash_attn
```

O `final_report.json` passa a incluir:

```text
sageattention_policy
sageattention_detected_inputs
sageattention_sanitize_changes
sageattention_remaining_enabled_values
```

Se nenhum input controlavel existir:

```text
sageattention_policy=no_payload_control_found
```

Nenhuma instalacao de `sageattention` foi adicionada nesta etapa.

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
0.1.10
```

## Validacoes

```bash
python3 -m py_compile docker/wan22-s2v-runpod-v1/runtime_probe.py
git diff --check
```
