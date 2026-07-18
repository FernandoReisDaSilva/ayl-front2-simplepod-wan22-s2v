# FP8 Wan Gate 0 Validation Plan

## Scope

This Gate 0 is isolated to the experimental FP8 image tree:

- `docker/simplepod-wan22-s2v-fastapi-v2-blackwell-fp8/`
- `scripts/fp8/temp_fp8_wan_gate0_probe_v1.py`

It does not alter the BF16 Blackwell runtime, persistent worker, FastAPI endpoints, SimplePod templates, production scripts, prompts, Wan parameters, R2 flow, or GHCR production tags.

The Wan source checkout is pinned for reproducibility:

- repository: `https://github.com/Wan-Video/Wan2.2.git`
- commit: `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`

The Docker build checks out that exact commit in detached mode and removes `.git`; it does not build from a floating upstream HEAD.

## Current Loading Path Audit

The validated BF16 worker loads Wan in `docker/simplepod-wan22-s2v-fastapi-v2-blackwell-fp8/app/wan22_s2v_persistent_worker.py`:

- `load_once()` inserts `/opt/Wan2.2` into `sys.path`.
- `load_once()` installs the SDPA attention fallback before model construction.
- `load_once()` sets `AYL_SAFETENSORS_CUDA_TO_CPU_PATCH=1` and installs the scoped `WanModel_S2V.from_pretrained` patch.
- `load_once()` imports `WanS2V` from `wan.speech2video`.
- `load_once()` instantiates `WanS2V(config=..., checkpoint_dir=..., device_id=0, t5_cpu=False, init_on_cpu=True, convert_model_dtype=True)`.
- `load_once()` restores the safetensors/from_pretrained patch after load and leaves the SDPA patch active for the worker lifetime.

The runtime wrapper in `docker/simplepod-wan22-s2v-fastapi-v2-blackwell-fp8/app/wan22_s2v_generate_wrapper.py` provides:

- `scoped_safetensors_cuda_to_cpu_patch(...)`
- `install_scoped_from_pretrained_patch()`
- `install_sdpa_attention_fallback_patch()`

The diagnostic path in `docker/simplepod-wan22-s2v-fastapi-v2-blackwell-fp8/app/main.py` confirmed that the native load path resolves to:

- `wan.speech2video.WanS2V`
- `wan.modules.s2v.model_s2v.WanModel_S2V.from_pretrained(...)`
- Accelerate dispatch through the native `from_pretrained` call.
- `torch_dtype=config.param_dtype`
- `device_map=torch.device("cuda:0")`

## Dependency Pin Audit

Audited files in the FP8 tree:

- `docker/simplepod-wan22-s2v-fastapi-v2-blackwell-fp8/requirements.txt`
- `docker/simplepod-wan22-s2v-fastapi-v2-blackwell-fp8/Dockerfile`
- `.github/workflows/build-simplepod-wan22-s2v-fastapi-v2-blackwell-fp8.yml`

No `requirements_fp8.txt`, `pyproject.toml`, `poetry.lock`, or lockfile currently exists inside the FP8 tree.

Direct dependency status:

- `torch==2.11.0` is pinned in the Dockerfile.
- `torchao==0.17.0` is pinned in the Dockerfile.
- All direct packages in `requirements.txt` are pinned with `==`.
- `dashscope` was the only unpinned direct dependency and is now pinned as `dashscope==1.26.3`.
- No direct dependency uses `>=`, `~=`, `*`, or `git+`.
- Wan source is pinned separately to commit `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`.

Remaining reproducibility note:

- Transitive dependencies are still resolved by `pip` at build time.
- After the first successful experimental build, capture the exact installed set with `pip freeze` from the image and promote it to an FP8-only constraints/lock file before using Gate 0 for comparisons across dates.

## Gate 0 Architecture

The new probe uses the same Wan loading principles, but does not use FastAPI, R2, SimplePod APIs, or the production worker:

1. Bootstrap Python, CUDA, Torch and TorchAO context.
2. Load `/opt/Wan2.2` and `/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B`.
3. Install SDPA attention fallback.
4. Install scoped safetensors CUDA-to-CPU staging patch only around `WanModel_S2V.from_pretrained`.
5. Instantiate `WanS2V` with `t5_cpu=False`, `offload_model=True` at generation time, and `convert_model_dtype=True`.
6. Inventory `WanS2V.noise_model` `nn.Linear` modules.
7. Inventory all modules under `WanS2V.noise_model` with quantized/skipped/failed decisions and explicit skip reasons.
8. Apply TorchAO `Float8WeightOnlyConfig` only to eligible large `nn.Linear` modules.
9. Verify module tree preservation and `nn.Linear` class preservation.
10. Run one minimal inference with synthetic local inputs and no video save.
11. Record metrics and unload.

## Quantization Plan

Quantized:

- `torch.nn.Linear` modules under `WanS2V.noise_model`.
- Only modules with at least `AYL_FP8_GATE0_MIN_LINEAR_PARAMS` parameters, default `16384`.

Kept BF16/native:

- LayerNorm/RMSNorm and all normalization modules, because Gate 0 only validates weight-only Linear quantization.
- Embeddings and token/text modules, because they are not Linear transformer projection targets.
- T5, VAE, wav2vec/audio encoder and audio preprocessing, because they are separate conditioning/codec components and not the first FP8 target.
- Non-Linear modules and small Linear modules, to reduce unsupported-kernel and accuracy risk in the first validation.

Known risks:

- TorchAO may reject a module after Accelerate dispatch.
- FP8 weight-only may reduce weight memory but not activation memory.
- Module wrapping may alter parameter representation while preserving module class.
- A synthetic minimal inference validates runtime compatibility only, not lip-sync quality.

## Instrumentation

The probe emits compact stdout markers and writes structured JSON:

- `bootstrap_started`
- `wan_load_started`
- `wan_load_finished`
- `fp8_quantization_started`
- `fp8_quantization_finished`
- `cuda_memory_before`
- `cuda_memory_after_load`
- `cuda_memory_after_quantization`
- `first_inference_started`
- `first_inference_finished`

Memory metrics:

- `allocated_gb`
- `reserved_gb`
- `peak_allocated_gb`
- `peak_reserved_gb`

Timing metrics:

- `load_seconds`
- `quantization_seconds`
- `first_inference_seconds`
- `runtime_seconds`

The report also records memory after final cleanup:

- `memory.cuda_memory_after_cleanup`
- `cleanup.cuda_memory_after_cleanup`

Environment identity:

- `environment.image_tag`
- `environment.wan_commit`
- `environment.torch_version`
- `environment.torchao_version`
- `environment.cuda_version`
- `environment.python_version`

## Success Criteria

Gate 0 passes only if:

- CUDA is available.
- Wan repo and Wan S2V model dir exist.
- `WanS2V` loads successfully.
- Safetensors CUDA-to-CPU staging patch is restored after load.
- SDPA attention fallback is installed or confirmed unnecessary.
- At least one eligible `nn.Linear` is quantized.
- No eligible `nn.Linear` quantization fails.
- Every inspected module has a quantization decision with either `status=quantized`, `status=skipped`, or `status=failed`.
- Skipped modules include a machine-readable `reason`, such as `non_linear_module`, `below_min_parameter_count`, or `excluded_name_token:<token>`.
- The module tree signature before and after quantization is identical.
- `nn.Linear` classes remain `nn.Linear`.
- Minimal `WanS2V.generate(...)` returns successfully.
- No long video is saved and no quality benchmark runs.
- Cleanup completes and `runtime_certification=PASS`.
- Environment identity is recorded in the report.

## Rollback Criteria

Rollback to BF16 or stop FP8 Gate 0 if:

- Wan load fails.
- TorchAO rejects eligible `nn.Linear` modules.
- Module tree changes after quantization.
- Minimal generation fails due to dtype, device, Accelerate hook, attention, or safetensors errors.
- VRAM increases beyond the BF16 baseline enough to remove the expected FP8 benefit.
- Cleanup leaves CUDA memory in an unsafe state for follow-up probes.

Rollback is operationally simple because the BF16 image tree and templates are untouched. Do not select the FP8 Gate 0 tag for production jobs.

## Recommended Commands

Build command, not executed in this preparation:

```bash
gh workflow run build-simplepod-wan22-s2v-fastapi-v2-blackwell-fp8.yml -f image_tag=0.3.02-blackwell-fp8-wan-gate0-v1
```

Paid SimplePod command, not executed in this preparation, after an experimental template points to the Gate 0 image:

```bash
python3 scripts/simplepod/temp_simplepod_fp8_runtime_probe_v1.py --template-id 26108 --execute --confirm-start --confirm-delete --startup-timeout-seconds 1200 --probe-timeout-seconds 900
```
