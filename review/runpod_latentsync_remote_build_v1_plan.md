# RunPod LatentSync Remote Build V1 Plan

Date: 2026-06-16

## Objective

Prepare a controlled remote-build route for the LatentSync V1 image because this Mac does not have Docker installed. This plan prepares the path only; it does not authorize a paid Pod, Docker build, Docker push, checkpoint download, GPU use, or inference.

The preferred first build is a lightweight image without checkpoints:

```text
LATENTSYNC_DOWNLOAD_CHECKPOINTS=0
```

The checkpoint build is a later step only after the lightweight build is proven.

## Core Risk

RunPod Pods are billable while alive. A failed builder that keeps running can continue charging. Any future execution must:

- require explicit `--execute --confirm-cost-risk`;
- avoid GPU unless there is a proven need;
- use no Network Volume in V1;
- upload progress/final reports to R2 instead of relying on `podLogs`;
- terminate automatically after build/push/report;
- surface `manual_cleanup_required: true` if termination cannot be verified.

## Route Options

### Option A: GitHub Actions Builder (Recommended)

Use GitHub Actions or another CI runner with Docker/buildx:

```text
Mac -> git push -> GitHub Actions -> registry
```

Why this is simplest:

- no paid RunPod Pod for build;
- native Docker/buildx environment;
- logs are native CI logs;
- secrets can be managed as GitHub Actions secrets;
- no Docker-in-Docker uncertainty on RunPod.

Best for the first no-checkpoint image.

### Option B: Registry Native Build

Use Docker Hub/GHCR/other registry build service if available:

```text
Mac -> git push -> registry build -> registry image
```

This is also simple if the registry supports the needed build resources and network access.

### Option C: RunPod Kaniko/BuildKit Builder Pod

Use a CPU-oriented disposable builder Pod with a build tool image such as Kaniko or BuildKit:

```text
Mac -> GitHub source ref -> RunPod builder Pod -> registry
                         -> R2 progress/report
```

This avoids a Docker daemon inside the Pod, but still has practical risks:

- builder image must include or be able to run the selected build tool;
- R2 report upload may need extra tooling;
- registry credentials must be passed safely;
- long image builds can exceed timeouts;
- if command override fails, Pod may stay alive.

### Option D: Docker-in-Docker Pod

Use a Docker daemon inside a Pod. This is not recommended for V1 unless a RunPod template is known to support it.

Risks:

- may require privileged mode or special runtime support;
- ordinary RunPod Pods may not allow Docker daemon startup;
- harder cleanup;
- higher chance of a paid stuck Pod.

### Option E: Another Builder Machine

Use any trusted machine with Docker installed:

```text
Mac -> git/R2/source archive -> builder machine -> registry
```

This is operationally straightforward if such a builder exists.

## Recommended Decision

For V1, prefer GitHub Actions or registry-native build over RunPod. If RunPod must be used, use a CPU builder Pod with Kaniko/BuildKit, no Network Volume, and no GPU. Do not use an A4000/GPU for image building.

Recommended RunPod execution candidate, if needed later:

- Cloud type: `COMMUNITY`
- Template: CPU-capable or minimal Ubuntu template if supported by the builder image
- GPU: avoid GPU; if RunPod requires a GPU field, use the cheapest available non-expensive candidate and keep `--allow-expensive-gpu` off
- Container disk: start with `40 GB` for no-checkpoint build; use `80 GB` only if dependency layers need it
- Network Volume: false
- Image: builder-specific image, not the final LatentSync image

Fallback if no suitable instance is available:

1. Do not retry with expensive GPU automatically.
2. Switch to GitHub Actions or another Docker-capable builder.
3. Build the no-checkpoint image first.

## Registry Prerequisites

Pick one target registry before any remote build:

- Docker Hub, GHCR, or another OCI registry visible to RunPod.
- Credentials with permission to push one image/tag.
- A final tag, for example:

```text
docker.io/USER/ayl-latentsync-runpod:0.1.0
ghcr.io/USER/ayl-latentsync-runpod:0.1.0
```

Required registry secret handling:

- never write password/token into intended payload;
- pass secrets only through RunPod `env`;
- redact any log/payload fields named `REGISTRY_PASSWORD`, `REGISTRY_TOKEN`, or similar.

## Environment Variables

Local orchestration:

- `RUNPOD_API_KEY`
- `R2_ENDPOINT`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `R2_BUCKET`
- `R2_REGION`

Remote build settings:

- `LATENTSYNC_SOURCE_GIT_URL`
- `LATENTSYNC_SOURCE_GIT_REF`
- `LATENTSYNC_RUNPOD_REMOTE_IMAGE_TAG`
- `LATENTSYNC_DOWNLOAD_CHECKPOINTS`, default `0`

Registry credentials:

- `REGISTRY_USERNAME`
- one of `REGISTRY_PASSWORD` or `REGISTRY_TOKEN`

Optional:

- `R2_REMOTE_BUILD_PREFIX`, default `tests/runpod_latentsync_remote_build_v1`

## Build Modes

### No-Checkpoint Build

Default mode:

```text
LATENTSYNC_DOWNLOAD_CHECKPOINTS=0
```

Purpose:

- validate Dockerfile dependency installation;
- reduce build time and image weight;
- avoid model downloads during first builder validation.

### Checkpoint Build

Explicit later mode:

```text
LATENTSYNC_DOWNLOAD_CHECKPOINTS=1
```

Purpose:

- embed `checkpoints/latentsync_unet.pt`;
- embed `checkpoints/whisper/tiny.pt`;
- improve future RunPod inference cold start.

Do this only after the no-checkpoint build succeeds.

## Logs And Reports

Do not depend on `podLogs`. The builder command should upload progress and final status to R2:

```text
{prefix}/progress/started.json
{prefix}/progress/build_started.json
{prefix}/progress/build_finished.json
{prefix}/output/remote_build_report.json
```

The local orchestrator should poll R2 for the final report and terminate the Pod when it appears or when timeout expires.

## Cleanup Strategy

Future execution must:

- terminate the Pod in `finally` once a Pod ID exists;
- report `manual_cleanup_required: true` if termination fails or cannot be verified;
- include the Pod ID in local logs only after creation;
- set a conservative timeout;
- avoid `sleep infinity` for builder Pods.

## Confirmation Gate

Default mode is dry-run only. The script must write:

- `tmp/runpod_latentsync_remote_build_v1/intended_payload.json`
- `logs/runpod_latentsync_remote_build_v1_log.json`

No mutation is allowed without both:

```bash
--execute --confirm-cost-risk
```

## Shortest Safe Path

1. Use GitHub Actions or another known Docker builder if available.
2. If RunPod is required, first run only a micro environment probe for builder tooling.
3. Build no-checkpoint image remotely.
4. Push to registry.
5. Only after the image is verified, consider checkpoint build.

