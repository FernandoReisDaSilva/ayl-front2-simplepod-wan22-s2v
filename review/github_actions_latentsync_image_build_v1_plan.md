# GitHub Actions LatentSync Image Build V1 Plan

Date: 2026-06-16

## Objective

Use GitHub Actions as the main remote builder for the LatentSync RunPod V1 Docker image because the Mac has no local Docker installation and RunPod Community Pod creation is currently unstable.

This route prepares a manual build workflow only. Do not run it until the GitHub repo/tag target is confirmed.

## Why GitHub Actions Is Primary

GitHub Actions is the shortest controlled path because it provides Docker/buildx without Docker Desktop on the Mac and without relying on a paid RunPod builder Pod. It also gives native logs, clear manual dispatch, and GHCR integration through `GITHUB_TOKEN`.

RunPod RTX 3090 Community plus R2 should be reserved for execution tests after the image exists:

- micro-probe: image starts, imports work, R2 marker/report works;
- first real lip-sync: one short `3-5s` video/audio pair.

Build belongs in GitHub Actions; inference belongs in RunPod.

## Workflow

Created:

```text
.github/workflows/build-latentsync-runpod-v1.yml
```

Manual trigger:

```text
Actions -> Build LatentSync RunPod V1 -> Run workflow
```

Inputs:

- `image_tag`: default `ghcr.io/OWNER/ayl-latentsync-runpod:0.1.0`.
- `download_checkpoints`: default `false`.
- `push_image`: default `false`.

Important: replace `OWNER` with the GitHub user or organization that owns the package namespace before pushing.

## Permissions

Workflow declares:

```yaml
permissions:
  contents: read
  packages: write
```

`contents: read` is always needed for checkout. `packages: write` is only needed when `push_image=true`, but GitHub Actions permissions are declared at workflow/job scope, so it is present in the workflow to support the manual push path.

No registry secrets are hardcoded. GHCR login uses:

```text
username: github.actor
password: secrets.GITHUB_TOKEN
```

## GHCR Setup

Before pushing:

1. Confirm GitHub Actions is enabled for the repository.
2. Confirm the workflow has permission to write packages.
3. Choose the final GHCR tag:

```text
ghcr.io/<owner>/ayl-latentsync-runpod:0.1.0
```

Use lowercase owner/repo-style image names for GHCR.

## Tag Strategy

Recommended first tag:

```text
ghcr.io/<owner>/ayl-latentsync-runpod:0.1.0
```

For checkpoint builds later:

```text
ghcr.io/<owner>/ayl-latentsync-runpod:0.1.0-with-checkpoints
```

Avoid `latest` until the first working RunPod execution path is proven.

## Build Modes

### Default: No Checkpoints

Inputs:

```text
download_checkpoints=false
push_image=false
```

Effect:

```text
DOWNLOAD_CHECKPOINTS=0
```

This validates the Dockerfile dependency path without downloading model files and without publishing an image by default.

### Push No-Checkpoint Image

Inputs:

```text
download_checkpoints=false
push_image=true
image_tag=ghcr.io/<owner>/ayl-latentsync-runpod:0.1.0
```

This publishes the dependency image for RunPod micro-probe.

### Later: Checkpoint Image

Inputs:

```text
download_checkpoints=true
push_image=true
image_tag=ghcr.io/<owner>/ayl-latentsync-runpod:0.1.0-with-checkpoints
```

This downloads and embeds `latentsync_unet.pt` and `whisper/tiny.pt`. Do this only after the no-checkpoint image path is proven.

## Risks

- Build time may be long because PyTorch CUDA wheels and media/model dependencies are large.
- GitHub-hosted runner disk may be tight for a checkpoint image.
- `download_checkpoints=true` downloads several GB and can increase image size to roughly `12-16 GB`.
- `push_image=true` publishes to GHCR and may consume package storage/bandwidth.
- The Dockerfile currently uses the upstream `micromamba` latest URL; pinning remains a future reproducibility improvement.

## Next Step After Image Published

After a no-checkpoint image is pushed:

1. Use RunPod RTX 3090 Community only for execution, not build.
2. Start with a paid micro-probe only after explicit approval:
   - create Pod from the GHCR image;
   - verify imports and CUDA visibility;
   - verify R2 progress/report upload;
   - terminate immediately.
3. Then run one short lip-sync test:
   - `3-5s` input video;
   - `3-5s` WAV;
   - R2 input/output;
   - terminate immediately.

Do not use RunPod for build unless GitHub Actions or registry-native build fails.
