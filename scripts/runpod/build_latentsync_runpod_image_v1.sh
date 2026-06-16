#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CURRENT_DIR="$(pwd)"

if [[ "$CURRENT_DIR" != "$REPO_ROOT" ]]; then
  echo "ERROR: run this script from the repository root: $REPO_ROOT" >&2
  echo "Current directory: $CURRENT_DIR" >&2
  exit 1
fi

IMAGE_TAG="${LATENTSYNC_RUNPOD_IMAGE_TAG:-ayl-latentsync-runpod:0.1.0}"
DOCKERFILE="${LATENTSYNC_RUNPOD_DOCKERFILE:-docker/latentsync-runpod-v1/Dockerfile}"
CONTEXT="${LATENTSYNC_RUNPOD_CONTEXT:-.}"
DOWNLOAD_CHECKPOINTS="${LATENTSYNC_DOWNLOAD_CHECKPOINTS:-0}"

echo "LatentSync RunPod image build V1"
echo "Repository root: $REPO_ROOT"
echo "Image tag: $IMAGE_TAG"
echo "Dockerfile: $DOCKERFILE"
echo "Build context: $CONTEXT"
echo "Download checkpoints: $DOWNLOAD_CHECKPOINTS"
if [[ "$DOWNLOAD_CHECKPOINTS" != "1" ]]; then
  echo "Checkpoint downloads are disabled. This is the default lightweight build mode."
else
  echo "Checkpoint downloads are enabled. This build will download large model files."
fi
echo "No push will be performed by this script."

docker build \
  --file "$DOCKERFILE" \
  --tag "$IMAGE_TAG" \
  --build-arg DOWNLOAD_CHECKPOINTS="$DOWNLOAD_CHECKPOINTS" \
  "$CONTEXT"

docker image ls "$IMAGE_TAG"
