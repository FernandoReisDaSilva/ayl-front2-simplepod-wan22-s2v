#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CURRENT_DIR="$(pwd)"

if [[ "$CURRENT_DIR" != "$REPO_ROOT" ]]; then
  echo "ERROR: run this script from the repository root: $REPO_ROOT" >&2
  echo "Current directory: $CURRENT_DIR" >&2
  exit 1
fi

SOURCE_IMAGE_TAG="${LATENTSYNC_RUNPOD_IMAGE_TAG:-ayl-latentsync-runpod:0.1.0}"
REMOTE_IMAGE_TAG="${LATENTSYNC_RUNPOD_REMOTE_IMAGE_TAG:?Set LATENTSYNC_RUNPOD_REMOTE_IMAGE_TAG, for example docker.io/USER/ayl-latentsync-runpod:0.1.0}"

echo "LatentSync RunPod image push V1"
echo "Repository root: $REPO_ROOT"
echo "Source image tag: $SOURCE_IMAGE_TAG"
echo "Remote image tag: $REMOTE_IMAGE_TAG"

docker tag "$SOURCE_IMAGE_TAG" "$REMOTE_IMAGE_TAG"
docker push "$REMOTE_IMAGE_TAG"
