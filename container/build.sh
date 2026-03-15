#!/bin/bash
# Build the NanoClaw agent container image

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="nanoclaw-agent"
TAG="${1:-latest}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"

echo "Building NanoClaw agent container image..."
echo "Image: ${IMAGE_NAME}:${TAG}"

# Check if we have a proxy and use host network for build
NETWORK_ARG=""
BUILD_ARGS=""

# Use host.docker.internal for macOS Docker/Colima
if [ -n "$HTTP_PROXY" ] || [ -n "$http_proxy" ]; then
  PROXY="${HTTP_PROXY:-$http_proxy}"
  # Replace 127.0.0.1 with host.docker.internal for Docker
  BUILD_PROXY="${PROXY//127.0.0.1/host.docker.internal}"
  BUILD_ARGS="--build-arg HTTP_PROXY=$BUILD_PROXY --build-arg http_proxy=$BUILD_PROXY"
fi
if [ -n "$HTTPS_PROXY" ] || [ -n "$https_proxy" ]; then
  PROXY="${HTTPS_PROXY:-$https_proxy}"
  BUILD_PROXY="${PROXY//127.0.0.1/host.docker.internal}"
  BUILD_ARGS="$BUILD_ARGS --build-arg HTTPS_PROXY=$BUILD_PROXY --build-arg https_proxy=$BUILD_PROXY"
fi

${CONTAINER_RUNTIME} build $BUILD_ARGS -t "${IMAGE_NAME}:${TAG}" .

echo ""
echo "Build complete!"
echo "Image: ${IMAGE_NAME}:${TAG}"
echo ""
echo "Test with:"
echo "  echo '{\"prompt\":\"What is 2+2?\",\"groupFolder\":\"test\",\"chatJid\":\"test@g.us\",\"isMain\":false}' | ${CONTAINER_RUNTIME} run -i ${IMAGE_NAME}:${TAG}"
