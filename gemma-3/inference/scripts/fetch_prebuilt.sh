#!/usr/bin/env bash
# Download LiteRT-LM prebuilt artifacts into third_party/litert_lm/.
#
# What ships in upstream GitHub releases (verified for v0.12.0):
#   - CLiteRTLM.xcframework.zip       (iOS device + simulator slices, C API)
#   - CLiteRTLM_mac.xcframework.zip   (macOS host)
#
# What is NOT in releases:
#   - Linux desktop .so
#   - Android arm64 .so
# Those must be built from source with bazel against the LiteRT-LM tree.
# This script downloads what is shipped and prints instructions for the rest.

set -euo pipefail

VERSION="${LITERT_LM_VERSION:-v0.12.0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/third_party/litert_lm"
INC="${DEST}/include/litert_lm"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

mkdir -p "${INC}/c" "${DEST}/lib/ios" "${DEST}/lib/desktop"

echo "==> Fetching LiteRT-LM ${VERSION}"

# 1. C API header (always needed; small)
curl -fsSL -o "${INC}/c/engine.h" \
    "https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/${VERSION}/c/engine.h"

# 2. iOS xcframework (released artifact)
echo "==> Downloading CLiteRTLM.xcframework.zip"
curl -fL -o "${TMP}/ios.zip" \
    "https://github.com/google-ai-edge/LiteRT-LM/releases/download/${VERSION}/CLiteRTLM.xcframework.zip"
unzip -q -o "${TMP}/ios.zip" -d "${DEST}/lib/ios/"

# 3. macOS xcframework (released artifact) — also useful for the desktop smoke test on macOS
if [[ "${OSTYPE:-}" == darwin* ]]; then
  echo "==> Downloading CLiteRTLM_mac.xcframework.zip"
  curl -fL -o "${TMP}/mac.zip" \
      "https://github.com/google-ai-edge/LiteRT-LM/releases/download/${VERSION}/CLiteRTLM_mac.xcframework.zip"
  unzip -q -o "${TMP}/mac.zip" -d "${DEST}/lib/desktop/"
fi

echo "${VERSION}" > "${DEST}/VERSION"

cat <<EOF

==> Done. Prebuilts in ${DEST}

Next steps per platform:
  - iOS:     ready (xcframework at ${DEST}/lib/ios/).
  - macOS:   ready on darwin hosts (xcframework at ${DEST}/lib/desktop/).
  - Linux desktop / Android arm64:
      Upstream does NOT ship release binaries. Build with bazel:
        git clone --depth 1 -b ${VERSION} https://github.com/google-ai-edge/LiteRT-LM /tmp/litert_lm_src
        cd /tmp/litert_lm_src
        bazel build -c opt //c:LiteRtLmCApi                              # linux host
        bazel build -c opt --config=android_arm64 //c:LiteRtLmCApi       # android
      Copy resulting libLiteRtLmCApi.so into:
        ${DEST}/lib/linux_x86_64/    (host)
        ${DEST}/lib/android_arm64-v8a/ (android)
EOF
