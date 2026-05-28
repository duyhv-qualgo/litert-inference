#!/usr/bin/env bash
# Build GemmaEngine.xcframework (iphoneos arm64 + iphonesimulator arm64+x86_64)
# wrapping the prebuilt CLiteRTLM.xcframework from LiteRT-LM.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="${ROOT}/dist"
rm -rf "${DIST}/GemmaEngine.xcframework" "${DIST}/include"
mkdir -p "${DIST}/include"
cp "${ROOT}/include/gemma_engine.h" "${DIST}/include/"

LITERT_XC="${ROOT}/third_party/litert_lm/lib/ios/CLiteRTLM.xcframework"
[[ -d "${LITERT_XC}" ]] || { echo "Missing ${LITERT_XC}; run fetch_prebuilt.sh"; exit 1; }

build_slice() {
  local SDK="$1" ABI="$2" OUT="$3"
  local BUILD="${ROOT}/build/ios-${SDK}-${ABI}"
  cmake -S "${ROOT}" -B "${BUILD}" -G Xcode \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_SYSROOT="${SDK}" \
    -DCMAKE_OSX_ARCHITECTURES="${ABI}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
  cmake --build "${BUILD}" --config Release --target gemma_engine
  mkdir -p "$(dirname "${OUT}")"
  cp "${BUILD}/Release-${SDK}/libgemma_engine.dylib" "${OUT}" 2>/dev/null \
    || cp "${BUILD}/libgemma_engine.dylib" "${OUT}"
}

DEV="${DIST}/slices/dev/libgemma_engine.dylib"
SIM="${DIST}/slices/sim/libgemma_engine.dylib"
build_slice iphoneos        "arm64"        "${DEV}"
build_slice iphonesimulator "arm64;x86_64" "${SIM}"

xcodebuild -create-xcframework \
  -library "${DEV}" -headers "${ROOT}/include" \
  -library "${SIM}" -headers "${ROOT}/include" \
  -output "${DIST}/GemmaEngine.xcframework"

echo "Built: ${DIST}/GemmaEngine.xcframework"
echo "Ship together with: ${LITERT_XC}"
