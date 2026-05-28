#!/usr/bin/env bash
# Build libgemma_engine.so + libgemma_jni.so for Android.
# Requires: ANDROID_NDK_HOME, a prebuilt libLiteRtLmCApi.so under
#   third_party/litert_lm/lib/android_<abi>/ (see scripts/fetch_prebuilt.sh).

set -euo pipefail

ABIS=("arm64-v8a")
for arg in "$@"; do
  case "$arg" in
    --with-emu)   ABIS+=("x86_64") ;;
    --with-armv7) ABIS+=("armeabi-v7a") ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

: "${ANDROID_NDK_HOME:?Set ANDROID_NDK_HOME to your NDK root}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="${ROOT}/dist/android"
rm -rf "${DIST}"
mkdir -p "${DIST}/include"
cp "${ROOT}/include/gemma_engine.h" "${DIST}/include/"

for ABI in "${ABIS[@]}"; do
  BUILD="${ROOT}/build/android-${ABI}"
  cmake -S "${ROOT}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="${ABI}" \
    -DANDROID_PLATFORM=android-26
  cmake --build "${BUILD}" -j --target gemma_engine gemma_jni

  OUT="${DIST}/jniLibs/${ABI}"
  mkdir -p "${OUT}"
  cp "${BUILD}/libgemma_engine.so" "${OUT}/"
  cp "${BUILD}/libgemma_jni.so"    "${OUT}/"
  cp "${ROOT}/third_party/litert_lm/lib/android_${ABI}/"libLiteRtLmCApi.so "${OUT}/" 2>/dev/null \
    || echo "WARN: libLiteRtLmCApi.so missing for ${ABI} — bundle manually." >&2
done

echo "Android artifacts: ${DIST}/jniLibs/<abi>/"
