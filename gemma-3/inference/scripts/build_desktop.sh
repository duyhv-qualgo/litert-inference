#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build/desktop"
cmake -S "${ROOT}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD}" -j --target test_gemma
echo "Built: ${BUILD}/test_gemma"
echo "Try:   ${BUILD}/test_gemma ../model/model.litertlm"
