# litert-inference

On-device LLM inference experiments.

## Components

- **`gemma-3/export/`** — PyTorch → `.litertlm` export for `google/gemma-3-270m-it`.
- **`gemma-3/model/`** — exported `model.litertlm` bundle.
- **`gemma-3/inference/`** — C++ inference layer wrapping LiteRT-LM C API, plus
  JNI / Obj-C++ shims and build scripts for Android (`libgemma_engine.so`,
  `libgemma_jni.so`) and iOS (`GemmaEngine.xcframework`). See
  [`gemma-3/inference/README.md`](gemma-3/inference/README.md).
- **`matmoe-inference-c/`** (submodule) — separate engine for a custom JAX
  encoder-decoder MoE translation model, using raw TFLite + XNNPACK.
