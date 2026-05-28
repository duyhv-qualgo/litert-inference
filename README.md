# litert-inference

On-device LLM inference experiments. Two independent stacks live here, one
per model family.

## Stacks

### Gemma-3 (LiteRT-LM)

Production-style LLM stack: the model is packaged as a `.litertlm` bundle and
served by the upstream [LiteRT-LM](https://ai.google.dev/edge/litert-lm)
runtime (which owns tokenization, Jinja prompt templating, KV cache, and
sampling). We only ship a thin C++ wrapper plus JNI / Obj-C++ shims.

- [`gemma-3/export/`](gemma-3/export) — PyTorch → `.litertlm` packager for
  `google/gemma-3-270m-it`. Wraps `litert_torch.generative.export_hf`.
- [`gemma-3/model/`](gemma-3/model) — the exported `model.litertlm` (285 MB)
  plus its XNNPACK cache file.
- [`gemma-3/inference/`](gemma-3/inference) — C++ wrapper around the
  LiteRT-LM **C API** (`c/engine.h`, `LiteRtLmConversation`,
  `litert_lm_conversation_send_message`) + JNI / Obj-C++ shims + Android /
  iOS / desktop build scripts. See its
  [README](gemma-3/inference/README.md) for build + run instructions and
  the Bazel recipe used to produce `libLiteRtLmCApi.so` for Linux /
  Android. Status: **Linux x86_64 desktop verified** (~1.7 s init,
  0.4–0.7 s per short generation, 4-thread CPU).

### MatMoE translation (raw TFLite)

Custom JAX-trained en↔vi encoder-decoder with cross-attention and a
Mixture-of-Experts FFN, exported to two `.tflite` graphs and driven by a
hand-written decode loop on top of TFLite + XNNPACK. No `.litertlm`, no
LiteRT-LM runtime.

- [`matmoe-inference-c/`](matmoe-inference-c) (submodule) — `SlmEngine`
  C++ library, CLI smoke test, and ship-to-mobile scripts
  (`build_ios.sh`, `build_android.sh`). ~5 ms / decoded token on M-series.

## Why two stacks

| | Gemma-3 | MatMoE |
|---|---|---|
| Model packaging | `.litertlm` bundle | two raw `.tflite` graphs |
| Runtime | LiteRT-LM (Google upstream) | TFLite + XNNPACK directly |
| Tokenizer | inside the bundle | caller-supplied |
| Decode loop | inside LiteRT-LM | hand-written in `slm_engine.cpp` |
| Build system for the runtime | Bazel (LiteRT-LM upstream) → consume prebuilt `.so` | CMake (TFLite ships CMake) |

Use Gemma-3 when you want a general instruction-tuned chat model and don't
want to reimplement decoding. Use MatMoE when you have a custom small model
and need full control of the inference loop (it's allocation-free in the
hot loop).

## Layout

```
litert-inference/
├── gemma-3/
│   ├── export/         PyTorch → .litertlm packager
│   ├── model/          exported model.litertlm (gitignored if large)
│   └── inference/      C++ wrapper + JNI + Obj-C++ + build scripts
└── matmoe-inference-c/ (submodule) custom-model TFLite engine
```
