# gemma-3 / inference

Mobile C++ inference layer for the Gemma-3 270M IT `.litertlm` bundle
exported by `../export/export_gemma3_270m.py`. Wraps the
[LiteRT-LM](https://ai.google.dev/edge/litert-lm) C API
(`c/engine.h` — `LiteRtLmEngine`, `LiteRtLmSession`,
`litert_lm_session_generate_content`) so apps don't link C++ ABI directly.

Unlike `../../matmoe-inference-c` (raw TFLite + hand-written decode loop for
a custom translation MoE), here LiteRT-LM owns tokenization, prompt
templating, KV cache, and sampling — we only marshal strings.

## Layout

```
include/gemma_engine.h          public C++ API
src/gemma_engine.cpp            C-API wrapper
src/test_main.cc                desktop CLI smoke test
platform/android/gemma_jni.cpp  JNI shim (libgemma_jni.so)
platform/ios/GemmaBridge.mm     Obj-C++ shim
scripts/fetch_prebuilt.sh       download LiteRT-LM C lib artifacts
scripts/build_desktop.sh        host build → test_gemma
scripts/build_android.sh        ABI cross-compile → jniLibs/<abi>/*.so
scripts/build_ios.sh            iphoneos/sim → GemmaEngine.xcframework
third_party/litert_lm/          (gitignored) fetched headers + libs
```

## Public API (`include/gemma_engine.h`)

```cpp
gemma::GemmaEngine eng;
eng.Init("model.litertlm", "cpu");

gemma::GenerationConfig cfg;
cfg.max_new_tokens = 128;
cfg.temperature    = 0.8f;

std::string answer = eng.Generate("Hello! Briefly introduce yourself.", cfg);

eng.GenerateStream("Write a haiku.", cfg,
    [](const std::string& chunk, bool done) {
        std::cout << chunk << std::flush;
        if (done) std::cout << "\n";
    });
```

## Build

### 1. Fetch LiteRT-LM C library
```bash
./scripts/fetch_prebuilt.sh   # downloads iOS + macOS xcframeworks for v0.12.0
```
Upstream releases ship only iOS + macOS C-API binaries. For Linux desktop
and Android arm64 you must build `//c:LiteRtLmCApi` from the LiteRT-LM repo
with bazel and drop `libLiteRtLmCApi.so` into
`third_party/litert_lm/lib/{linux_x86_64,android_arm64-v8a}/`. The script
prints the exact bazel commands.

### 2. Desktop smoke test
```bash
./scripts/build_desktop.sh
./build/desktop/test_gemma ../model/model.litertlm
```
Expected: three prompts answered with per-call latency, comparable to
`../export/verify_litertlm.py`.

### 3. Android
```bash
export ANDROID_NDK_HOME=/path/to/ndk
./scripts/build_android.sh                  # arm64-v8a
./scripts/build_android.sh --with-emu       # + x86_64
```
Output: `dist/android/jniLibs/<abi>/{libgemma_engine.so, libgemma_jni.so,
libLiteRtLmCApi.so}`. Drop into `app/src/main/jniLibs/<abi>/` and push
`model.litertlm` to the device (e.g. `/data/local/tmp/`).

Kotlin call site (matches the JNI symbols in `gemma_jni.cpp`):

```kotlin
package com.example.gemma

class GemmaEngine private constructor(private val handle: Long) {
    fun generate(prompt: String, maxTokens: Int = 128, temperature: Float = 0.8f,
                 topK: Int = 40, topP: Float = 0.95f, seed: Long = 0): String =
        nativeGenerate(handle, prompt, maxTokens, temperature, topK, topP, seed)

    fun generateStream(prompt: String, cb: Callback, maxTokens: Int = 128,
                       temperature: Float = 0.8f, topK: Int = 40,
                       topP: Float = 0.95f, seed: Long = 0) =
        nativeGenerateStream(handle, prompt, maxTokens, temperature, topK, topP, seed, cb)

    fun reset()   = nativeReset(handle)
    fun close()   = nativeRelease(handle)

    interface Callback { fun onChunk(chunk: String, done: Boolean) }

    companion object {
        init { System.loadLibrary("gemma_jni") }
        fun load(modelPath: String, backend: String = "cpu"): GemmaEngine? {
            val h = nativeInit(modelPath, backend)
            return if (h == 0L) null else GemmaEngine(h)
        }
        @JvmStatic private external fun nativeInit(path: String, backend: String): Long
        @JvmStatic private external fun nativeRelease(handle: Long)
        @JvmStatic private external fun nativeGenerate(
            handle: Long, prompt: String, maxTokens: Int, temperature: Float,
            topK: Int, topP: Float, seed: Long): String
        @JvmStatic private external fun nativeGenerateStream(
            handle: Long, prompt: String, maxTokens: Int, temperature: Float,
            topK: Int, topP: Float, seed: Long, cb: Callback)
        @JvmStatic private external fun nativeReset(handle: Long)
    }
}
```

### 4. iOS
```bash
./scripts/build_ios.sh
```
Output: `dist/GemmaEngine.xcframework` + the public header. Ship in Xcode
alongside `third_party/litert_lm/lib/ios/CLiteRTLM.xcframework`.

Swift call site (with the bridging header exposing `GemmaBridge`):

```swift
let bridge = GemmaBridge()
guard bridge.init(withModelPath: modelPath, backend: "cpu") else {
    fatalError(bridge.lastError())
}
let answer = bridge.generate("Translate 'good morning' to Vietnamese.",
                             maxTokens: 64, temperature: 0.8,
                             topK: 40, topP: 0.95)

bridge.generateStream("Write a haiku.",
                      maxTokens: 64, temperature: 0.8, topK: 40, topP: 0.95) {
    chunk, done in
    print(chunk, terminator: "")
    if done { print() }
}
```

## Performance

| Metric | Backend | Value |
|---|---|---|
| Init (load `model.litertlm`) | cpu | TBD (fill after first run) |
| Generate, 64 tokens          | cpu | TBD |
| Generate, 64 tokens          | gpu | TBD |

## See also

- Re-export the bundle: `../export/export_gemma3_270m.py`
- Python parity reference: `../export/verify_litertlm.py`
- LiteRT-LM upstream: https://github.com/google-ai-edge/LiteRT-LM
