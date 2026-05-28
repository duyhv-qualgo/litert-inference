# gemma-3 / inference

Mobile C++ inference layer for the Gemma-3 270M IT `.litertlm` bundle exported
by `../export/export_gemma3_270m.py`. Wraps the
[LiteRT-LM](https://ai.google.dev/edge/litert-lm) **C API**
(`c/engine.h` — `LiteRtLmEngine`, `LiteRtLmConversation`,
`litert_lm_conversation_send_message`) so apps don't link the LiteRT-LM C++
ABI directly.

Unlike `../../matmoe-inference-c` (raw TFLite + hand-written decode loop for
a custom translation MoE), here **LiteRT-LM owns tokenization, Jinja prompt
templating, KV cache, and sampling** — we only marshal JSON messages in and
plain text out.

Status: **Linux x86_64 desktop end-to-end verified.** Android/iOS scripts
are ready; their prebuilts are downloads (iOS) or one bazel command away
(Android).

## Layout

```
include/gemma_engine.h          public C++ API
src/gemma_engine.cpp            C-API wrapper (Conversation::send_message)
src/test_main.cc                desktop CLI smoke test
platform/android/gemma_jni.cpp  JNI shim (libgemma_jni.so)
platform/ios/GemmaBridge.mm     Obj-C++ shim
scripts/fetch_prebuilt.sh       download LiteRT-LM C-API binaries (iOS/macOS)
scripts/build_desktop.sh        host build → test_gemma
scripts/build_android.sh        ABI cross-compile → jniLibs/<abi>/*.so
scripts/build_ios.sh            iphoneos/sim → GemmaEngine.xcframework
third_party/litert_lm/          (gitignored) fetched headers + libs
```

## Public API (`include/gemma_engine.h`)

```cpp
gemma::GemmaEngine eng;
if (!eng.Init("model.litertlm", "cpu")) {
    std::cerr << eng.last_error() << "\n";
    return 1;
}

gemma::GenerationConfig cfg;
cfg.max_new_tokens = 128;
cfg.temperature    = 0.8f;   // 0 → greedy

std::string answer = eng.Generate("Translate 'good morning' to Vietnamese.", cfg);

eng.GenerateStream("Write a haiku.", cfg,
    [](const std::string& chunk, bool done) {
        std::cout << chunk << std::flush;
        if (done) std::cout << "\n";
    });

eng.ResetSession();   // drop multi-turn context
```

`Generate` is blocking; `GenerateStream` blocks until the engine signals
`done=true` (it pumps the engine callback inline). The wrapper synthesises
the message JSON `{"role":"user","content":[{"type":"text","text":"…"}]}`
and extracts the `"text"` fields from the response — no third-party JSON
dependency.

## Build

### 1. Get the LiteRT-LM C library

#### iOS / macOS (released binary)
```bash
./scripts/fetch_prebuilt.sh    # tag pinned in scripts/fetch_prebuilt.sh
```
Drops the `CLiteRTLM.xcframework` into `third_party/litert_lm/lib/ios/` and
(on darwin hosts) the macOS xcframework into `…/lib/desktop/`.

#### Linux x86_64 / Android arm64 (build from source)
Upstream does not ship release `.so`s for these targets. Build them once with
bazel, then point CMake at the result.

```bash
# one-time prereqs (host clang 11 is too old for XNNPACK; install clang ≥18)
~/miniconda3/bin/conda create -y -n llvm --override-channels \
    -c conda-forge 'clangxx>=18'
curl -fsSL -o ~/.local/bin/bazelisk \
    https://github.com/bazelbuild/bazelisk/releases/download/v1.22.0/bazelisk-linux-amd64
chmod +x ~/.local/bin/bazelisk
ln -sf ~/.local/bin/bazelisk ~/.local/bin/bazel

# clone + LFS pull (LFS is required — the build links against prebuilt .so
# shims in prebuilt/<os>/ which are LFS-stored)
git clone --depth 1 --branch v0.12.0 \
    https://github.com/google-ai-edge/LiteRT-LM.git /mnt/data/litert_lm/src
cd /mnt/data/litert_lm/src && git lfs pull

# add a shared-library wrapper for //c:engine (bazel's cc_binary linkshared)
cat >> c/BUILD <<'EOF'
cc_binary(
    name = "libLiteRtLmCApi.so",
    linkshared = 1,
    linkopts = ["-Wl,-soname,libLiteRtLmCApi.so"],
    deps = [":engine"],
)
EOF

# build (linux host)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate llvm
bazel --output_user_root=/mnt/data/litert_lm/bazel_root build -c opt \
    --action_env=CC=$(which clang) --action_env=CXX=$(which clang++) \
    --action_env=PATH=$CONDA_PREFIX/bin:/usr/bin:/bin \
    //c:libLiteRtLmCApi.so

# copy artefacts into this repo
DEST=$(pwd)/../../gemma-3/inference/third_party/litert_lm   # adjust if needed
mkdir -p $DEST/include/litert_lm/c $DEST/lib/linux_x86_64
cp c/engine.h                          $DEST/include/litert_lm/c/
cp bazel-bin/c/libLiteRtLmCApi.so      $DEST/lib/linux_x86_64/
cp prebuilt/linux_x86_64/*.so          $DEST/lib/linux_x86_64/   # sibling deps
```

For Android, swap the final bazel command for
`bazel build -c opt --config=android_arm64 //c:libLiteRtLmCApi.so` (needs
`ANDROID_NDK_HOME`), then copy into `…/lib/android_arm64-v8a/`.

First bazel build is ~5 min on a 48-core box (analysis dominates); incremental
~10 s. Cache lives under `--output_user_root` (~50 GB).

### 2. Desktop smoke test

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate llvm
./scripts/build_desktop.sh
export LD_LIBRARY_PATH=$(pwd)/third_party/litert_lm/lib/linux_x86_64:$CONDA_PREFIX/lib
./build/desktop/test_gemma ../model/model.litertlm cpu
```

`LD_LIBRARY_PATH` is needed because `libLiteRtLmCApi.so` dynamically loads
sibling `.so`s (`libGemmaModelConstraintProvider`, `libLiteRt`, …) by name,
and the bazel-built `.so` has no rpath. The conda lib dir is needed because
the binary is linked against conda's newer libstdc++ (GLIBCXX_3.4.32+).
Bundle both directories into the runtime path of any shipping app.

Observed output (Linux x86_64, 4-thread CPU, gemma-3-270m-it):

```
Init: 1681 ms
[435 ms] Hello! Briefly introduce yourself.       -> "Hello! Briefly introduce myself."
[445 ms] Translate 'good morning' to Vietnamese.  -> "Chào buổi sáng tốt, mọi người."
[702 ms] Summarize: LiteRT-LM runs LLMs on-device -> "LiteRT-LM uses LLMs on-device across Android, iOS, and desktop."
```

For semantic parity reference, also run
`python ../export/verify_litertlm.py --model ../model/model.litertlm`.

### 3. Android

```bash
export ANDROID_NDK_HOME=/path/to/ndk
./scripts/build_android.sh                  # arm64-v8a
./scripts/build_android.sh --with-emu       # + x86_64 (for Studio emulator)
```

Output: `dist/android/jniLibs/<abi>/{libgemma_engine.so, libgemma_jni.so,
libLiteRtLmCApi.so}` plus the public header. Drop the whole `<abi>/`
directory into `app/src/main/jniLibs/<abi>/` and push `model.litertlm` to
the device (e.g. `/data/local/tmp/`).

Kotlin call site (matches the JNI symbols in `platform/android/gemma_jni.cpp`):

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

    fun reset() = nativeReset(handle)
    fun close() = nativeRelease(handle)

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
./scripts/fetch_prebuilt.sh   # populates third_party/litert_lm/lib/ios/
./scripts/build_ios.sh
```

Output: `dist/GemmaEngine.xcframework` + `dist/include/gemma_engine.h`. Ship
in Xcode together with `third_party/litert_lm/lib/ios/CLiteRTLM.xcframework`.

Swift call site (with the bridging header exposing `GemmaBridge` from
`platform/ios/GemmaBridge.mm`):

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

Measured on Linux x86_64, 4-thread CPU backend, `max_new_tokens=64`, model
`gemma-3-270m-it` (`model.litertlm`, 285 MB):

| Metric                  | Value    |
|---|---|
| Init (mmap + warm-up)   | ~1.7 s   |
| Generate (short reply)  | 0.4–0.7 s |

Mobile numbers TBD — please update once measured on a target device.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `clang: error: unknown argument: '-mavxvnniint8'` during bazel build | Host clang < 16. Install `clangxx>=18` from conda-forge (see step 1). |
| `cannot open shared object file: libGemmaModelConstraintProvider.so` at runtime | Sibling `.so`s missing from search path. Bundle the entire `third_party/litert_lm/lib/<platform>/` next to your binary, or set `LD_LIBRARY_PATH`. |
| `undefined reference to … GLIBCXX_3.4.32` at link time | System libstdc++ too old. Link with the conda libstdc++ (`-L $CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib`); CMake handles this automatically when `CONDA_PREFIX` is set. |
| Repetition / gibberish from model | The wrapper goes through `Conversation::send_message`, which applies the Gemma chat template. If you bypass that and call `Session::generate_content` directly, the model sees a raw prompt and degrades. |

## See also

- Re-export the bundle: `../export/export_gemma3_270m.py`
- Python parity reference: `../export/verify_litertlm.py`
- LiteRT-LM upstream: https://github.com/google-ai-edge/LiteRT-LM
- C API header (after fetch): `third_party/litert_lm/include/litert_lm/c/engine.h`
