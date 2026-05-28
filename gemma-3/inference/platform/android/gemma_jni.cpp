#include <jni.h>

#include <memory>
#include <string>

#include "gemma_engine.h"

namespace {

inline gemma::GemmaEngine* AsEngine(jlong h) {
    return reinterpret_cast<gemma::GemmaEngine*>(h);
}

std::string JStr(JNIEnv* env, jstring s) {
    if (!s) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out(c);
    env->ReleaseStringUTFChars(s, c);
    return out;
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_example_gemma_GemmaEngine_nativeInit(
    JNIEnv* env, jclass, jstring jmodel_path, jstring jbackend) {
    auto eng = std::make_unique<gemma::GemmaEngine>();
    if (!eng->Init(JStr(env, jmodel_path), JStr(env, jbackend))) {
        return 0;
    }
    return reinterpret_cast<jlong>(eng.release());
}

JNIEXPORT void JNICALL
Java_com_example_gemma_GemmaEngine_nativeRelease(JNIEnv*, jclass, jlong h) {
    delete AsEngine(h);
}

JNIEXPORT jstring JNICALL
Java_com_example_gemma_GemmaEngine_nativeGenerate(
    JNIEnv* env, jclass, jlong h, jstring jprompt,
    jint max_new_tokens, jfloat temperature, jint top_k, jfloat top_p,
    jlong seed) {
    gemma::GenerationConfig cfg;
    cfg.max_new_tokens = max_new_tokens;
    cfg.temperature    = temperature;
    cfg.top_k          = top_k;
    cfg.top_p          = top_p;
    cfg.seed           = static_cast<uint64_t>(seed);
    std::string out = AsEngine(h)->Generate(JStr(env, jprompt), cfg);
    return env->NewStringUTF(out.c_str());
}

JNIEXPORT void JNICALL
Java_com_example_gemma_GemmaEngine_nativeGenerateStream(
    JNIEnv* env, jclass, jlong h, jstring jprompt,
    jint max_new_tokens, jfloat temperature, jint top_k, jfloat top_p,
    jlong seed, jobject jcallback) {
    JavaVM* vm = nullptr;
    env->GetJavaVM(&vm);
    jobject gcb = env->NewGlobalRef(jcallback);
    jclass cb_cls = env->GetObjectClass(jcallback);
    jmethodID m_on = env->GetMethodID(cb_cls, "onChunk", "(Ljava/lang/String;Z)V");

    gemma::GenerationConfig cfg;
    cfg.max_new_tokens = max_new_tokens;
    cfg.temperature    = temperature;
    cfg.top_k          = top_k;
    cfg.top_p          = top_p;
    cfg.seed           = static_cast<uint64_t>(seed);

    AsEngine(h)->GenerateStream(
        JStr(env, jprompt), cfg,
        [vm, gcb, m_on](const std::string& chunk, bool done) {
            JNIEnv* e = nullptr;
            bool attached = false;
            if (vm->GetEnv(reinterpret_cast<void**>(&e), JNI_VERSION_1_6) != JNI_OK) {
                vm->AttachCurrentThread(&e, nullptr);
                attached = true;
            }
            jstring js = e->NewStringUTF(chunk.c_str());
            e->CallVoidMethod(gcb, m_on, js, static_cast<jboolean>(done));
            e->DeleteLocalRef(js);
            if (attached) vm->DetachCurrentThread();
        });

    env->DeleteGlobalRef(gcb);
}

JNIEXPORT void JNICALL
Java_com_example_gemma_GemmaEngine_nativeReset(JNIEnv*, jclass, jlong h) {
    AsEngine(h)->ResetSession();
}

}  // extern "C"
