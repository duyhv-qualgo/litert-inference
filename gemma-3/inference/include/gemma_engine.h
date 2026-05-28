#ifndef GEMMA_INFERENCE_GEMMA_ENGINE_H_
#define GEMMA_INFERENCE_GEMMA_ENGINE_H_

#include <cstdint>
#include <functional>
#include <string>

struct LiteRtLmEngine;
struct LiteRtLmConversation;

namespace gemma {

struct GenerationConfig {
    int      max_new_tokens = 256;
    float    temperature    = 0.8f;
    int      top_k          = 40;
    float    top_p          = 0.95f;
    uint64_t seed           = 0;
};

class GemmaEngine {
 public:
    GemmaEngine();
    ~GemmaEngine();

    GemmaEngine(const GemmaEngine&) = delete;
    GemmaEngine& operator=(const GemmaEngine&) = delete;

    // backend: "cpu" or "gpu". Returns false and sets last_error() on failure.
    bool Init(const std::string& model_path,
              const std::string& backend = "cpu");

    std::string Generate(const std::string& prompt,
                         const GenerationConfig& cfg = {});

    using TokenCallback = std::function<void(const std::string& chunk, bool done)>;
    bool GenerateStream(const std::string& prompt,
                        const GenerationConfig& cfg,
                        TokenCallback on_token);

    void ResetSession();

    const std::string& last_error() const { return last_error_; }

 private:
    bool RecreateConversation(const GenerationConfig& cfg);

    LiteRtLmEngine*       engine_       = nullptr;
    LiteRtLmConversation* conversation_ = nullptr;
    std::string           last_error_;
};

}  // namespace gemma

#endif  // GEMMA_INFERENCE_GEMMA_ENGINE_H_
