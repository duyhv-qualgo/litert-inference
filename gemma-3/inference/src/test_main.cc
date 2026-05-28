#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "gemma_engine.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "Usage: %s <model.litertlm> [backend=cpu] [prompt]\n",
                     argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string backend    = (argc >= 3) ? argv[2] : "cpu";

    std::vector<std::string> prompts;
    if (argc >= 4) {
        prompts.emplace_back(argv[3]);
    } else {
        prompts = {
            "Hello! Briefly introduce yourself.",
            "Translate 'good morning' to Vietnamese.",
            "Summarize: LiteRT-LM runs LLMs on-device across Android, iOS, "
            "and desktop.",
        };
    }

    gemma::GemmaEngine eng;
    auto t0 = std::chrono::steady_clock::now();
    if (!eng.Init(model_path, backend)) {
        std::fprintf(stderr, "Init failed: %s\n", eng.last_error().c_str());
        return 2;
    }
    auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - t0).count();
    std::printf("Init: %lld ms\n", static_cast<long long>(dt_ms));

    gemma::GenerationConfig cfg;
    cfg.max_new_tokens = 64;

    for (const auto& p : prompts) {
        t0 = std::chrono::steady_clock::now();
        std::string out = eng.Generate(p, cfg);
        dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - t0).count();
        if (out.empty()) {
            std::fprintf(stderr, "Empty output for %s: %s\n",
                         p.c_str(), eng.last_error().c_str());
            return 3;
        }
        if (out.size() > 200) out.resize(200);
        std::printf("[%lld ms] %s -> %s\n",
                    static_cast<long long>(dt_ms), p.c_str(), out.c_str());
    }
    return 0;
}
