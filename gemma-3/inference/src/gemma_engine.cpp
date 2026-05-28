#include "gemma_engine.h"

#include <condition_variable>
#include <mutex>
#include <string>

#include "litert_lm/c/engine.h"

namespace gemma {

namespace {

// Escape a UTF-8 string for embedding inside a JSON string literal.
std::string JsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

std::string BuildUserMessageJson(const std::string& prompt) {
    return std::string("{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"")
         + JsonEscape(prompt) + "\"}]}";
}

// Parse out the concatenation of every "text":"..." occurrence in a JSON
// blob. Handles standard JSON string escapes. Good enough for response shapes
// like {"role":"model","content":[{"type":"text","text":"..."}]}.
std::string ExtractText(const std::string& j) {
    std::string out;
    const std::string key = "\"text\":\"";
    size_t i = 0;
    while ((i = j.find(key, i)) != std::string::npos) {
        i += key.size();
        while (i < j.size()) {
            char c = j[i++];
            if (c == '\\' && i < j.size()) {
                char e = j[i++];
                switch (e) {
                    case '"':  out += '"';  break;
                    case '\\': out += '\\'; break;
                    case '/':  out += '/';  break;
                    case 'b':  out += '\b'; break;
                    case 'f':  out += '\f'; break;
                    case 'n':  out += '\n'; break;
                    case 'r':  out += '\r'; break;
                    case 't':  out += '\t'; break;
                    case 'u': {
                        if (i + 4 > j.size()) return out;
                        unsigned cp = 0;
                        for (int k = 0; k < 4; ++k) {
                            char h = j[i++];
                            cp <<= 4;
                            if (h >= '0' && h <= '9') cp |= h - '0';
                            else if (h >= 'a' && h <= 'f') cp |= h - 'a' + 10;
                            else if (h >= 'A' && h <= 'F') cp |= h - 'A' + 10;
                        }
                        if (cp < 0x80) {
                            out += static_cast<char>(cp);
                        } else if (cp < 0x800) {
                            out += static_cast<char>(0xC0 | (cp >> 6));
                            out += static_cast<char>(0x80 | (cp & 0x3F));
                        } else {
                            out += static_cast<char>(0xE0 | (cp >> 12));
                            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            out += static_cast<char>(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default: out += e;
                }
            } else if (c == '"') {
                break;
            } else {
                out += c;
            }
        }
    }
    return out;
}

LiteRtLmSessionConfig* MakeSessionConfig(const GenerationConfig& cfg) {
    LiteRtLmSessionConfig* sc = litert_lm_session_config_create();
    litert_lm_session_config_set_max_output_tokens(sc, cfg.max_new_tokens);
    litert_lm_session_config_set_apply_prompt_template(sc, true);
    LiteRtLmSamplerParams sp{};
    sp.type        = (cfg.temperature <= 0.0f) ? kLiteRtLmSamplerTypeGreedy
                                               : kLiteRtLmSamplerTypeTopP;
    sp.top_k       = cfg.top_k;
    sp.top_p       = cfg.top_p;
    sp.temperature = cfg.temperature;
    sp.seed        = static_cast<int32_t>(cfg.seed);
    litert_lm_session_config_set_sampler_params(sc, &sp);
    return sc;
}

}  // namespace

GemmaEngine::GemmaEngine() = default;

GemmaEngine::~GemmaEngine() {
    if (conversation_) litert_lm_conversation_delete(conversation_);
    if (engine_)       litert_lm_engine_delete(engine_);
}

bool GemmaEngine::Init(const std::string& model_path,
                       const std::string& backend) {
    LiteRtLmEngineSettings* settings = litert_lm_engine_settings_create(
        model_path.c_str(), backend.c_str(), nullptr, nullptr);
    if (!settings) {
        last_error_ = "litert_lm_engine_settings_create failed";
        return false;
    }
    engine_ = litert_lm_engine_create(settings);
    litert_lm_engine_settings_delete(settings);
    if (!engine_) {
        last_error_ = "litert_lm_engine_create failed (check model path / backend)";
        return false;
    }
    return RecreateConversation(GenerationConfig{});
}

bool GemmaEngine::RecreateConversation(const GenerationConfig& cfg) {
    if (conversation_) {
        litert_lm_conversation_delete(conversation_);
        conversation_ = nullptr;
    }
    LiteRtLmSessionConfig*      sc = MakeSessionConfig(cfg);
    LiteRtLmConversationConfig* cc = litert_lm_conversation_config_create();
    litert_lm_conversation_config_set_session_config(cc, sc);
    conversation_ = litert_lm_conversation_create(engine_, cc);
    litert_lm_conversation_config_delete(cc);
    litert_lm_session_config_delete(sc);
    if (!conversation_) {
        last_error_ = "litert_lm_conversation_create failed";
        return false;
    }
    return true;
}

std::string GemmaEngine::Generate(const std::string& prompt,
                                  const GenerationConfig& cfg) {
    if (!engine_ || !RecreateConversation(cfg)) return {};
    std::string msg = BuildUserMessageJson(prompt);
    LiteRtLmJsonResponse* resp = litert_lm_conversation_send_message(
        conversation_, msg.c_str(), nullptr, nullptr);
    if (!resp) {
        last_error_ = "conversation_send_message returned null";
        return {};
    }
    const char* j = litert_lm_json_response_get_string(resp);
    std::string out = j ? ExtractText(j) : std::string{};
    litert_lm_json_response_delete(resp);
    return out;
}

namespace {

struct StreamCtx {
    GemmaEngine::TokenCallback cb;
    std::mutex                 mu;
    std::condition_variable    cv;
    bool                       done = false;
    std::string                err;
};

void StreamTrampoline(void* data, const char* chunk, bool is_final,
                      const char* error_msg) {
    auto* ctx = static_cast<StreamCtx*>(data);
    if (error_msg) {
        std::lock_guard<std::mutex> g(ctx->mu);
        ctx->err  = error_msg;
        ctx->done = true;
        ctx->cv.notify_all();
        return;
    }
    if (chunk && *chunk) {
        std::string text = ExtractText(chunk);
        if (text.empty()) text.assign(chunk);
        ctx->cb(text, false);
    }
    if (is_final) {
        ctx->cb("", true);
        std::lock_guard<std::mutex> g(ctx->mu);
        ctx->done = true;
        ctx->cv.notify_all();
    }
}

}  // namespace

bool GemmaEngine::GenerateStream(const std::string& prompt,
                                 const GenerationConfig& cfg,
                                 TokenCallback on_token) {
    if (!engine_ || !RecreateConversation(cfg)) return false;
    StreamCtx ctx;
    ctx.cb = std::move(on_token);
    std::string msg = BuildUserMessageJson(prompt);
    int rc = litert_lm_conversation_send_message_stream(
        conversation_, msg.c_str(), nullptr, nullptr,
        &StreamTrampoline, &ctx);
    if (rc != 0) {
        last_error_ = "conversation_send_message_stream returned non-zero";
        return false;
    }
    std::unique_lock<std::mutex> lk(ctx.mu);
    ctx.cv.wait(lk, [&] { return ctx.done; });
    if (!ctx.err.empty()) {
        last_error_ = ctx.err;
        return false;
    }
    return true;
}

void GemmaEngine::ResetSession() {
    RecreateConversation(GenerationConfig{});
}

}  // namespace gemma
