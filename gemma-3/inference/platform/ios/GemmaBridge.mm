#import <Foundation/Foundation.h>

#include <memory>

#include "gemma_engine.h"

@interface GemmaBridge : NSObject
- (BOOL)initWithModelPath:(NSString*)path backend:(NSString*)backend;
- (NSString*)generate:(NSString*)prompt
            maxTokens:(int)maxTokens
          temperature:(float)temperature
                 topK:(int)topK
                 topP:(float)topP;
- (void)generateStream:(NSString*)prompt
             maxTokens:(int)maxTokens
           temperature:(float)temperature
                  topK:(int)topK
                  topP:(float)topP
              onChunk:(void (^)(NSString* chunk, BOOL done))onChunk;
- (void)reset;
- (NSString*)lastError;
@end

@implementation GemmaBridge {
    std::unique_ptr<gemma::GemmaEngine> _eng;
}

- (instancetype)init {
    if ((self = [super init])) {
        _eng = std::make_unique<gemma::GemmaEngine>();
    }
    return self;
}

- (BOOL)initWithModelPath:(NSString*)path backend:(NSString*)backend {
    return _eng->Init([path UTF8String], [backend UTF8String]) ? YES : NO;
}

static gemma::GenerationConfig MakeCfg(int n, float t, int k, float p) {
    gemma::GenerationConfig c;
    c.max_new_tokens = n;
    c.temperature    = t;
    c.top_k          = k;
    c.top_p          = p;
    return c;
}

- (NSString*)generate:(NSString*)prompt
            maxTokens:(int)maxTokens
          temperature:(float)temperature
                 topK:(int)topK
                 topP:(float)topP {
    auto cfg = MakeCfg(maxTokens, temperature, topK, topP);
    std::string out = _eng->Generate([prompt UTF8String], cfg);
    return [NSString stringWithUTF8String:out.c_str()];
}

- (void)generateStream:(NSString*)prompt
             maxTokens:(int)maxTokens
           temperature:(float)temperature
                  topK:(int)topK
                  topP:(float)topP
              onChunk:(void (^)(NSString* chunk, BOOL done))onChunk {
    auto cfg = MakeCfg(maxTokens, temperature, topK, topP);
    _eng->GenerateStream(
        [prompt UTF8String], cfg,
        [onChunk](const std::string& chunk, bool done) {
            NSString* s = [NSString stringWithUTF8String:chunk.c_str()];
            onChunk(s, done ? YES : NO);
        });
}

- (void)reset { _eng->ResetSession(); }

- (NSString*)lastError {
    return [NSString stringWithUTF8String:_eng->last_error().c_str()];
}

@end
