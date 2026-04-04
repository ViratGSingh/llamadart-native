#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct PeaAdapter PeaAdapter;

PeaAdapter* pea_load(const char* path);
void        pea_free(PeaAdapter* adapter);
void        pea_inject(PeaAdapter* adapter,
                       struct llama_context* ctx,
                       const int32_t* tokens,
                       int32_t n_tokens);

#ifdef __cplusplus
}
#endif