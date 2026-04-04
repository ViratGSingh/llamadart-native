#include "pea_adapter.h"
#include "llama.h"
#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdio>

static constexpr uint32_t PEA_MAGIC = 0x50454100;

static uint64_t fnv1a(const char* s, size_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

struct PeaEntry {
    std::string        label;
    std::vector<float> vector;
};

struct PeaAdapter {
    std::unordered_map<uint64_t, PeaEntry> table;
    int   hidden_dim   = 0;
    float inject_scale = 0.08f;
};

PeaAdapter* pea_load(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return nullptr;

    uint32_t magic, version, n_entries, hidden_dim;
    f.read((char*)&magic,      4);
    f.read((char*)&version,    4);
    f.read((char*)&n_entries,  4);
    f.read((char*)&hidden_dim, 4);

    if (magic != PEA_MAGIC || !f) return nullptr;

    auto* adapter       = new PeaAdapter();
    adapter->hidden_dim = (int)hidden_dim;

    for (uint32_t i = 0; i < n_entries; i++) {
        uint64_t hash;
        uint16_t label_len;
        f.read((char*)&hash,      8);
        f.read((char*)&label_len, 2);

        PeaEntry entry;
        entry.label.resize(label_len);
        f.read(&entry.label[0], label_len);          // fixed: mutable char*
        entry.vector.resize(hidden_dim);
        f.read((char*)entry.vector.data(), hidden_dim * sizeof(float));

        if (!f) break;
        adapter->table[hash] = std::move(entry);
    }

    return adapter;
}

void pea_free(PeaAdapter* adapter) {
    delete adapter;
}

void pea_inject(PeaAdapter* adapter,
                struct llama_context* ctx,
                const int32_t* tokens,
                int32_t n_tokens) {
    if (!adapter || !ctx || n_tokens < 2) return;

    // Get vocab size for bounds checking
    const llama_model* model  = llama_get_model(ctx);
    const llama_vocab* vocab  = llama_model_get_vocab(model);
    const int          n_vocab = llama_vocab_n_tokens(vocab);

    // Get logits for the last token position — this is what gets sampled next
    float* logits = llama_get_logits_ith(ctx, n_tokens - 1);
    if (!logits) return;

    for (int i = 0; i < n_tokens - 1; i++) {
        char     key[32];
        int      klen = snprintf(key, sizeof(key),
                                 "%d %d", tokens[i], tokens[i + 1]);
        uint64_t h    = fnv1a(key, (size_t)klen);

        auto it = adapter->table.find(h);
        if (it == adapter->table.end()) continue;

        // L2 magnitude of taste vector as bias strength proxy
        const auto& vec = it->second.vector;
        float mag = 0.0f;
        for (float v : vec) mag += v * v;
        mag = sqrtf(mag / (float)vec.size());

        float bias = adapter->inject_scale * mag;

        // Apply bias directly to logits array
        llama_token tok = (llama_token)tokens[i + 1];
        if (tok >= 0 && tok < n_vocab) {
            logits[tok] += bias;
        }
    }
}