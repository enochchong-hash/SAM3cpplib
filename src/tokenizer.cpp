// sam3cpplib -- ported from tools/sam3.cpp @ 049884b (wip/trt-phase1,
// upstream PABannier/sam3.cpp + gemma4 patches 0001-0013). See docs/PLAN.md.
#include "sam3_internal.h"

/*****************************************************************************
** BPE Tokenizer — CLIP-style byte-level BPE
*****************************************************************************/

/*
** ── UTF-8 helpers ────────────────────────────────────────────────────────────
*/

static int sam3_utf8_len(uint8_t c) {
    if (c < 0x80) return 1;
    if (c < 0xC0) return 1;  // continuation (shouldn't start here)
    if (c < 0xE0) return 2;
    if (c < 0xF0) return 3;
    return 4;
}

static std::string sam3_codepoint_to_utf8(int cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

// Check if position i in s starts a Unicode letter.
// Handles ASCII letters + treats any multibyte UTF-8 start byte as a letter.
// This is a reasonable approximation without ICU.
static bool sam3_is_letter(const std::string& s, size_t i) {
    uint8_t c = (uint8_t)s[i];
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return true;
    if (c >= 0xC0) return true;  // multibyte UTF-8 → treat as letter
    return false;
}

/*
** ── Byte-to-unicode mapping (CLIP / GPT-2 style) ────────────────────────────
*/

// Maps each byte 0-255 to a unique unicode character (as UTF-8 string).
// Printable bytes map to themselves; non-printable bytes map to U+0100..U+0143.
static void sam3_init_byte_encoder(std::unordered_map<uint8_t, std::string>& enc) {
    // Collect printable byte values
    std::vector<int> bs;
    for (int i = 33; i <= 126; ++i) bs.push_back(i);
    for (int i = 161; i <= 172; ++i) bs.push_back(i);
    for (int i = 174; i <= 255; ++i) bs.push_back(i);

    // Corresponding codepoints (printable → identity)
    std::vector<int> cs(bs.begin(), bs.end());

    // Non-printable bytes get codepoints starting at 256
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    enc.clear();
    for (size_t i = 0; i < bs.size(); ++i) {
        enc[(uint8_t)bs[i]] = sam3_codepoint_to_utf8(cs[i]);
    }
}

/*
** ── Merge key helper ─────────────────────────────────────────────────────────
*/

// Unit separator (0x1F) cannot appear in byte-encoded BPE tokens.
static inline std::string sam3_merge_key(const std::string& a, const std::string& b) {
    std::string k;
    k.reserve(a.size() + 1 + b.size());
    k += a;
    k += '\x1f';
    k += b;
    return k;
}

/*
** ── Load embedded BPE tokenizer from binary stream ──────────────────────────
*/

bool sam3_load_bpe_vocab_from_stream(std::ifstream& fin, sam3_bpe_tokenizer& tok) {
    uint32_t tok_magic;
    fin.read(reinterpret_cast<char*>(&tok_magic), 4);
    if (tok_magic != SAM3_TOK_MAGIC) {
        fprintf(stderr, "%s: invalid tokenizer magic: 0x%08x (expected 0x%08x)\n",
                __func__, tok_magic, SAM3_TOK_MAGIC);
        return false;
    }

    // Read vocab
    int32_t n_vocab;
    fin.read(reinterpret_cast<char*>(&n_vocab), 4);
    tok.encoder.clear();
    tok.decoder.clear();
    for (int i = 0; i < n_vocab; ++i) {
        int32_t token_len;
        fin.read(reinterpret_cast<char*>(&token_len), 4);
        std::string token(token_len, '\0');
        fin.read(&token[0], token_len);
        int32_t token_id;
        fin.read(reinterpret_cast<char*>(&token_id), 4);
        tok.encoder[token] = token_id;
        tok.decoder[token_id] = token;
    }

    // Read merges
    int32_t n_merges;
    fin.read(reinterpret_cast<char*>(&n_merges), 4);
    tok.merges.clear();
    tok.merge_ranks.clear();
    for (int i = 0; i < n_merges; ++i) {
        int32_t len_a;
        fin.read(reinterpret_cast<char*>(&len_a), 4);
        std::string a(len_a, '\0');
        fin.read(&a[0], len_a);
        int32_t len_b;
        fin.read(reinterpret_cast<char*>(&len_b), 4);
        std::string b(len_b, '\0');
        fin.read(&b[0], len_b);
        tok.merge_ranks[sam3_merge_key(a, b)] = (int)tok.merges.size();
        tok.merges.push_back({std::move(a), std::move(b)});
    }

    if (fin.fail()) return false;

    // Init byte encoder and special tokens
    sam3_init_byte_encoder(tok.byte_encoder);
    tok.sot_token = 49406;
    tok.eot_token = 49407;

    fprintf(stderr, "%s: loaded %zu vocab entries, %zu merges\n",
            __func__, tok.encoder.size(), tok.merges.size());
    return true;
}

/*
** ── BPE encode a single word ─────────────────────────────────────────────────
*/

// Split a UTF-8 string into individual unicode characters.
static std::vector<std::string> sam3_utf8_chars(const std::string& s) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < s.size()) {
        int len = sam3_utf8_len((uint8_t)s[i]);
        chars.push_back(s.substr(i, len));
        i += len;
    }
    return chars;
}

// Apply BPE merges to a byte-encoded word string.
// Returns space-separated BPE tokens (e.g. "he llo</w>").
static std::string sam3_bpe_encode(sam3_bpe_tokenizer& tok, const std::string& token) {
    auto cit = tok.cache.find(token);
    if (cit != tok.cache.end()) return cit->second;

    // Split into unicode chars, append </w> to last
    std::vector<std::string> word = sam3_utf8_chars(token);
    if (word.empty()) return "";
    word.back() += "</w>";

    if (word.size() == 1) {
        tok.cache[token] = word[0];
        return word[0];
    }

    while (true) {
        // Find pair with lowest merge rank
        int best_rank = INT_MAX;
        std::string best_first, best_second;

        for (size_t i = 0; i + 1 < word.size(); ++i) {
            auto it = tok.merge_ranks.find(sam3_merge_key(word[i], word[i + 1]));
            if (it != tok.merge_ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_first = word[i];
                best_second = word[i + 1];
            }
        }

        if (best_rank == INT_MAX) break;

        // Merge all occurrences of this pair
        std::string merged = best_first + best_second;
        std::vector<std::string> new_word;
        for (size_t i = 0; i < word.size();) {
            if (i + 1 < word.size() &&
                word[i] == best_first && word[i + 1] == best_second) {
                new_word.push_back(merged);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i++;
            }
        }
        word = std::move(new_word);
        if (word.size() == 1) break;
    }

    // Join with spaces
    std::string result;
    for (size_t i = 0; i < word.size(); ++i) {
        if (i > 0) result += ' ';
        result += word[i];
    }
    tok.cache[token] = result;
    return result;
}

/*
** ── Pre-tokenizer (CLIP regex approximation) ─────────────────────────────────
*/

// Splits text into word tokens following the CLIP pattern:
//   <|startoftext|> | <|endoftext|> | 's|'t|'re|'ve|'m|'ll|'d
//   | [\p{L}]+ | [\p{N}] | [^\s\p{L}\p{N}]+
static std::vector<std::string> sam3_pretokenize(const std::string& text) {
    std::vector<std::string> tokens;
    size_t i = 0;
    const size_t n = text.size();

    while (i < n) {
        uint8_t c = (uint8_t)text[i];

        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            i++;
            continue;
        }

        if (i + 15 <= n && text.compare(i, 15, "<|startoftext|>") == 0) {
            tokens.push_back("<|startoftext|>");
            i += 15;
            continue;
        }
        if (i + 13 <= n && text.compare(i, 13, "<|endoftext|>") == 0) {
            tokens.push_back("<|endoftext|>");
            i += 13;
            continue;
        }

        // Must check contractions before letters since ' isn't a letter
        if (c == '\'') {
            if (i + 2 <= n) {
                char c2 = text[i + 1];
                if (c2 == 's' || c2 == 't' || c2 == 'm' || c2 == 'd') {
                    tokens.push_back(text.substr(i, 2));
                    i += 2;
                    continue;
                }
            }
            if (i + 3 <= n) {
                std::string c3 = text.substr(i + 1, 2);
                if (c3 == "re" || c3 == "ve" || c3 == "ll") {
                    tokens.push_back(text.substr(i, 3));
                    i += 3;
                    continue;
                }
            }
            // Fall through — not a contraction
        }

        if (sam3_is_letter(text, i)) {
            size_t start = i;
            while (i < n && sam3_is_letter(text, i)) {
                i += sam3_utf8_len((uint8_t)text[i]);
            }
            tokens.push_back(text.substr(start, i - start));
            continue;
        }

        if (c >= '0' && c <= '9') {
            tokens.push_back(text.substr(i, 1));
            i++;
            continue;
        }

        {
            size_t start = i;
            while (i < n) {
                uint8_t ch = (uint8_t)text[i];
                if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') break;
                if (sam3_is_letter(text, i)) break;
                if (ch >= '0' && ch <= '9') break;
                i++;
            }
            if (i > start) tokens.push_back(text.substr(start, i - start));
        }
    }

    return tokens;
}

/*
** ── sam3_tokenize — full pipeline ────────────────────────────────────────────
*/

// Tokenize text into a fixed-length token ID vector [ctx_len].
// Format: [SOT, bpe_tokens..., EOT, 0, 0, ..., 0]
std::vector<int32_t> sam3_tokenize(sam3_bpe_tokenizer& tok,
                                          const std::string& text,
                                          int ctx_len) {
    std::string lower;
    lower.reserve(text.size());
    for (char c : text) {
        lower += (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c;
    }

    std::string clean;
    clean.reserve(lower.size());
    bool last_ws = true;
    for (char c : lower) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!last_ws) {
                clean += ' ';
                last_ws = true;
            }
        } else {
            clean += c;
            last_ws = false;
        }
    }
    if (!clean.empty() && clean.back() == ' ') clean.pop_back();

    auto words = sam3_pretokenize(clean);

    std::vector<int32_t> ids;
    ids.push_back(tok.sot_token);

    for (const auto& word : words) {
        std::string encoded;
        for (uint8_t b : word) {
            auto it = tok.byte_encoder.find(b);
            if (it != tok.byte_encoder.end()) {
                encoded += it->second;
            }
        }

        std::string bpe_result = sam3_bpe_encode(tok, encoded);

        size_t start = 0;
        while (start < bpe_result.size()) {
            size_t end = bpe_result.find(' ', start);
            if (end == std::string::npos) end = bpe_result.size();
            std::string bpe_tok = bpe_result.substr(start, end - start);

            auto eit = tok.encoder.find(bpe_tok);
            if (eit != tok.encoder.end()) {
                ids.push_back(eit->second);
            }
            // Unknown tokens are silently dropped (matches CLIP behavior
            // where all byte sequences are in the vocab)

            start = end + 1;
        }
    }

    ids.push_back(tok.eot_token);

    if ((int)ids.size() > ctx_len) {
        ids.resize(ctx_len);
        ids.back() = tok.eot_token;
    }

    ids.resize(ctx_len, 0);

    return ids;
}

