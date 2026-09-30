// Portions of byte-level BPE handling are derived from Palm-Infra's Apache-2.0
// tokenizer implementation. See examples/internal/tokenizer.LICENSE. This
// header is modified and narrows the interface to the pinned Qwen3.6 profile.
#ifndef NCNN_MOE_EXAMPLES_INTERNAL_TOKENIZER_H
#define NCNN_MOE_EXAMPLES_INTERNAL_TOKENIZER_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

class Tokenizer final
{
public:
    Tokenizer();
    ~Tokenizer();
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    bool load(const std::string& model_directory, size_t vocabulary_size = 0);

    std::vector<int32_t> encode(std::string_view text) const;
    std::vector<int32_t> apply_chat(std::string_view messages_json, bool enable_thinking) const;
    std::string decode(int32_t id, std::string& pending, bool final = false) const;
    const std::vector<int32_t>& stop_tokens() const noexcept
    {
        return stops;
    }

private:
    struct UnicodeProfile;

    enum : uint8_t
    {
        Present = 1,
        Special = 2
    };

    struct Entry
    {
        uint32_t offset = 0;
        uint32_t length = 0;
        uint8_t flags = 0;
    };
    static_assert(sizeof(Entry) == 12, "token index layout changed");

    struct Merge
    {
        uint32_t rank = 0;
        uint32_t id = 0;
    };

    struct AddedToken
    {
        std::string content;
        uint32_t id = 0;
    };

    bool load_impl(const std::string& model_directory, size_t vocabulary_size);
    void bpe(std::string_view raw, std::vector<int32_t>& ids) const;

    std::vector<Entry> index;
    std::vector<uint8_t> bytes;
    std::unordered_map<uint64_t, Merge> merges;
    std::vector<AddedToken> added_tokens;
    std::vector<int32_t> stops;
    std::array<int32_t, 256> byte_ids{};
    std::unique_ptr<UnicodeProfile> unicode;
};

} // namespace moe
} // namespace ncnn

#endif // NCNN_MOE_EXAMPLES_INTERNAL_TOKENIZER_H
