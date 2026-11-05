#include "test_runtime_decls.h"
#include "test_runtime_support.h"

namespace ncnn {
namespace moe {

class TestExpertVictimCache final : public ExpertVictimCache
{
public:
    explicit TestExpertVictimCache(uint64_t _capacity)
        : cache_size(_capacity)
    {
    }

    void admit(std::string key, std::shared_ptr<const TensorData> gate_up, std::shared_ptr<const TensorData> down, ExpertVictimExecutionMetadata) override
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        ExpertVictimPair pair;
        pair.gate_up = std::make_shared<TensorData>(*gate_up);
        pair.down = std::make_shared<TensorData>(*down);
        entries_[std::move(key)] = std::move(pair);
        ++statistics_.admissions;
        ++statistics_.stores;
        statistics_.resident_size += gate_up->mxfp4_blocks.size() + gate_up->mxfp4_scales.size() + down->mxfp4_blocks.size() + down->mxfp4_scales.size();
    }

    void set_residency_coordinator(std::shared_ptr<ExpertResidencyCoordinator> coordinator) override
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        coordinator_ = std::move(coordinator);
    }

    std::shared_ptr<ExpertResidencyCoordinator> coordinator() const
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        return coordinator_;
    }

    void block_restore(std::string key)
    {
        const std::lock_guard<std::mutex> lock(barrier_mutex_);
        blocked_restore_key_ = std::move(key);
        blocked_restore_entered_ = false;
        blocked_restore_released_ = false;
    }

    bool wait_for_blocked_restore()
    {
        std::unique_lock<std::mutex> lock(barrier_mutex_);
        return barrier_entered_.wait_for(lock, std::chrono::seconds(5), [this] {
            return blocked_restore_entered_;
        });
    }

    void release_blocked_restore()
    {
        {
            const std::lock_guard<std::mutex> lock(barrier_mutex_);
            blocked_restore_released_ = true;
        }
        barrier_released_.notify_all();
    }

    std::optional<ExpertVictimPair> restore(const std::string& key, const TensorData&, const TensorData&) override
    {
        std::optional<ExpertVictimPair> result;
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            const auto existing = entries_.find(key);
            if (existing != entries_.end())
                result = existing->second;
            if (existing == entries_.end())
            {
                ++statistics_.misses;
                return std::nullopt;
            }
            ++statistics_.hits;
        }
        {
            std::unique_lock<std::mutex> lock(barrier_mutex_);
            if (key == blocked_restore_key_)
            {
                blocked_restore_entered_ = true;
                barrier_entered_.notify_all();
                barrier_released_.wait(lock, [this] {
                    return blocked_restore_released_;
                });
            }
        }
        return result;
    }

    void wait_for_background_work() override
    {
    }

    ExpertVictimCacheStatistics statistics() const override
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        return statistics_;
    }

    uint64_t capacity() const noexcept override
    {
        return cache_size;
    }

private:
    uint64_t cache_size = 0;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, ExpertVictimPair> entries_;
    std::shared_ptr<ExpertResidencyCoordinator> coordinator_;
    ExpertVictimCacheStatistics statistics_;
    std::mutex barrier_mutex_;
    std::condition_variable barrier_entered_;
    std::condition_variable barrier_released_;
    std::string blocked_restore_key_;
    bool blocked_restore_entered_ = false;
    bool blocked_restore_released_ = false;
};

class DelayedExpertVictimCache final : public ExpertVictimCache
{
public:
    void admit(std::string, std::shared_ptr<const TensorData>, std::shared_ptr<const TensorData>, ExpertVictimExecutionMetadata) override
    {
    }

    std::optional<ExpertVictimPair> restore(const std::string& key, const TensorData& gate_up_source, const TensorData& down_source) override
    {
        std::this_thread::sleep_for(key == "slow" ? std::chrono::milliseconds(100) : std::chrono::milliseconds(1));
        ExpertVictimPair pair;
        pair.gate_up = make_tensor(gate_up_source, 7);
        pair.down = make_tensor(down_source, 9);
        return pair;
    }

    void wait_for_background_work() override
    {
    }

    ExpertVictimCacheStatistics statistics() const override
    {
        return {};
    }

    uint64_t capacity() const noexcept override
    {
        return 68;
    }

private:
    static std::shared_ptr<TensorData> make_tensor(const TensorData& source, uint8_t value)
    {
        auto result = std::make_shared<TensorData>(source);
        result->mxfp4_file_storage.reset();
        result->mxfp4_blocks.resize(source.mxfp4_file_storage->blocks_size);
        std::fill_n(result->mxfp4_blocks.data(), result->mxfp4_blocks.size(), value);
        result->mxfp4_scales.resize(source.mxfp4_file_storage->scales_size);
        std::fill_n(result->mxfp4_scales.data(), result->mxfp4_scales.size(), uint8_t{127});
        return result;
    }
};

void test_adaptive_cache_wait_policy()
{
    // Delay cold weights beyond the adaptive Router's 2 ms admission threshold.
    const auto run = [](uint32_t flags) {
        CompiledModel model;
        model.descriptor.hidden_size = 32;
        model.descriptor.expert_count = 1;
        model.opt.flags = flags;
        model.opt.optimization_flags = g_test_optimization_flags;
        auto victim = std::make_shared<DelayedExpertVictimCache>();
        model.expert_cache = std::make_shared<ExpertCache>(1632, 1, victim, ExpertIoMode::Buffered);
        const auto add_weight = [&](const char* name, uint32_t rows) {
            TensorData tensor;
            tensor.dtype = DType::MxFp4;
            tensor.shape = {rows, 32};
            auto storage = std::make_shared<MxFp4FileStorage>();
            storage->blocks_size = rows * 16;
            storage->scales_size = rows;
            tensor.mxfp4_file_storage = std::move(storage);
            auto added = model.weights.add(name, std::move(tensor));
            check(static_cast<bool>(added));
            return added.value();
        };
        MoeBlockPlan moe;
        moe.top_k = 3;
        moe.experts.resize(1);
        ExpertPlan& expert = moe.experts.front();
        expert.layout = ExpertLayout::PackedGateUpDown;
        expert.activation = ExpertActivation::Silu;
        expert.gate_up_weight = add_weight("gate_up", 64);
        expert.down_weight = add_weight("down", 32);
        expert.weight_size = 1632;
        expert.cache_key = "slow";
        LayerState state;
        state.normalized.reset(1, 32, true);
        state.resize_experts(1);
        state.active_experts().front().batch.expert_id = 0;
        state.active_experts().front().batch.routes = {{0, 2, 1.0f}};
        ExpertScratch scratch;
        SessionStatistics statistics;
        check(static_cast<bool>(forward_moe(model, moe, state, &statistics, scratch, 0, ExecutionBackend::Cpu, false)));
        check(state.experts_executed);
        check(state.active_experts().front().output.rows() == 1);
        check(state.active_experts().front().output.columns() == 32);
        for (float value : state.active_experts().front().output.values())
            check(value == 0.0f);
        const bool use_adaptive_wait = has_flag(flags, OptionRouterPrediction)
                                       && has_flag(flags, OptionRankAdaptivePrefetch);
        check(statistics.expert_route_rank_demands[2] == (use_adaptive_wait ? 1 : 0));
        return statistics.expert_route_rank_demand_queue_time_microseconds[2];
    };
    check(run(OptionRouterPrediction | OptionRankAdaptivePrefetch) >= 2000);
    check(run(OptionRouterPrediction) == 0);
    check(run(OptionRankAdaptivePrefetch) == 0);
    check(run(0) == 0);
}

void test_sharded_expert_victim_cache()
{
    auto first = std::make_shared<TestExpertVictimCache>(34);
    auto second = std::make_shared<TestExpertVictimCache>(68);
    auto sharded = create_sharded_victim_cache({
        first,
        nullptr,
        second,
    });
    check(static_cast<bool>(sharded));
    check(static_cast<bool>(sharded->capacity() == 102));
    auto coordinator = std::make_shared<ExpertResidencyCoordinator>();
    auto decorated = create_reuse_victim_cache(sharded, 4);
    decorated->set_residency_coordinator(coordinator);
    check(first->coordinator() == coordinator && second->coordinator() == coordinator);
    decorated->set_residency_coordinator({});
    check(!first->coordinator() && !second->coordinator());
    {
        ExpertCache host(68, 1, decorated, ExpertIoMode::Buffered);
        host.set_residency_coordinator(coordinator);
        check(first->coordinator() == coordinator && second->coordinator() == coordinator);
        host.set_residency_coordinator({});
        check(!first->coordinator() && !second->coordinator());
    }

    auto gate_up = std::make_shared<TensorData>();
    gate_up->dtype = DType::MxFp4;
    gate_up->shape = {2, 32};
    gate_up->mxfp4_blocks.resize(32);
    std::fill_n(gate_up->mxfp4_blocks.data(), gate_up->mxfp4_blocks.size(), uint8_t{7});
    gate_up->mxfp4_scales.resize(2);
    std::fill_n(gate_up->mxfp4_scales.data(), gate_up->mxfp4_scales.size(), uint8_t{127});
    auto down = std::make_shared<TensorData>();
    down->dtype = DType::MxFp4;
    down->shape = {1, 1};
    down->mxfp4_blocks.resize(16);
    std::fill_n(down->mxfp4_blocks.data(), down->mxfp4_blocks.size(), uint8_t{9});
    down->mxfp4_scales.resize(1);
    std::fill_n(down->mxfp4_scales.data(), down->mxfp4_scales.size(), uint8_t{126});

    sharded->admit("layer.0.expert.1", gate_up, down);
    sharded->admit("layer.7.expert.3", gate_up, down);
    sharded->wait_for_background_work();
    auto restored_first = sharded->restore("layer.0.expert.1", *gate_up, *down);
    auto restored_second = sharded->restore("layer.7.expert.3", *gate_up, *down);
    check(static_cast<bool>(restored_first));
    check(static_cast<bool>(restored_second));
    check(static_cast<bool>(restored_first->gate_up->mxfp4_blocks.front() == 7));
    check(static_cast<bool>(restored_second->down->mxfp4_blocks.front() == 9));
    const ExpertVictimCacheStatistics statistics = sharded->statistics();
    check(static_cast<bool>(statistics.admissions == 2));
    check(static_cast<bool>(statistics.stores == 2));
    check(static_cast<bool>(statistics.hits == 2));

    auto filtered_inner = std::make_shared<TestExpertVictimCache>(1024);
    auto filtered = create_reuse_victim_cache(filtered_inner, 4);
    check(static_cast<bool>(filtered));
    for (uint32_t index = 0; index < 8; ++index)
    {
        filtered->admit("weak." + std::to_string(index), gate_up, down);
    }
    for (uint32_t index = 0; index < 2; ++index)
    {
        filtered->admit("weak." + std::to_string(index + 1), gate_up, down);
    }
    const ExpertVictimCacheStatistics filtered_statistics = filtered->statistics();
    check(static_cast<bool>(filtered_statistics.admissions == 4));
    check(static_cast<bool>(filtered_statistics.filtered_admissions == 6));
    check(static_cast<bool>(filtered_statistics.reused_admissions == 2));
    check(static_cast<bool>(filtered_statistics.probe_admissions == 2));
    check(static_cast<bool>(filtered->restore("weak.0", *gate_up, *down)));
    check(static_cast<bool>(filtered->restore("weak.1", *gate_up, *down)));
    check(static_cast<bool>(filtered->restore("weak.2", *gate_up, *down)));
    check(static_cast<bool>(!filtered->restore("weak.3", *gate_up, *down)));
}

void test_mapped_file_range_and_shared_buffer(bool low_fd)
{
    MxFp4ByteBuffer bytes = {1, 2, 3};
    check(bytes.size() == 3);
    check(bytes.front() == 1 && bytes.back() == 3);
    bytes = {bytes.back(), bytes.front()};
    check(bytes.size() == 2);
    check(bytes.front() == 3 && bytes.back() == 1);
    bytes = {};
    check(bytes.empty());
    check(bytes.data() == nullptr);

    const std::filesystem::path directory = create_unique_test_directory("ncnn_moe_mapped_file_test_");
    const std::filesystem::path path = directory / "range.bin";
    const size_t file_size = 3 * 4096 + 257;
    std::vector<uint8_t> expected(file_size);
    for (size_t index = 0; index < expected.size(); ++index)
        expected[index] = static_cast<uint8_t>((index * 29 + 7) % 251);
    {
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(expected.data()), static_cast<std::streamsize>(expected.size()));
    }

    {
        constexpr size_t offset = 123;
        constexpr size_t byte_count = 2 * 4096 + 91;
        auto mapped = MappedFileRange::open(path, offset, byte_count);
        check(static_cast<bool>(mapped));
        mapped.value()->prefault();
        check(static_cast<bool>(mapped.value()->size() == byte_count));
        check(static_cast<bool>(mapped.value()->data()[0] == expected[offset]));
        check(static_cast<bool>(mapped.value()->data()[byte_count - 1] == expected[offset + byte_count - 1]));

        MxFp4ByteBuffer shared = mapped.value()->share_bytes();
        check(static_cast<bool>(shared.size() == byte_count));
        check(static_cast<bool>(shared.front() == expected[offset]));
        check(static_cast<bool>(shared.back() == expected[offset + byte_count - 1]));
        MxFp4ByteBuffer copy = shared;
        copy.front() ^= 0xff;
        check(static_cast<bool>(shared.front() == expected[offset]));
        check(static_cast<bool>(copy.front() != shared.front()));

        auto shared_data = mapped.value()->share_data();
        const std::weak_ptr<MappedFileRange> mapped_lifetime = mapped.value();
        auto independent = MappedFileRange::open(path, offset + 1, byte_count - 2);
        check(static_cast<bool>(independent));
        check(independent.value().get() != mapped.value().get());
        auto independent_data = independent.value()->share_data();
        const std::weak_ptr<MappedFileRange> independent_lifetime = independent.value();

        mapped.value().reset();
        independent.value().reset();
        check(!mapped_lifetime.expired());
        check(!independent_lifetime.expired());
        check(shared.front() == expected[offset]);
        check(shared.back() == expected[offset + byte_count - 1]);
        check(shared_data.get()[0] == expected[offset]);
        check(shared_data.get()[byte_count - 1] == expected[offset + byte_count - 1]);

        shared_data.reset();
        check(!mapped_lifetime.expired());
        check(shared.front() == expected[offset]);
        check(shared.back() == expected[offset + byte_count - 1]);
        shared.resize(0);
        check(mapped_lifetime.expired());
        check(!independent_lifetime.expired());
        check(independent_data.get()[0] == expected[offset + 1]);
        check(independent_data.get()[byte_count - 3] == expected[offset + byte_count - 2]);
        independent_data.reset();
        check(independent_lifetime.expired());

        auto truncated = MappedFileRange::open(path, file_size - 8, 16);
        check(static_cast<bool>(!truncated));
        check(static_cast<bool>(truncated.error().code == ErrorCode::InvalidModel));
        auto empty = MappedFileRange::open(path, offset, 0);
        check(!empty);
        check(empty.error().code == ErrorCode::InvalidArgument);
        auto missing = MappedFileRange::open(directory / "missing.bin", 0, 1);
        check(!missing);
        check(missing.error().code == ErrorCode::IoError);
    }

    if (low_fd)
    {
        for (size_t index = 0; index < 24; ++index)
        {
            auto truncated = MappedFileRange::open(path, file_size - 8, 16);
            check(!truncated && truncated.error().code == ErrorCode::InvalidModel);
        }

        std::array<size_t, 32> offsets{};
        std::vector<std::shared_ptr<const uint8_t>> ranges;
        ranges.reserve(offsets.size());
        for (size_t index = 0; index < offsets.size(); ++index)
        {
            offsets[index] = 123 + index * 31;
            auto mapped = MappedFileRange::open(path, offsets[index], 128);
            check(static_cast<bool>(mapped));
            auto data = mapped.value()->share_data();
            mapped.value().reset();
            ranges.push_back(std::move(data));
        }
        for (size_t index = 0; index < ranges.size(); ++index)
        {
            check(ranges[index].get()[0] == expected[offsets[index]]);
            check(ranges[index].get()[127] == expected[offsets[index] + 127]);
        }
    }

    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
}

void test_safetensors_dense_mmap()
{
    const std::filesystem::path directory = create_unique_test_directory("ncnn_moe_safetensors_mmap_test_");
    const std::filesystem::path path = directory / "model.safetensors";
    std::string header = R"({"bf16":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"f32":{"dtype":"F32","shape":[2],"data_offsets":[4,12]},"bank":{"dtype":"BF16","shape":[2,1,1],"data_offsets":[12,16]},"bias_bank":{"dtype":"BF16","shape":[2,2],"data_offsets":[16,24]}})";
    while (header.size() % 8 != 0)
        header.push_back(' ');
    const uint64_t header_size = header.size();
    const std::array<uint16_t, 2> bfloat_values = {float_to_bfloat16(1.0f), float_to_bfloat16(-2.0f)};
    const std::array<float, 2> float_values = {3.5f, -2.25f};
    const std::array<uint16_t, 2> bank_values = {float_to_bfloat16(1.5f), float_to_bfloat16(-4.0f)};
    const std::array<uint16_t, 4> bias_bank_values = {
        float_to_bfloat16(2.5f),
        float_to_bfloat16(-6.0f),
        float_to_bfloat16(3.25f),
        float_to_bfloat16(-8.0f),
    };
    {
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
        stream.write(header.data(), static_cast<std::streamsize>(header.size()));
        stream.write(reinterpret_cast<const char*>(bfloat_values.data()), sizeof(bfloat_values));
        stream.write(reinterpret_cast<const char*>(float_values.data()), sizeof(float_values));
        stream.write(reinterpret_cast<const char*>(bank_values.data()), sizeof(bank_values));
        stream.write(reinterpret_cast<const char*>(bias_bank_values.data()), sizeof(bias_bank_values));
    }

    TensorData retained_bank_slice;
    TensorData retained_bias_slice;
    {
        auto archive = SafetensorsArchive::open(directory);
        check(static_cast<bool>(archive));
        auto bfloat = archive.value().load_tensor("bf16");
        check(static_cast<bool>(bfloat));
        check(static_cast<bool>(bfloat.value().mapped_data));
        check(static_cast<bool>(bfloat.value().bfloat16_data.empty()));
        check(static_cast<bool>(bfloat.value().bfloat16_values().size() == 2));
        check_near(bfloat16_to_float(bfloat.value().bfloat16_values()[1]), -2.0f, 0.0f);

        auto floating = archive.value().load_tensor("f32");
        check(static_cast<bool>(floating));
        check(static_cast<bool>(floating.value().mapped_data));
        check(static_cast<bool>(floating.value().float32_data.empty()));
        check(static_cast<bool>(floating.value().float32_values().size() == 2));
        check_near(floating.value().float32_values()[0], 3.5f, 0.0f);

        auto slice = archive.value().load_bfloat16_slice("bf16", 1, {1});
        check(static_cast<bool>(slice));
        check(static_cast<bool>(slice.value().mapped_data));
        check_near(bfloat16_to_float(slice.value().bfloat16_values()[0]), -2.0f, 0.0f);

        WeightMapping mapping;
        auto bank = add_bfloat16_expert_bank(mapping,
                                             archive.value(),
                                             "layers.0.",
                                             "gate_up.weight",
                                             "bank",
                                             2,
                                             {1, 1});
        check(static_cast<bool>(bank));
        check(mapping.size() == 2);
        const TensorData& first = mapping.at("layers.0.experts.0.gate_up.weight");
        const TensorData& second = mapping.at("layers.0.experts.1.gate_up.weight");
        check(first.dtype == DType::BFloat16);
        check(second.dtype == DType::BFloat16);
        check(first.shape == std::vector<uint32_t>{1, 1});
        check(second.shape == std::vector<uint32_t>{1, 1});
        check(first.mapped_size == sizeof(uint16_t));
        check(second.mapped_size == sizeof(uint16_t));
        check(static_cast<bool>(first.mapped_data));
        check(static_cast<bool>(second.mapped_data));
        check(second.mapped_data.get() == first.mapped_data.get() + sizeof(uint16_t));
        check(!first.mapped_data.owner_before(second.mapped_data)
              && !second.mapped_data.owner_before(first.mapped_data));
        check_near(bfloat16_to_float(first.bfloat16_values()[0]), 1.5f, 0.0f);
        check_near(bfloat16_to_float(second.bfloat16_values()[0]), -4.0f, 0.0f);
        retained_bank_slice = second;

        WeightMapping bias_mapping;
        auto bias_bank = add_bfloat16_expert_bank(bias_mapping,
                                                  archive.value(),
                                                  "layers.0.",
                                                  "bias",
                                                  "bias_bank",
                                                  2,
                                                  {2});
        check(static_cast<bool>(bias_bank));
        check(bias_mapping.size() == 2);
        const TensorData& first_bias = bias_mapping.at("layers.0.experts.0.bias");
        const TensorData& second_bias = bias_mapping.at("layers.0.experts.1.bias");
        check(first_bias.shape == std::vector<uint32_t>{2});
        check(second_bias.shape == std::vector<uint32_t>{2});
        check(first_bias.mapped_size == 2 * sizeof(uint16_t));
        check(second_bias.mapped_size == 2 * sizeof(uint16_t));
        check(static_cast<bool>(first_bias.mapped_data));
        check(static_cast<bool>(second_bias.mapped_data));
        check(second_bias.mapped_data.get() == first_bias.mapped_data.get() + 2 * sizeof(uint16_t));
        check(!first_bias.mapped_data.owner_before(second_bias.mapped_data)
              && !second_bias.mapped_data.owner_before(first_bias.mapped_data));
        check_near(bfloat16_to_float(first_bias.bfloat16_values()[0]), 2.5f, 0.0f);
        check_near(bfloat16_to_float(first_bias.bfloat16_values()[1]), -6.0f, 0.0f);
        check_near(bfloat16_to_float(second_bias.bfloat16_values()[0]), 3.25f, 0.0f);
        check_near(bfloat16_to_float(second_bias.bfloat16_values()[1]), -8.0f, 0.0f);
        retained_bias_slice = second_bias;

        const auto check_invalid_bank = [&](const char* source, uint32_t expert_count, std::vector<uint32_t> shape) {
            WeightMapping invalid_mapping;
            auto invalid_bank = add_bfloat16_expert_bank(invalid_mapping,
                                                         archive.value(),
                                                         "layers.0.",
                                                         "gate_up.weight",
                                                         source,
                                                         expert_count,
                                                         shape);
            check(!invalid_bank);
            check(invalid_mapping.empty());
        };
        check_invalid_bank("bank", 1, {2, 1});
        check_invalid_bank("bank", 1, {1, 1});
        check_invalid_bank("bank", 0, {1, 1});
        check_invalid_bank("bank", 2, {0, 1});
        check_invalid_bank("bank", 2, {1, 0});
        check_invalid_bank("bias_bank", 2, {1});
    }
    check(static_cast<bool>(retained_bank_slice.mapped_data));
    check(retained_bank_slice.shape == std::vector<uint32_t>{1, 1});
    check(retained_bank_slice.mapped_size == sizeof(uint16_t));
    check_near(bfloat16_to_float(retained_bank_slice.bfloat16_values()[0]),
               -4.0f,
               0.0f);
    check(static_cast<bool>(retained_bias_slice.mapped_data));
    check(retained_bias_slice.shape == std::vector<uint32_t>{2});
    check(retained_bias_slice.mapped_size == 2 * sizeof(uint16_t));
    check_near(bfloat16_to_float(retained_bias_slice.bfloat16_values()[0]),
               3.25f,
               0.0f);
    check_near(bfloat16_to_float(retained_bias_slice.bfloat16_values()[1]),
               -8.0f,
               0.0f);

    retained_bank_slice = {};
    retained_bias_slice = {};
    const std::filesystem::path unaligned_path = directory / "unaligned.safetensors";
    std::string unaligned_header = R"({"padding":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"bank":{"dtype":"BF16","shape":[2,1,1],"data_offsets":[1,5]},"bias_bank":{"dtype":"BF16","shape":[2,1],"data_offsets":[5,9]}})";
    while (unaligned_header.size() % 8 != 0)
        unaligned_header.push_back(' ');
    const uint64_t unaligned_header_size = unaligned_header.size();
    const uint8_t padding = 0;
    const std::array<uint16_t, 2> unaligned_bias_values = {
        float_to_bfloat16(5.5f),
        float_to_bfloat16(-10.0f),
    };
    {
        std::ofstream stream(unaligned_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(&unaligned_header_size),
                     sizeof(unaligned_header_size));
        stream.write(unaligned_header.data(),
                     static_cast<std::streamsize>(unaligned_header.size()));
        stream.write(reinterpret_cast<const char*>(&padding), sizeof(padding));
        stream.write(reinterpret_cast<const char*>(bank_values.data()), sizeof(bank_values));
        stream.write(reinterpret_cast<const char*>(unaligned_bias_values.data()), sizeof(unaligned_bias_values));
    }

    {
        auto archive = SafetensorsArchive::open_file(unaligned_path);
        check(static_cast<bool>(archive));
        WeightMapping mapping;
        auto bank = add_bfloat16_expert_bank(mapping,
                                             archive.value(),
                                             "speculative.layers.0.",
                                             "down.weight",
                                             "bank",
                                             2,
                                             {1, 1});
        check(static_cast<bool>(bank));
        check(mapping.size() == 2);
        const TensorData& first = mapping.at("speculative.layers.0.experts.0.down.weight");
        const TensorData& second = mapping.at("speculative.layers.0.experts.1.down.weight");
        check(first.shape == std::vector<uint32_t>{1, 1});
        check(second.shape == std::vector<uint32_t>{1, 1});
        check(!first.mapped_data);
        check(!second.mapped_data);
        check(first.bfloat16_data.size() == 1);
        check(second.bfloat16_data.size() == 1);
        check_near(bfloat16_to_float(first.bfloat16_values()[0]), 1.5f, 0.0f);
        check_near(bfloat16_to_float(second.bfloat16_values()[0]), -4.0f, 0.0f);

        WeightMapping bias_mapping;
        auto bias_bank = add_bfloat16_expert_bank(bias_mapping,
                                                  archive.value(),
                                                  "speculative.layers.0.",
                                                  "bias",
                                                  "bias_bank",
                                                  2,
                                                  {1});
        check(static_cast<bool>(bias_bank));
        check(bias_mapping.size() == 2);
        const TensorData& first_bias = bias_mapping.at("speculative.layers.0.experts.0.bias");
        const TensorData& second_bias = bias_mapping.at("speculative.layers.0.experts.1.bias");
        check(first_bias.shape == std::vector<uint32_t>{1});
        check(second_bias.shape == std::vector<uint32_t>{1});
        check(!first_bias.mapped_data);
        check(!second_bias.mapped_data);
        check(first_bias.bfloat16_data.size() == 1);
        check(second_bias.bfloat16_data.size() == 1);
        check_near(bfloat16_to_float(first_bias.bfloat16_values()[0]), 5.5f, 0.0f);
        check_near(bfloat16_to_float(second_bias.bfloat16_values()[0]), -10.0f, 0.0f);
    }

    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
}

void test_safetensors_packed_mxfp4_expert()
{
    const std::filesystem::path directory = create_unique_test_directory("ncnn_moe_safetensors_packed_test_");
    const auto write_archive = [](const std::filesystem::path& path, std::string header, const std::vector<uint8_t>& data) {
        while (header.size() % 8 != 0)
            header.push_back(' ');
        const uint64_t header_size = header.size();
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
        stream.write(header.data(), static_cast<std::streamsize>(header.size()));
        stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    };
    write_archive(directory / "model.safetensors", R"({"gate_blocks":{"dtype":"U8","shape":[1,1,1,16],"data_offsets":[0,16]},"gate_scales":{"dtype":"U8","shape":[1,1,1],"data_offsets":[16,17]}})", std::vector<uint8_t>(17, 3));
    std::vector<uint8_t> packed(17, 9);
    packed.back() = 123;
    write_archive(directory / "ncnn-moe-packed-experts.safetensors",
                  R"({"__ncnn_moe_packed__.0.gate_blocks":{"dtype":"U8","shape":[1,1,16],"data_offsets":[0,16]},"__ncnn_moe_packed__.0.gate_scales":{"dtype":"U8","shape":[1,1],"data_offsets":[16,17]}})", packed);

    auto archive = SafetensorsArchive::open(directory);
    check(static_cast<bool>(archive));
    auto expert = archive.value().load_mxfp4_expert("gate_blocks", "gate_scales", 0, 1, 32, SafetensorLoadDeferMxfp4Data);
    check(static_cast<bool>(expert));
    check(static_cast<bool>(expert.value().mxfp4_file_storage));
    const MxFp4FileStorage& storage = *expert.value().mxfp4_file_storage;
    check(static_cast<bool>(storage.blocks_path == storage.scales_path));
    check(static_cast<bool>(storage.blocks_offset + storage.blocks_size == storage.scales_offset));

    ExpertCache cache(34, 1, {}, ExpertIoMode::Buffered);
    auto pair = cache.acquire_pair(expert.value(), expert.value());
    check(static_cast<bool>(pair));
    check(static_cast<bool>(pair.value().gate_up->mxfp4_blocks.front() == 9));
    check(static_cast<bool>(pair.value().gate_up->mxfp4_scales.front() == 123));

    const std::filesystem::path matrix_path = directory / "matrix_payloads.safetensors";
    write_archive(matrix_path, R"({
        "blocks":{"dtype":"I8","shape":[1,16],"data_offsets":[0,16]},
        "scales":{"dtype":"F8_E8M0","shape":[1,1],"data_offsets":[16,17]},
        "expert_blocks":{"dtype":"U8","shape":[2,1,1,16],"data_offsets":[17,49]},
        "expert_scales":{"dtype":"U8","shape":[2,1,1],"data_offsets":[49,51]}
    })",
                  std::vector<uint8_t>(51, 3));
    auto matrix_archive = SafetensorsArchive::open_file(matrix_path);
    check(static_cast<bool>(matrix_archive));
    const auto check_invalid = [](const Result<TensorData>& result) {
        check(!result);
        check(result.error().code == ErrorCode::InvalidModel);
    };
    for (uint32_t flags : std::array<uint32_t, 2>{0, SafetensorLoadDeferMxfp4Data})
    {
        auto tensor = matrix_archive.value().load_mxfp4_tensor("blocks", "scales", 1, 32, flags);
        check(static_cast<bool>(tensor));
        auto interleaved = matrix_archive.value().load_mxfp4("blocks", "scales", "blocks", "scales", 1, 32, flags);
        check(static_cast<bool>(interleaved));
        check(interleaved.value().shape == std::vector<uint32_t>({2, 32}));
        check_invalid(matrix_archive.value().load_mxfp4_tensor("blocks", "scales", 2, 32, flags));
        check_invalid(matrix_archive.value().load_mxfp4("blocks", "scales", "blocks", "scales", 2, 32, flags));
        check_invalid(matrix_archive.value().load_mxfp4_expert("expert_blocks", "expert_scales", 2, 1, 32, flags));
        check_invalid(matrix_archive.value().load_mxfp4("blocks", "scales", "blocks", "scales", std::numeric_limits<uint32_t>::max() / 2 + 1, 32, flags));
    }

    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
}

void test_safetensors_qnk_source_binding()
{
    const std::filesystem::path directory = create_unique_test_directory("ncnn_moe_safetensors_qnk_packed_test_");
    const auto write_archive = [](const std::filesystem::path& path, std::string header, const std::vector<uint8_t>& data) {
        while (header.size() % 8 != 0)
            header.push_back(' ');
        const uint64_t header_size = header.size();
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
        stream.write(header.data(), static_cast<std::streamsize>(header.size()));
        stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    };
    constexpr uint32_t expert_count = 2;
    constexpr uint32_t rows = 2;
    constexpr uint32_t columns = 256;
    const size_t expert_bytes = static_cast<size_t>(qnk_storage_bytes(DType::Q4K, rows, columns));
    std::vector<uint8_t> source(expert_count * expert_bytes + expert_bytes, 3);
    write_archive(directory / "model.safetensors",
                  R"({"experts":{"dtype":"U8","shape":[2,2,1,144],"data_offsets":[0,576]},"matrix":{"dtype":"U8","shape":[2,1,144],"data_offsets":[576,864]}})",
                  source);
    std::vector<uint8_t> packed(expert_bytes * 2, 91);
    write_archive(directory / "ncnn-moe-packed-qnk.safetensors",
                  R"({"__ncnn_moe_packed__.1.experts":{"dtype":"U8","shape":[2,1,144],"data_offsets":[0,288]},"__ncnn_moe_packed__.matrix":{"dtype":"U8","shape":[2,1,144],"data_offsets":[288,576]}})",
                  packed);

    auto archive = SafetensorsArchive::open(directory);
    check(static_cast<bool>(archive));
    const auto detected_dtype = archive.value().find_qnk_expert_dtype("experts",
                                                                      expert_count,
                                                                      rows,
                                                                      columns);
    check(static_cast<bool>(detected_dtype && detected_dtype.value() == DType::Q4K));
    auto expert = archive.value().load_qnk_expert("experts", DType::Q4K, 1, expert_count, rows, columns);
    check(static_cast<bool>(expert));
    check(expert.value().dtype == DType::Q4K);
    check(expert.value().shape == std::vector<uint32_t>{rows, columns});
    check(expert.value().qnk_values().size() == expert_bytes);
    check(expert.value().qnk_values().front() == 3);
    check(expert.value().qnk_values().back() == 3);
    auto matrix = archive.value().load_qnk_tensor("matrix", DType::Q4K, rows, columns);
    check(static_cast<bool>(matrix));
    check(matrix.value().qnk_values().size() == expert_bytes);
    check(matrix.value().qnk_values().front() == 3);

    const std::filesystem::path sidecar_only_directory = create_unique_test_directory("ncnn_moe_safetensors_qnk_sidecar_only_test_");
    write_archive(sidecar_only_directory / "ncnn-moe-packed-qnk.safetensors",
                  R"({"__ncnn_moe_packed__.0.experts":{"dtype":"U8","shape":[2,1,144],"data_offsets":[0,288]},"__ncnn_moe_packed__.1.experts":{"dtype":"U8","shape":[2,1,144],"data_offsets":[288,576]}})",
                  packed);
    auto sidecar_only = SafetensorsArchive::open(sidecar_only_directory);
    check(static_cast<bool>(sidecar_only));
    check(!sidecar_only.value().find_qnk_expert_dtype("experts", expert_count, rows, columns));
    check(!sidecar_only.value().load_qnk_expert("experts", DType::Q4K, 0, expert_count, rows, columns));

    const std::filesystem::path bfloat16_directory = create_unique_test_directory("ncnn_moe_safetensors_qnk_bfloat16_sidecar_test_");
    write_archive(bfloat16_directory / "model.safetensors",
                  R"({"experts":{"dtype":"BF16","shape":[2,2,256],"data_offsets":[0,2048]}})",
                  std::vector<uint8_t>(2048, 7));
    write_archive(bfloat16_directory / "ncnn-moe-packed-qnk.safetensors",
                  R"({"__ncnn_moe_packed__.0.experts":{"dtype":"U8","shape":[2,1,144],"data_offsets":[0,288]},"__ncnn_moe_packed__.1.experts":{"dtype":"U8","shape":[2,1,144],"data_offsets":[288,576]}})",
                  packed);
    auto bfloat16_with_sidecar = SafetensorsArchive::open(bfloat16_directory);
    check(static_cast<bool>(bfloat16_with_sidecar));
    check(!bfloat16_with_sidecar.value().find_qnk_expert_dtype("experts", expert_count, rows, columns));
    check(!bfloat16_with_sidecar.value().load_qnk_expert("experts", DType::Q4K, 0, expert_count, rows, columns));

    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    std::filesystem::remove_all(sidecar_only_directory, ignored);
    std::filesystem::remove_all(bfloat16_directory, ignored);
}

void test_file_backed_bfloat16_expert_cache()
{
    const auto mapped_bfloat16 = [](std::vector<uint32_t> shape,
                                    std::initializer_list<float> source_values) {
        auto storage = std::make_shared<std::vector<uint16_t>>();
        storage->reserve(source_values.size());
        for (float value : source_values)
            storage->push_back(float_to_bfloat16(value));

        TensorData tensor;
        tensor.dtype = DType::BFloat16;
        tensor.shape = std::move(shape);
        tensor.mapped_data = std::shared_ptr<const uint8_t>(storage,
                                                            reinterpret_cast<const uint8_t*>(storage->data()));
        tensor.mapped_size = storage->size() * sizeof(uint16_t);
        return tensor;
    };

    const TensorData gate_up = mapped_bfloat16({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const TensorData down = mapped_bfloat16({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    constexpr uint64_t pair_size = 16;

    ExpertCache mapped_cache(pair_size, 0, {}, ExpertIoMode::Mmap);
    auto mapped = mapped_cache.acquire_pair(gate_up, down);
    check(static_cast<bool>(mapped));
    check(static_cast<bool>(mapped.value().gate_up->mapped_data));
    check(mapped.value().gate_up->bfloat16_data.empty());
    check_near(bfloat16_to_float(mapped.value().gate_up->bfloat16_values()[2]),
               3.0f, 0.0f);
    const std::string key = ExpertCache::make_pair_key(gate_up, down);
    check(mapped_cache.is_ready(gate_up, down, key));
    auto hit = mapped_cache.acquire_pair(gate_up, down, std::numeric_limits<uint32_t>::max(), key);
    check(static_cast<bool>(hit));
    const ExpertCacheStatistics mapped_statistics = mapped_cache.statistics();
    check(mapped_statistics.misses == 1);
    check(mapped_statistics.hits == 1);
    check(mapped_statistics.bytes_read == pair_size);
    check(mapped_statistics.resident_size == pair_size);
    check(mapped_statistics.mapped_ranges == 2);
    check(mapped_statistics.mapped_bytes == pair_size);

    ExpertCache copied_cache(pair_size);
    auto copied = copied_cache.acquire_pair(gate_up, down);
    check(static_cast<bool>(copied));
    check(!copied.value().gate_up->mapped_data);
    check(copied.value().gate_up->bfloat16_data.size() == 4);
    check_near(bfloat16_to_float(copied.value().down->bfloat16_values()[3]),
               8.0f, 0.0f);

    const TensorData gate_up_two = mapped_bfloat16({2, 2}, {9.0f, 10.0f, 11.0f, 12.0f});
    const TensorData down_two = mapped_bfloat16({2, 2}, {13.0f, 14.0f, 15.0f, 16.0f});
    const TensorData gate_up_three = mapped_bfloat16({2, 2}, {17.0f, 18.0f, 19.0f, 20.0f});
    const TensorData down_three = mapped_bfloat16({2, 2}, {21.0f, 22.0f, 23.0f, 24.0f});
    const std::array<std::string, 3> request_keys = {
        ExpertCache::make_pair_key(gate_up, down),
        ExpertCache::make_pair_key(gate_up_two, down_two),
        ExpertCache::make_pair_key(gate_up_three, down_three),
    };
    const std::array<ExpertCachePairRequest, 3> requests = {{
        {&gate_up, &down, 0, request_keys[0]},
        {&gate_up_two, &down_two, 0, request_keys[1]},
        {&gate_up_three, &down_three, 0, request_keys[2]},
    }};
    // Read-ahead must preserve a bounded prefix even when these tiny reads
    // complete during submission, and must never wait for an active lease.
    ExpertCache read_ahead_cache(pair_size * 2);
    auto requested = read_ahead_cache.request_pairs(requests);
    check(static_cast<bool>(requested));
    check(requested.value() == 2);
    read_ahead_cache.wait_for_background_work();
    check(read_ahead_cache.is_ready(gate_up, down, requests[0].prepared_key));
    check(read_ahead_cache.is_ready(gate_up_two, down_two, requests[1].prepared_key));
    check(!read_ahead_cache.is_ready(gate_up_three, down_three, requests[2].prepared_key));
    check(read_ahead_cache.statistics().queued_reads == 2);
    check(read_ahead_cache.statistics().hits == 0);
    std::array<ExpertCacheLease, 2> pinned;
    auto pinned_ready = read_ahead_cache.try_acquire_ready_pairs(std::span(requests).first(2), pinned);
    check(static_cast<bool>(pinned_ready) && pinned_ready.value());
    requested = read_ahead_cache.request_pairs(std::span(requests).last(1));
    check(static_cast<bool>(requested));
    check(requested.value() == 0);
    check(read_ahead_cache.statistics().queued_reads == 2);
    pinned = {};
    requested = read_ahead_cache.request_pairs(std::span(requests).last(1));
    check(static_cast<bool>(requested));
    check(requested.value() == 1);
    read_ahead_cache.wait_for_background_work();
    check(read_ahead_cache.is_ready(gate_up_three, down_three, requests[2].prepared_key));
    check(read_ahead_cache.statistics().resident_size <= pair_size * 2);
    requested = read_ahead_cache.request_pairs({});
    check(static_cast<bool>(requested) && requested.value() == 0);
    ExpertCache invalid_read_ahead(pair_size * 2);
    auto invalid_requests = requests;
    invalid_requests[2].down = nullptr;
    check(!invalid_read_ahead.request_pairs(invalid_requests));
    check(invalid_read_ahead.statistics().queued_reads == 0);

    ExpertCache bounded_cache(pair_size * 2);
    std::array<uint8_t, 3> acquired_pairs{};
    size_t acquired_count = 0;
    while (acquired_count != requests.size())
    {
        std::array<ExpertCachePairRequest, 3> pending_requests{};
        std::array<size_t, 3> pending_indices{};
        size_t pending_count = 0;
        for (size_t index = 0; index < requests.size(); ++index)
        {
            if (acquired_pairs[index] != 0)
                continue;
            pending_requests[pending_count] = requests[index];
            pending_indices[pending_count] = index;
            ++pending_count;
        }
        std::array<ExpertCacheLease, 3> leases;
        auto ready = bounded_cache.wait_acquire_ready_pairs(std::span<const ExpertCachePairRequest>(pending_requests.data(), pending_count),
                                                            std::span<ExpertCacheLease>(leases.data(), pending_count), true, 8);
        check(static_cast<bool>(ready));
        // The requested target exceeds this tiny cache's admitted prefix.
        // Wait for that prefix only, release it, then admit the remaining pair.
        check(ready.value() == std::min<size_t>(pending_count, 2));
        for (size_t index = 0; index < pending_count; ++index)
        {
            if (!leases[index].gate_up)
                continue;
            acquired_pairs[pending_indices[index]] = 1;
            ++acquired_count;
        }
    }
    check(bounded_cache.statistics().resident_size <= pair_size * 2);
}

void test_expert_residency_coordinator()
{
    auto coordinator = std::make_shared<ExpertResidencyCoordinator>();
    check(!coordinator->register_host({}));
    check(!coordinator->register_device({}));
    auto host = coordinator->register_host("pair");
    auto device_zero = coordinator->register_device("pair");
    auto device_one = coordinator->register_device("pair");
    check(host && device_zero && device_one);
    check(coordinator->host_resident("pair") && coordinator->device_resident("pair"));
    auto copied_device = device_zero;
    device_zero.reset();
    check(coordinator->device_resident("pair"));
    copied_device.reset();
    check(coordinator->device_resident("pair"));
    host.reset();
    check(!coordinator->host_resident("pair") && coordinator->device_resident("pair"));
    device_one.reset();
    check(!coordinator->host_resident("pair") && !coordinator->device_resident("pair"));

    // Tokens outlive the cache/backend owner without retaining either cache
    // or calling into a destroyed object when a submission releases its pins.
    auto orphan_owner = std::make_shared<ExpertResidencyCoordinator>();
    auto orphan = orphan_owner->register_device("orphan");
    const std::weak_ptr<ExpertResidencyCoordinator> weak_owner = orphan_owner;
    orphan_owner.reset();
    check(weak_owner.expired());
    orphan.reset();

    std::atomic<bool> consistent{true};
    std::array<std::thread, 4> workers;
    for (size_t index = 0; index < workers.size(); ++index)
    {
        workers[index] = std::thread([&] {
            for (size_t iteration = 0; iteration < 256; ++iteration)
            {
                auto host_copy = coordinator->register_host("concurrent");
                auto device_copy = coordinator->register_device("concurrent");
                if (!host_copy || !device_copy
                    || !coordinator->host_resident("concurrent")
                    || !coordinator->device_resident("concurrent"))
                    consistent.store(false, std::memory_order_relaxed);
            }
        });
    }
    for (std::thread& worker : workers)
        worker.join();
    check(consistent.load(std::memory_order_relaxed));
    check(!coordinator->host_resident("concurrent") && !coordinator->device_resident("concurrent"));
}

void test_file_backed_qnk_expert_cache()
{
    constexpr uint32_t columns = 256;
    const std::array<DType, 6> dtypes = {
        DType::Q2K, DType::Q3K, DType::Q4K, DType::Q5K, DType::Q6K, DType::Q8K};
    const std::filesystem::path directory = create_unique_test_directory("ncnn_moe_qnk_expert_cache_test_");
    const auto write_file = [](const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    };
    const auto append_range = [](std::vector<uint8_t>& bank, size_t size, uint8_t seed) {
        const uint64_t offset = bank.size();
        for (size_t index = 0; index < size; ++index)
            bank.push_back(static_cast<uint8_t>(seed + index * 13));
        return std::pair<uint64_t, std::vector<uint8_t>>{
            offset,
            std::vector<uint8_t>(bank.begin() + static_cast<std::ptrdiff_t>(offset), bank.end())};
    };
    const auto file_backed = [columns](DType dtype,
                                       uint32_t rows,
                                       const std::filesystem::path& path,
                                       uint64_t offset,
                                       uint64_t size,
                                       uint64_t secondary_offset = 0,
                                       uint64_t secondary_size = 0) {
        TensorData tensor;
        tensor.dtype = dtype;
        tensor.shape = {rows, columns};
        tensor.qnk_interleave_rows = false;
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->blocks_path = path.string();
        storage->blocks_offset = offset;
        storage->blocks_size = size;
        if (secondary_size != 0)
        {
            storage->secondary_blocks_path = path.string();
            storage->secondary_blocks_offset = secondary_offset;
            storage->secondary_blocks_size = secondary_size;
        }
        tensor.mxfp4_file_storage = std::move(storage);
        return tensor;
    };

    TensorData pinned_gate;
    TensorData pinned_down;
    TensorData reload_gate;
    TensorData reload_down;
    uint64_t pinned_capacity = 0;
    uint64_t pinned_raw_size = 0;
    for (size_t index = 0; index < dtypes.size(); ++index)
    {
        const DType gate_dtype = dtypes[index];
        const DType down_dtype = dtypes[(index + 1) % dtypes.size()];
        const std::filesystem::path path = directory / ("qnk_" + std::to_string(index) + ".bin");
        std::vector<uint8_t> bank;
        const uint64_t gate_half_size = qnk_storage_bytes(gate_dtype, 4, columns);
        const uint64_t gate_size = qnk_storage_bytes(gate_dtype, 8, columns);
        const uint64_t down_size = qnk_storage_bytes(down_dtype, 4, columns);
        auto gate_first = append_range(bank, static_cast<size_t>(gate_half_size), static_cast<uint8_t>(11 + index));
        auto gate_second = append_range(bank, static_cast<size_t>(gate_half_size), static_cast<uint8_t>(31 + index));
        auto down_first = append_range(bank, static_cast<size_t>(down_size), static_cast<uint8_t>(51 + index));
        auto gate_reload = append_range(bank, static_cast<size_t>(gate_size), static_cast<uint8_t>(71 + index));
        auto down_reload = append_range(bank, static_cast<size_t>(down_size), static_cast<uint8_t>(91 + index));
        write_file(path, bank);

        TensorData gate = file_backed(gate_dtype, 8, path, gate_first.first, gate_half_size,
                                      gate_second.first, gate_half_size);
        TensorData down = file_backed(down_dtype, 4, path, down_first.first, down_size);
        TensorData gate_next = file_backed(gate_dtype, 8, path, gate_reload.first, gate_size);
        TensorData down_next = file_backed(down_dtype, 4, path, down_reload.first, down_size);
        for (uint32_t column = 0; column < columns; ++column)
            gate.input_column_indices.push_back(columns - column - 1);
        gate_next.input_column_indices = gate.input_column_indices;
        down.qnk_interleave_rows = true;
        down_next.qnk_interleave_rows = true;

        check(gate.quantized_data.empty() && !gate.mapped_data);
        check(down.quantized_data.empty() && !down.mapped_data);
        check(ExpertCache::make_pair_key(gate, down) != ExpertCache::make_pair_key(gate_next, down_next));
        TensorData changed_map = gate;
        std::swap(changed_map.input_column_indices[0], changed_map.input_column_indices[1]);
        check(ExpertCache::make_pair_key(gate, down) != ExpertCache::make_pair_key(changed_map, down));
        const std::string map_bytes(reinterpret_cast<const char*>(gate.input_column_indices.data()),
                                    gate.input_column_indices.size() * sizeof(uint32_t));
        check(ExpertCache::make_pair_key(gate, down).find(map_bytes) != std::string::npos);
        TensorData changed_dtype = gate;
        changed_dtype.dtype = down_dtype;
        check(ExpertCache::make_pair_key(gate, down) != ExpertCache::make_pair_key(changed_dtype, down));
        TensorData changed_secondary = gate;
        auto changed_storage = std::make_shared<MxFp4FileStorage>(*gate.mxfp4_file_storage);
        ++changed_storage->secondary_blocks_offset;
        changed_secondary.mxfp4_file_storage = std::move(changed_storage);
        check(ExpertCache::make_pair_key(gate, down) != ExpertCache::make_pair_key(changed_secondary, down));

        const uint64_t raw_pair_size = gate_size + down_size;
        const uint64_t metadata_size = static_cast<uint64_t>(gate.input_column_indices.size()) * sizeof(uint32_t);
        const uint64_t resident_pair_size = raw_pair_size + metadata_size * 2;
        for (const ExpertIoMode mode : {ExpertIoMode::Buffered, ExpertIoMode::Mmap, ExpertIoMode::Direct})
        {
            ExpertCache cache(resident_pair_size, 1, {}, mode);
            auto pair = cache.acquire_pair(gate, down);
            check(static_cast<bool>(pair));
            const auto gate_values = pair.value().gate_up->qnk_values();
            const auto down_values = pair.value().down->qnk_values();
            check(gate_values.size() == gate_size);
            check(down_values.size() == down_size);
            check(std::equal(gate_first.second.begin(), gate_first.second.end(), gate_values.begin()));
            check(std::equal(gate_second.second.begin(), gate_second.second.end(), gate_values.begin() + gate_first.second.size()));
            check(std::equal(down_first.second.begin(), down_first.second.end(), down_values.begin()));
            check(pair.value().gate_up->dtype == gate_dtype && pair.value().down->dtype == down_dtype);
            check(pair.value().gate_up->qnk_interleave_rows == gate.qnk_interleave_rows);
            check(pair.value().down->qnk_interleave_rows == down.qnk_interleave_rows);
            check(pair.value().gate_up->input_column_indices == gate.input_column_indices);
            check(pair.value().gate_up->quantized_data.empty() && pair.value().gate_up->mapped_data);
            check(pair.value().down->quantized_data.empty() && pair.value().down->mapped_data);
            const ExpertCacheStatistics stats = cache.statistics();
            check(stats.bytes_read == raw_pair_size);
            check(stats.resident_size == resident_pair_size);
            check(stats.mapped_ranges <= 1);
            check(stats.mapped_bytes + stats.buffered_read_bytes + stats.direct_read_bytes == raw_pair_size);
        }

        if (gate_dtype == DType::Q4K)
        {
            TensorData mxfp4_gate = gate;
            mxfp4_gate.dtype = DType::MxFp4;
            ExpertCache mixed_gate_cache(1024 * 1024, 1, {}, ExpertIoMode::Buffered);
            auto rejected_mixed_gate = mixed_gate_cache.request_pair(mxfp4_gate, down);
            check(!rejected_mixed_gate && rejected_mixed_gate.error().code == ErrorCode::InvalidArgument);
            check(mixed_gate_cache.statistics().bytes_read == 0);

            TensorData mxfp4_down = down;
            mxfp4_down.dtype = DType::MxFp4;
            ExpertCache mixed_down_cache(1024 * 1024, 1, {}, ExpertIoMode::Buffered);
            auto rejected_mixed_down = mixed_down_cache.request_pair(gate, mxfp4_down);
            check(!rejected_mixed_down && rejected_mixed_down.error().code == ErrorCode::InvalidArgument);
            check(mixed_down_cache.statistics().bytes_read == 0);

            pinned_gate = gate;
            pinned_down = down;
            reload_gate = gate_next;
            reload_down = down_next;
            pinned_capacity = resident_pair_size;
            pinned_raw_size = raw_pair_size;

            ExpertCache coalescing_cache(resident_pair_size * 2,
                                         1,
                                         {},
                                         ExpertIoMode::Buffered,
                                         ExpertCacheCrossExpertReadCoalescing);
            const std::string gate_key = ExpertCache::make_pair_key(gate, down);
            const std::string reload_key = ExpertCache::make_pair_key(gate_next, down_next);
            const std::array<ExpertCachePairRequest, 2> batch = {{
                {&gate, &down, 0, gate_key},
                {&gate_next, &down_next, 0, reload_key},
            }};
            auto queued = coalescing_cache.request_pairs(batch);
            check(queued && queued.value() == batch.size());
            coalescing_cache.wait_for_background_work();
            auto first_loaded = coalescing_cache.acquire_pair(gate, down);
            auto second_loaded = coalescing_cache.acquire_pair(gate_next, down_next);
            check(first_loaded && second_loaded);
            check(first_loaded.value().gate_up->qnk_values().front() == gate_first.second.front());
            check(second_loaded.value().gate_up->qnk_values().front() == gate_reload.second.front());
            check(coalescing_cache.statistics().bytes_read == raw_pair_size * 2);
            check(coalescing_cache.statistics().coalesced_read_batches == 0);
        }
    }

    {
        ExpertCache pin_cache(pinned_capacity, 1, {}, ExpertIoMode::Buffered);
        auto first = pin_cache.acquire_pair(pinned_gate, pinned_down);
        check(static_cast<bool>(first));
        std::shared_ptr<const TensorData> retained = first.value().gate_up;
        first.value() = {};
        const std::string reload_key = ExpertCache::make_pair_key(reload_gate, reload_down);
        const std::array<ExpertCachePairRequest, 1> reload_request = {{
            {&reload_gate, &reload_down, 0, reload_key},
        }};
        std::array<ExpertCacheLease, 1> reload_lease;
        auto blocked = pin_cache.wait_acquire_ready_pairs(reload_request, reload_lease, true, 1, true, false);
        check(blocked && blocked.value() == 0);
        check(retained && retained->qnk_values().size() == qnk_storage_bytes(DType::Q4K, 8, columns));
        retained.reset();
        auto reloaded = pin_cache.wait_acquire_ready_pairs(reload_request, reload_lease);
        check(reloaded && reloaded.value() == 1);
        check(reload_lease[0].gate_up->qnk_values().front() == 71 + 2);
        reload_lease[0] = {};
        auto restored = pin_cache.acquire_pair(pinned_gate, pinned_down);
        check(static_cast<bool>(restored));
        check(restored.value().gate_up->qnk_values().front() == 11 + 2);
        const ExpertCacheStatistics pin_stats = pin_cache.statistics();
        check(pin_stats.bytes_read == pinned_raw_size * 3);
        check(pin_stats.resident_size == pinned_capacity);
        check(pin_stats.evictions == 2);

        const uint64_t raw_size = qnk_storage_bytes(DType::Q4K, 8, columns)
                                  + qnk_storage_bytes(DType::Q5K, 4, columns);
        const uint64_t metadata_size = static_cast<uint64_t>(pinned_gate.input_column_indices.size()) * sizeof(uint32_t);
        const uint64_t packed_size = qnk_packed_storage_bytes(DType::Q4K, 8, columns)
                                     + qnk_packed_storage_bytes(DType::Q5K, 4, columns);
        const uint64_t reserved_size = raw_size + metadata_size * 2 + packed_size;
        ExpertCache packed_too_small(reserved_size - 1, 1, {}, ExpertIoMode::Buffered, 0, 1, true);
        auto rejected_packed = packed_too_small.acquire_pair(pinned_gate, pinned_down);
        check(!rejected_packed && rejected_packed.error().code == ErrorCode::InvalidArgument);
        check(packed_too_small.statistics().bytes_read == 0);
        ExpertCache packed_exact(reserved_size, 1, {}, ExpertIoMode::Buffered, 0, 1, true);
        auto packed = packed_exact.acquire_pair(pinned_gate, pinned_down);
        check(static_cast<bool>(packed));
        check(packed_exact.statistics().resident_size == reserved_size);
        check(packed_exact.statistics().bytes_read == raw_size);

        const auto expect_invalid = [&](const TensorData& gate) {
            ExpertCache cache(1024 * 1024, 1, {}, ExpertIoMode::Buffered);
            auto request = cache.request_pair(gate, pinned_down);
            check(!request && request.error().code == ErrorCode::InvalidModel);
            check(cache.statistics().bytes_read == 0);
        };
        TensorData invalid = pinned_gate;
        invalid.quantized_data.push_back(1);
        expect_invalid(invalid);
        invalid = pinned_gate;
        auto invalid_storage = std::make_shared<MxFp4FileStorage>(*pinned_gate.mxfp4_file_storage);
        invalid_storage->scales_path = "unexpected-scale-range";
        invalid_storage->scales_size = 1;
        invalid.mxfp4_file_storage = std::move(invalid_storage);
        expect_invalid(invalid);
        invalid = pinned_gate;
        invalid_storage = std::make_shared<MxFp4FileStorage>(*pinned_gate.mxfp4_file_storage);
        invalid_storage->blocks_offset = std::numeric_limits<uint64_t>::max() - invalid_storage->blocks_size + 1;
        invalid.mxfp4_file_storage = std::move(invalid_storage);
        expect_invalid(invalid);
        invalid = pinned_gate;
        invalid_storage = std::make_shared<MxFp4FileStorage>(*pinned_gate.mxfp4_file_storage);
        --invalid_storage->secondary_blocks_size;
        invalid.mxfp4_file_storage = std::move(invalid_storage);
        expect_invalid(invalid);
        invalid = pinned_gate;
        invalid.input_column_indices[1] = invalid.input_column_indices[0];
        expect_invalid(invalid);
        invalid = pinned_gate;
        invalid.quantization_scales.push_back(1.0f);
        expect_invalid(invalid);

        const std::filesystem::path missing_path = directory / "missing.bin";
        invalid = pinned_gate;
        invalid_storage = std::make_shared<MxFp4FileStorage>(*pinned_gate.mxfp4_file_storage);
        invalid_storage->blocks_path = missing_path.string();
        invalid.mxfp4_file_storage = std::move(invalid_storage);
        ExpertCache missing_cache(1024 * 1024, 1, {}, ExpertIoMode::Buffered);
        auto missing = missing_cache.acquire_pair(invalid, pinned_down);
        check(!missing && missing.error().code == ErrorCode::IoError);

        const std::filesystem::path truncated_path = directory / "truncated.bin";
        write_file(truncated_path, {1});
        invalid = pinned_gate;
        invalid_storage = std::make_shared<MxFp4FileStorage>(*pinned_gate.mxfp4_file_storage);
        invalid_storage->blocks_path = truncated_path.string();
        invalid.mxfp4_file_storage = std::move(invalid_storage);
        ExpertCache truncated_cache(1024 * 1024, 1, {}, ExpertIoMode::Buffered);
        auto truncated = truncated_cache.acquire_pair(invalid, pinned_down);
        check(!truncated && truncated.error().code == ErrorCode::IoError);
    }

    {
        constexpr size_t entry_count = 96;
        ExpertCache rehash_cache(pinned_capacity * entry_count, 1, {}, ExpertIoMode::Buffered);
        for (size_t index = 0; index < entry_count; ++index)
        {
            std::string prepared_key = "qnk-rehash-" + std::to_string(index) + std::string(64, 'k');
            auto queued = rehash_cache.request_pair(pinned_gate, pinned_down, 0, prepared_key);
            check(queued && !queued.value());
            prepared_key.assign(prepared_key.size(), 'x');
        }
        rehash_cache.wait_for_background_work();
        for (size_t index = 0; index < entry_count; ++index)
        {
            const std::string prepared_key = "qnk-rehash-" + std::to_string(index) + std::string(64, 'k');
            auto hit = rehash_cache.acquire_pair(pinned_gate, pinned_down, 0, prepared_key);
            check(hit && hit.value().gate_up->qnk_values().front() == 11 + 2);
        }
        const ExpertCacheStatistics rehash_stats = rehash_cache.statistics();
        check(rehash_stats.misses == entry_count);
        check(rehash_stats.bytes_read == pinned_raw_size * entry_count);
        check(rehash_stats.resident_size == pinned_capacity * entry_count);
        check(rehash_stats.evictions == 0);
    }

    {
        ExpertCache reinsert_cache(pinned_capacity, 1, {}, ExpertIoMode::Buffered);
        const std::string reused_key = "qnk-reinsert";
        auto original = reinsert_cache.acquire_pair(pinned_gate, pinned_down, 0, reused_key);
        check(original && original.value().gate_up->qnk_values().front() == 11 + 2);
        original.value() = {};

        const std::string other_key = "qnk-evict";
        auto other = reinsert_cache.acquire_pair(reload_gate, reload_down, 0, other_key);
        check(other && other.value().gate_up->qnk_values().front() == 71 + 2);
        other.value() = {};

        const std::string reinserted_key = "qnk-reinsert";
        auto reinserted = reinsert_cache.acquire_pair(pinned_gate, pinned_down, 0, reinserted_key);
        check(reinserted && reinserted.value().gate_up->qnk_values().front() == 11 + 2);
        check(reinsert_cache.statistics().evictions == 2);
    }

    {
        ExpertCacheLease retained;
        {
            ExpertCache temporary_cache(pinned_capacity, 1, {}, ExpertIoMode::Buffered);
            std::string prepared_key = "qnk-cache-owner-gone";
            auto acquired = temporary_cache.acquire_pair(pinned_gate, pinned_down, 0, prepared_key);
            check(static_cast<bool>(acquired));
            retained = std::move(acquired).value();
            prepared_key.assign(prepared_key.size(), 'x');
        }
        check(retained.gate_up && retained.down);
        check(retained.gate_up->qnk_values().front() == 11 + 2);
        check(retained.down->qnk_values().front() == 51 + 2);
        check(retained.gate_up_operator && retained.down_operator);
    }

    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
}

void test_file_backed_mxfp4_expert_cache()
{
    const VulkanRuntimePtr vulkan_runtime = create_vulkan_runtime();
    const std::filesystem::path directory = create_unique_test_directory("ncnn_moe_expert_cache_test_");
    const std::filesystem::path blocks_path = directory / "blocks.bin";
    const std::filesystem::path scales_path = directory / "scales.bin";
    {
        std::vector<uint8_t> blocks(64);
        for (size_t index = 0; index < blocks.size(); ++index)
            blocks[index] = static_cast<uint8_t>(index);
        std::ofstream stream(blocks_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(blocks.data()), static_cast<std::streamsize>(blocks.size()));
    }
    {
        const std::vector<uint8_t> scales = {101, 102, 103, 104};
        std::ofstream stream(scales_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(scales.data()), static_cast<std::streamsize>(scales.size()));
    }

    auto file_backed = [&](uint64_t block_offset, uint64_t scale_offset, uint64_t block_size = 16) {
        TensorData tensor;
        tensor.dtype = DType::MxFp4;
        tensor.shape = {1, 32};
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->blocks_path = blocks_path.string();
        storage->blocks_offset = block_offset;
        storage->blocks_size = block_size;
        storage->scales_path = scales_path.string();
        storage->scales_offset = scale_offset;
        storage->scales_size = 1;
        tensor.mxfp4_file_storage = std::move(storage);
        return tensor;
    };

    const TensorData gate_zero = file_backed(0, 0);
    const TensorData down_zero = file_backed(16, 1);
    const TensorData gate_one = file_backed(32, 2);
    const TensorData down_one = file_backed(48, 3);
    ExpertCache cache(34, 0, {}, ExpertIoMode::Mmap);
    const CompiledOperator* first_gate_up_operator = nullptr;
    const CompiledOperator* first_down_operator = nullptr;
    {
        auto first = cache.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(first));
        check(static_cast<bool>(first.value().gate_up->mxfp4_blocks.front() == 0));
        check(static_cast<bool>(first.value().down->mxfp4_blocks.front() == 16));
        check(static_cast<bool>(first.value().gate_up->mxfp4_scales.front() == 101));
        check(static_cast<bool>(first.value().down->mxfp4_scales.front() == 102));
        check(static_cast<bool>(first.value().gate_up_operator != nullptr));
        check(static_cast<bool>(first.value().down_operator != nullptr));
        first_gate_up_operator = first.value().gate_up_operator;
        first_down_operator = first.value().down_operator;
    }
    {
        const std::string prepared_key = ExpertCache::make_pair_key(gate_zero, down_zero);
        check(cache.is_ready(gate_zero, down_zero, prepared_key));
        auto hit = cache.acquire_pair(gate_zero, down_zero, std::numeric_limits<uint32_t>::max(), prepared_key);
        check(static_cast<bool>(hit));
        check(static_cast<bool>(hit.value().gate_up_operator == first_gate_up_operator));
        check(static_cast<bool>(hit.value().down_operator == first_down_operator));
    }
    {
        const std::string prepared_key = ExpertCache::make_pair_key(gate_zero, down_zero);
        const std::array<ExpertCachePairRequest, 1> requests = {{
            &gate_zero,
            &down_zero,
            std::numeric_limits<uint32_t>::max(),
            prepared_key,
        }};
        std::array<ExpertCacheLease, 1> leases;
        auto acquired = cache.try_acquire_ready_pairs(requests, leases);
        check(static_cast<bool>(acquired));
        check(static_cast<bool>(acquired.value()));
        check(static_cast<bool>(leases[0].gate_up->mxfp4_blocks.front() == 0));
        check(static_cast<bool>(leases[0].gate_up_operator == first_gate_up_operator));
        check(static_cast<bool>(leases[0].down_operator == first_down_operator));
    }
    {
        auto second = cache.acquire_pair(gate_one, down_one);
        check(static_cast<bool>(second));
        check(static_cast<bool>(second.value().gate_up->mxfp4_blocks.front() == 32));
        check(static_cast<bool>(second.value().down->mxfp4_scales.front() == 104));
    }
    const ExpertCacheStatistics statistics = cache.statistics();
    check(static_cast<bool>(statistics.hits == 2));
    check(static_cast<bool>(statistics.misses == 2));
    check(static_cast<bool>(statistics.evictions == 1));
    check(static_cast<bool>(statistics.bytes_read == 68));
    check(static_cast<bool>(statistics.resident_size == 34));
    check(static_cast<bool>(statistics.mapped_ranges == 8));
    check(static_cast<bool>(statistics.mapped_bytes == statistics.bytes_read));
    check(static_cast<bool>(statistics.num_io_threads > 0));

    enum class LeaseAcquirePath
    {
        Acquire,
        TryReady,
        WaitReady
    };
    const auto check_single_payload_pin = [&](LeaseAcquirePath path, bool keep_gate_up) {
        ExpertCache pin_cache(34, 1, {}, ExpertIoMode::Buffered);
        const std::array<ExpertCachePairRequest, 1> first_request = {{
            {&gate_zero, &down_zero, 0, "lease-pin-zero"},
        }};
        const std::array<ExpertCachePairRequest, 1> second_request = {{
            {&gate_one, &down_one, 0, "lease-pin-one"},
        }};
        ExpertCacheLease lease;
        if (path == LeaseAcquirePath::Acquire)
        {
            auto acquired = pin_cache.acquire_pair(gate_zero, down_zero, 0, first_request[0].prepared_key);
            check(static_cast<bool>(acquired));
            lease = std::move(acquired).value();
        }
        else if (path == LeaseAcquirePath::TryReady)
        {
            {
                auto warmed = pin_cache.acquire_pair(gate_zero, down_zero, 0, first_request[0].prepared_key);
                check(static_cast<bool>(warmed));
            }
            auto acquired = pin_cache.try_acquire_ready_pairs(first_request, std::span<ExpertCacheLease>(&lease, 1));
            check(acquired && acquired.value());
        }
        else
        {
            auto acquired = pin_cache.wait_acquire_ready_pairs(first_request, std::span<ExpertCacheLease>(&lease, 1));
            check(acquired && acquired.value() == 1);
        }

        check(lease.gate_up && lease.down);
        check(lease.gate_up->mxfp4_blocks.front() == 0);
        check(lease.gate_up->mxfp4_scales.front() == 101);
        check(lease.down->mxfp4_blocks.front() == 16);
        check(lease.down->mxfp4_scales.front() == 102);
        check(lease.gate_up_operator && lease.down_operator);
        std::shared_ptr<const TensorData> retained = keep_gate_up ? lease.gate_up : lease.down;
        lease = {};
        check(static_cast<bool>(retained));
        check(retained->mxfp4_blocks.front() == (keep_gate_up ? 0 : 16));
        check(retained->mxfp4_scales.front() == (keep_gate_up ? 101 : 102));

        std::array<ExpertCacheLease, 1> next_lease;
        auto blocked = pin_cache.wait_acquire_ready_pairs(second_request, next_lease, true, 1, true, false);
        check(blocked && blocked.value() == 0);
        check(!next_lease[0].gate_up && !next_lease[0].down);
        check(pin_cache.statistics().queued_reads == 1);
        retained.reset();

        auto admitted = pin_cache.wait_acquire_ready_pairs(second_request, next_lease);
        check(admitted && admitted.value() == 1);
        check(next_lease[0].gate_up->mxfp4_blocks.front() == 32);
        check(next_lease[0].gate_up->mxfp4_scales.front() == 103);
        check(next_lease[0].down->mxfp4_blocks.front() == 48);
        check(next_lease[0].down->mxfp4_scales.front() == 104);
        const ExpertCacheStatistics pin_statistics = pin_cache.statistics();
        check(pin_statistics.bytes_read == 68);
        check(pin_statistics.resident_size == 34);
        check(pin_statistics.evictions == 1);
    };
    for (const LeaseAcquirePath path : {LeaseAcquirePath::Acquire,
                                        LeaseAcquirePath::TryReady,
                                        LeaseAcquirePath::WaitReady})
    {
        check_single_payload_pin(path, true);
        check_single_payload_pin(path, false);
    }

    ExpertCacheLease retained_lease;
    {
        ExpertCache pin_cache(34, 1, {}, ExpertIoMode::Buffered);
        auto acquired = pin_cache.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(acquired));
        check(static_cast<bool>(acquired.value().gate_up_operator != nullptr));
        check(static_cast<bool>(acquired.value().down_operator != nullptr));
        ExpertCacheLease copied = acquired.value();
        retained_lease = std::move(copied);
    }
    check(static_cast<bool>(retained_lease.gate_up_operator != nullptr));
    check(static_cast<bool>(retained_lease.down_operator != nullptr));
    check(static_cast<bool>(retained_lease.gate_up_operator->mxfp4_q8_packed == nullptr));
    check(static_cast<bool>(retained_lease.down_operator->mxfp4_q8_packed == nullptr));

    const std::filesystem::path packed_path = directory / "packed.bin";
    {
        std::vector<uint8_t> packed;
        packed.reserve(34);
        for (uint8_t value = 0; value < 16; ++value)
            packed.push_back(value);
        packed.push_back(101);
        for (uint8_t value = 16; value < 32; ++value)
            packed.push_back(value);
        packed.push_back(102);
        std::ofstream stream(packed_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(packed.data()), static_cast<std::streamsize>(packed.size()));
    }
    auto packed_tensor = [&](uint64_t block_offset, uint64_t scale_offset) {
        TensorData tensor;
        tensor.dtype = DType::MxFp4;
        tensor.shape = {1, 32};
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->blocks_path = packed_path.string();
        storage->blocks_offset = block_offset;
        storage->blocks_size = 16;
        storage->scales_path = packed_path.string();
        storage->scales_offset = scale_offset;
        storage->scales_size = 1;
        tensor.mxfp4_file_storage = std::move(storage);
        return tensor;
    };
    const TensorData packed_gate = packed_tensor(0, 16);
    const TensorData packed_down = packed_tensor(17, 33);
    ExpertCache packed_cache(34, 1, {}, ExpertIoMode::Buffered);
    auto packed_pair = packed_cache.acquire_pair(packed_gate, packed_down);
    check(static_cast<bool>(packed_pair));
    check(static_cast<bool>(packed_pair.value().gate_up->mxfp4_blocks.front() == 0));
    check(static_cast<bool>(packed_pair.value().gate_up->mxfp4_scales.front() == 101));
    check(static_cast<bool>(packed_pair.value().down->mxfp4_blocks.front() == 16));
    check(static_cast<bool>(packed_pair.value().down->mxfp4_scales.front() == 102));
    const ExpertCacheStatistics packed_statistics = packed_cache.statistics();
    check(static_cast<bool>(packed_statistics.buffered_read_ranges == 1));
    check(static_cast<bool>(packed_statistics.buffered_read_bytes == 34));
    const std::shared_ptr<const TensorData> dummy_gate = packed_pair.value().gate_up;
    const std::shared_ptr<const TensorData> dummy_down = packed_pair.value().down;

    const std::filesystem::path coalesced_path = directory / "coalesced.bin";
    {
        std::vector<uint8_t> bytes(68);
        for (size_t index = 0; index < bytes.size(); ++index)
            bytes[index] = static_cast<uint8_t>(index);
        std::ofstream stream(coalesced_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    auto coalesced_tensor = [](const std::filesystem::path& path, uint64_t block_offset, uint64_t scale_offset) {
        TensorData tensor;
        tensor.dtype = DType::MxFp4;
        tensor.shape = {1, 32};
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->blocks_path = path.string();
        storage->blocks_offset = block_offset;
        storage->blocks_size = 16;
        storage->scales_path = path.string();
        storage->scales_offset = scale_offset;
        storage->scales_size = 1;
        tensor.mxfp4_file_storage = std::move(storage);
        return tensor;
    };
    const TensorData coalesced_gate_zero = coalesced_tensor(coalesced_path, 0, 16);
    const TensorData coalesced_down_zero = coalesced_tensor(coalesced_path, 17, 33);
    const TensorData coalesced_gate_one = coalesced_tensor(coalesced_path, 34, 50);
    const TensorData coalesced_down_one = coalesced_tensor(coalesced_path, 51, 67);
    auto coalescing_victim = std::make_shared<TestExpertVictimCache>(34);
    coalescing_victim->admit("coalescing-dummy",
                             dummy_gate,
                             dummy_down,
                             {});
    coalescing_victim->block_restore("coalescing-dummy");
    ExpertCache coalesced_cache(102,
                                1,
                                coalescing_victim,
                                ExpertIoMode::Buffered,
                                ExpertCacheCrossExpertReadCoalescing);
    auto coalesced_dummy_request = coalesced_cache.request_pair(coalesced_gate_zero,
                                                                coalesced_down_zero,
                                                                0,
                                                                "coalescing-dummy");
    const bool coalesced_dummy_queued = static_cast<bool>(coalesced_dummy_request && !coalesced_dummy_request.value());
    const bool coalescing_restore_entered = coalesced_dummy_queued
                                                ? coalescing_victim->wait_for_blocked_restore()
                                                : false;
    auto coalesced_request_zero = coalesced_cache.request_pair(coalesced_gate_zero, coalesced_down_zero, 0);
    auto coalesced_request_one = coalesced_cache.request_pair(coalesced_gate_one, coalesced_down_one, 1);
    coalescing_victim->release_blocked_restore();
    check(coalesced_dummy_queued);
    check(coalescing_restore_entered);
    check(static_cast<bool>(coalesced_request_zero && !coalesced_request_zero.value()));
    check(static_cast<bool>(coalesced_request_one && !coalesced_request_one.value()));
    coalesced_cache.wait_for_background_work();
    auto coalesced_zero = coalesced_cache.acquire_pair(coalesced_gate_zero, coalesced_down_zero, 0);
    auto coalesced_one = coalesced_cache.acquire_pair(coalesced_gate_one, coalesced_down_one, 1);
    check(static_cast<bool>(coalesced_zero));
    check(static_cast<bool>(coalesced_one));
    check(coalesced_zero.value().gate_up->mxfp4_blocks.front() == 0);
    check(coalesced_zero.value().down->mxfp4_blocks.front() == 17);
    check(coalesced_one.value().gate_up->mxfp4_blocks.front() == 34);
    check(coalesced_one.value().down->mxfp4_scales.front() == 67);
    const ExpertCacheStatistics coalesced_statistics = coalesced_cache.statistics();
    check(coalesced_statistics.buffered_read_ranges == 1);
    check(coalesced_statistics.buffered_read_bytes == 68);
    check(coalesced_statistics.coalesced_read_batches == 1);
    check(coalesced_statistics.coalesced_experts == 2);
    check(coalesced_statistics.coalesced_read_ranges_saved == 1);
    check(coalesced_statistics.resident_size == 102);
    check(coalesced_zero.value().gate_up->mxfp4_blocks.data() + 16
          == coalesced_zero.value().gate_up->mxfp4_scales.data());
    check(coalesced_zero.value().gate_up->mxfp4_scales.data() + 1
          == coalesced_zero.value().down->mxfp4_blocks.data());
    check(coalesced_zero.value().down->mxfp4_blocks.data() + 16
          == coalesced_zero.value().down->mxfp4_scales.data());
    check(coalesced_zero.value().gate_up->mxfp4_blocks.data()
          != coalesced_one.value().gate_up->mxfp4_blocks.data());

    constexpr size_t coalesced_many_count = 8;
    constexpr size_t coalesced_many_pair_size = 34;
    const std::filesystem::path coalesced_many_path = directory / "coalesced_many.bin";
    {
        std::vector<uint8_t> bytes(coalesced_many_count * coalesced_many_pair_size);
        for (size_t index = 0; index < bytes.size(); ++index)
            bytes[index] = static_cast<uint8_t>(index);
        std::ofstream stream(coalesced_many_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    std::array<TensorData, coalesced_many_count> coalesced_many_gates;
    std::array<TensorData, coalesced_many_count> coalesced_many_downs;
    std::array<std::string, coalesced_many_count> coalesced_many_keys;
    std::array<ExpertCachePairRequest, coalesced_many_count> coalesced_many_requests{};
    for (size_t index = 0; index < coalesced_many_count; ++index)
    {
        const uint64_t offset = index * coalesced_many_pair_size;
        coalesced_many_gates[index] = coalesced_tensor(coalesced_many_path, offset, offset + 16);
        coalesced_many_downs[index] = coalesced_tensor(coalesced_many_path, offset + 17, offset + 33);
        coalesced_many_keys[index] = "coalesced-many-" + std::to_string(index);
        coalesced_many_requests[index] = {&coalesced_many_gates[index],
                                          &coalesced_many_downs[index],
                                          0,
                                          coalesced_many_keys[index],
                                          {}};
    }
    auto coalesced_many_victim = std::make_shared<TestExpertVictimCache>(34);
    coalesced_many_victim->admit("coalesced-many-dummy", dummy_gate, dummy_down, {});
    coalesced_many_victim->block_restore("coalesced-many-dummy");
    constexpr uint64_t coalesced_many_capacity = coalesced_many_count * coalesced_many_pair_size + 34;
    ExpertCache coalesced_many_cache(coalesced_many_capacity,
                                     1,
                                     coalesced_many_victim,
                                     ExpertIoMode::Buffered,
                                     ExpertCacheCrossExpertReadCoalescing);
    auto coalesced_many_dummy = coalesced_many_cache.request_pair(coalesced_many_gates[0],
                                                                  coalesced_many_downs[0],
                                                                  0,
                                                                  "coalesced-many-dummy");
    const bool coalesced_many_dummy_queued = static_cast<bool>(coalesced_many_dummy && !coalesced_many_dummy.value());
    const bool coalesced_many_restore_entered = coalesced_many_dummy_queued
                                                    ? coalesced_many_victim->wait_for_blocked_restore()
                                                    : false;
    auto coalesced_many_queued = coalesced_many_cache.request_pairs(coalesced_many_requests);
    coalesced_many_victim->release_blocked_restore();
    check(coalesced_many_dummy_queued);
    check(coalesced_many_restore_entered);
    check(static_cast<bool>(coalesced_many_queued && coalesced_many_queued.value() == coalesced_many_count));
    coalesced_many_cache.wait_for_background_work();
    std::array<ExpertCacheLease, coalesced_many_count> coalesced_many_leases;
    std::array<const uint8_t*, coalesced_many_count> coalesced_many_buffers{};
    for (size_t index = 0; index < coalesced_many_count; ++index)
    {
        auto acquired = coalesced_many_cache.acquire_pair(coalesced_many_gates[index],
                                                          coalesced_many_downs[index],
                                                          0,
                                                          coalesced_many_keys[index]);
        check(static_cast<bool>(acquired));
        coalesced_many_leases[index] = std::move(acquired).value();
        const ExpertCacheLease& lease = coalesced_many_leases[index];
        const size_t offset = index * coalesced_many_pair_size;
        check(lease.gate_up->mxfp4_blocks.front() == static_cast<uint8_t>(offset));
        check(lease.gate_up->mxfp4_scales.front() == static_cast<uint8_t>(offset + 16));
        check(lease.down->mxfp4_blocks.front() == static_cast<uint8_t>(offset + 17));
        check(lease.down->mxfp4_scales.front() == static_cast<uint8_t>(offset + 33));
        check(lease.gate_up->mxfp4_blocks.data() + 16 == lease.gate_up->mxfp4_scales.data());
        check(lease.gate_up->mxfp4_scales.data() + 1 == lease.down->mxfp4_blocks.data());
        check(lease.down->mxfp4_blocks.data() + 16 == lease.down->mxfp4_scales.data());
        coalesced_many_buffers[index] = lease.gate_up->mxfp4_blocks.data();
        for (size_t previous = 0; previous < index; ++previous)
            check(coalesced_many_buffers[previous] != coalesced_many_buffers[index]);
    }
    const ExpertCacheStatistics coalesced_many_statistics = coalesced_many_cache.statistics();
    check(coalesced_many_statistics.buffered_read_ranges == 1);
    check(coalesced_many_statistics.buffered_read_bytes == coalesced_many_count * coalesced_many_pair_size);
    check(coalesced_many_statistics.coalesced_read_batches == 1);
    check(coalesced_many_statistics.coalesced_experts == coalesced_many_count);
    check(coalesced_many_statistics.coalesced_read_ranges_saved == coalesced_many_count - 1);
    check(coalesced_many_statistics.resident_size == coalesced_many_capacity);

    const std::filesystem::path coalesced_truncated_path = directory / "coalesced_truncated.bin";
    {
        std::vector<uint8_t> bytes(17, 0x5a);
        std::ofstream stream(coalesced_truncated_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    std::array<TensorData, coalesced_many_count> coalesced_truncated_gates;
    std::array<TensorData, coalesced_many_count> coalesced_truncated_downs;
    std::array<std::string, coalesced_many_count> coalesced_truncated_keys;
    std::array<ExpertCachePairRequest, coalesced_many_count> coalesced_truncated_requests{};
    for (size_t index = 0; index < coalesced_many_count; ++index)
    {
        const uint64_t offset = index * coalesced_many_pair_size;
        coalesced_truncated_gates[index] = coalesced_tensor(coalesced_truncated_path, offset, offset + 16);
        coalesced_truncated_downs[index] = coalesced_tensor(coalesced_truncated_path, offset + 17, offset + 33);
        coalesced_truncated_keys[index] = "coalesced-truncated-" + std::to_string(index);
        coalesced_truncated_requests[index] = {&coalesced_truncated_gates[index],
                                               &coalesced_truncated_downs[index],
                                               0,
                                               coalesced_truncated_keys[index],
                                               {}};
    }
    auto coalesced_truncated_victim = std::make_shared<TestExpertVictimCache>(34);
    coalesced_truncated_victim->admit("coalesced-truncated-dummy", dummy_gate, dummy_down, {});
    coalesced_truncated_victim->block_restore("coalesced-truncated-dummy");
    ExpertCache coalesced_truncated_cache(coalesced_many_capacity,
                                          1,
                                          coalesced_truncated_victim,
                                          ExpertIoMode::Buffered,
                                          ExpertCacheCrossExpertReadCoalescing);
    auto coalesced_truncated_dummy = coalesced_truncated_cache.request_pair(coalesced_truncated_gates[0],
                                                                            coalesced_truncated_downs[0],
                                                                            0,
                                                                            "coalesced-truncated-dummy");
    const bool coalesced_truncated_dummy_queued = static_cast<bool>(coalesced_truncated_dummy
                                                                    && !coalesced_truncated_dummy.value());
    const bool coalesced_truncated_restore_entered = coalesced_truncated_dummy_queued
                                                         ? coalesced_truncated_victim->wait_for_blocked_restore()
                                                         : false;
    // The restored sentinel remains resident when the following disk batch fails.
    const uint64_t coalesced_truncated_baseline_size = 34;
    auto coalesced_truncated_queued = coalesced_truncated_cache.request_pairs(coalesced_truncated_requests);
    coalesced_truncated_victim->release_blocked_restore();
    check(coalesced_truncated_dummy_queued);
    check(coalesced_truncated_restore_entered);
    check(static_cast<bool>(coalesced_truncated_queued
                            && coalesced_truncated_queued.value() == coalesced_many_count));
    coalesced_truncated_cache.wait_for_background_work();
    const ExpertCacheStatistics coalesced_truncated_statistics = coalesced_truncated_cache.statistics();
    check(coalesced_truncated_statistics.resident_size == coalesced_truncated_baseline_size);
    check(coalesced_truncated_statistics.buffered_read_ranges == 0);
    check(coalesced_truncated_statistics.buffered_read_bytes == 0);
    check(coalesced_truncated_statistics.bytes_read == 0);
    check(coalesced_truncated_statistics.coalesced_read_batches == 1);
    for (size_t index = 0; index < coalesced_many_count; ++index)
        check(!coalesced_truncated_cache.is_ready(coalesced_truncated_gates[index],
                                                  coalesced_truncated_downs[index],
                                                  coalesced_truncated_keys[index]));
    auto coalesced_truncated_failure = coalesced_truncated_cache.acquire_pair(coalesced_truncated_gates[0],
                                                                              coalesced_truncated_downs[0],
                                                                              0,
                                                                              coalesced_truncated_keys[0]);
    check(!coalesced_truncated_failure);
    check(coalesced_truncated_failure.error().code == ErrorCode::IoError);
    check(coalesced_truncated_cache.statistics().resident_size == coalesced_truncated_baseline_size);

    const std::filesystem::path interleaved_coalesced_path = directory / "interleaved_coalesced.bin";
    {
        std::vector<uint8_t> bytes(306);
        for (size_t index = 0; index < bytes.size(); ++index)
            bytes[index] = static_cast<uint8_t>(index);
        std::ofstream stream(interleaved_coalesced_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    auto interleaved_coalesced_gate = [&](uint64_t offset) {
        TensorData tensor;
        tensor.dtype = DType::MxFp4;
        tensor.shape = {4, 32};
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->blocks_path = interleaved_coalesced_path.string();
        storage->blocks_offset = offset;
        storage->blocks_size = 32;
        storage->scales_path = interleaved_coalesced_path.string();
        storage->scales_offset = offset + 32;
        storage->scales_size = 2;
        storage->secondary_blocks_path = interleaved_coalesced_path.string();
        storage->secondary_blocks_offset = offset + 34;
        storage->secondary_blocks_size = 32;
        storage->secondary_scales_path = interleaved_coalesced_path.string();
        storage->secondary_scales_offset = offset + 66;
        storage->secondary_scales_size = 2;
        storage->interleave_rows = true;
        tensor.mxfp4_file_storage = std::move(storage);
        return tensor;
    };
    auto interleaved_coalesced_down = [&](uint64_t offset) {
        TensorData tensor;
        tensor.dtype = DType::MxFp4;
        tensor.shape = {2, 32};
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->blocks_path = interleaved_coalesced_path.string();
        storage->blocks_offset = offset;
        storage->blocks_size = 32;
        storage->scales_path = interleaved_coalesced_path.string();
        storage->scales_offset = offset + 32;
        storage->scales_size = 2;
        tensor.mxfp4_file_storage = std::move(storage);
        return tensor;
    };
    const TensorData interleaved_gate_zero = interleaved_coalesced_gate(0);
    const TensorData interleaved_down_zero = interleaved_coalesced_down(68);
    const TensorData interleaved_gate_one = interleaved_coalesced_gate(102);
    const TensorData interleaved_down_one = interleaved_coalesced_down(170);
    const TensorData interleaved_gate_two = interleaved_coalesced_gate(204);
    const TensorData interleaved_down_two = interleaved_coalesced_down(272);
    auto interleaved_coalescing_victim = std::make_shared<TestExpertVictimCache>(34);
    interleaved_coalescing_victim->admit("interleaved-coalescing-dummy", dummy_gate, dummy_down, {});
    interleaved_coalescing_victim->block_restore("interleaved-coalescing-dummy");
    ExpertCache interleaved_coalesced_cache(238,
                                            1,
                                            interleaved_coalescing_victim,
                                            ExpertIoMode::Buffered,
                                            ExpertCacheCrossExpertReadCoalescing);
    auto interleaved_dummy_request = interleaved_coalesced_cache.request_pair(packed_gate,
                                                                              packed_down,
                                                                              0,
                                                                              "interleaved-coalescing-dummy");
    const bool interleaved_dummy_queued = static_cast<bool>(interleaved_dummy_request && !interleaved_dummy_request.value());
    const bool interleaved_restore_entered = interleaved_dummy_queued
                                                 ? interleaved_coalescing_victim->wait_for_blocked_restore()
                                                 : false;
    auto interleaved_request_zero = interleaved_coalesced_cache.request_pair(interleaved_gate_zero, interleaved_down_zero, 0);
    auto interleaved_request_one = interleaved_coalesced_cache.request_pair(interleaved_gate_one, interleaved_down_one, 1);
    interleaved_coalescing_victim->release_blocked_restore();
    check(interleaved_dummy_queued);
    check(interleaved_restore_entered);
    check(static_cast<bool>(interleaved_request_zero && !interleaved_request_zero.value()));
    check(static_cast<bool>(interleaved_request_one && !interleaved_request_one.value()));
    interleaved_coalesced_cache.wait_for_background_work();
    auto interleaved_zero = interleaved_coalesced_cache.acquire_pair(interleaved_gate_zero, interleaved_down_zero, 0);
    auto interleaved_one = interleaved_coalesced_cache.acquire_pair(interleaved_gate_one, interleaved_down_one, 1);
    check(static_cast<bool>(interleaved_zero));
    check(static_cast<bool>(interleaved_one));
    check(interleaved_zero.value().gate_up->mxfp4_blocks[0] == 0);
    check(interleaved_zero.value().gate_up->mxfp4_blocks[16] == 34);
    check(interleaved_zero.value().gate_up->mxfp4_blocks[32] == 16);
    check(interleaved_zero.value().gate_up->mxfp4_blocks[48] == 50);
    check(interleaved_zero.value().gate_up->mxfp4_scales[0] == 32);
    check(interleaved_zero.value().gate_up->mxfp4_scales[1] == 66);
    check(interleaved_zero.value().gate_up->mxfp4_scales[2] == 33);
    check(interleaved_zero.value().gate_up->mxfp4_scales[3] == 67);
    check(interleaved_zero.value().down->mxfp4_blocks.front() == 68);
    check(interleaved_zero.value().down->mxfp4_scales.front() == 100);
    check(interleaved_zero.value().gate_up->mxfp4_blocks.data() + 64
          == interleaved_zero.value().gate_up->mxfp4_scales.data());
    check(interleaved_zero.value().gate_up->mxfp4_scales.data() + 4
          == interleaved_zero.value().down->mxfp4_blocks.data());
    check(interleaved_zero.value().down->mxfp4_blocks.data() + 32
          == interleaved_zero.value().down->mxfp4_scales.data());
    check(interleaved_zero.value().gate_up->mxfp4_blocks.data()
          != interleaved_one.value().gate_up->mxfp4_blocks.data());
    const ExpertCacheStatistics interleaved_statistics = interleaved_coalesced_cache.statistics();
    check(interleaved_statistics.buffered_read_ranges == 1);
    check(interleaved_statistics.buffered_read_bytes == 204);
    check(interleaved_statistics.coalesced_read_batches == 1);
    check(interleaved_statistics.coalesced_experts == 2);
    check(interleaved_statistics.resident_size == 238);

    ExpertCacheLease retained_interleaved = std::move(interleaved_zero).value();
    ExpertCacheLease released_interleaved = std::move(interleaved_one).value();
    released_interleaved = {};
    auto interleaved_third = interleaved_coalesced_cache.acquire_pair(interleaved_gate_two, interleaved_down_two, 2);
    check(static_cast<bool>(interleaved_third));
    check(interleaved_third.value().gate_up->mxfp4_blocks.front() == 204);
    check(interleaved_third.value().down->mxfp4_scales.front() == static_cast<uint8_t>(304));
    check(interleaved_coalesced_cache.statistics().evictions >= 1);
    check(interleaved_coalesced_cache.statistics().resident_size <= interleaved_coalesced_cache.capacity());
    check(retained_interleaved.gate_up->mxfp4_blocks[0] == 0);
    check(retained_interleaved.gate_up->mxfp4_blocks[48] == 50);
    check(retained_interleaved.gate_up->mxfp4_scales[3] == 67);
    check(retained_interleaved.down->mxfp4_blocks.front() == 68);
    check(retained_interleaved.down->mxfp4_scales.front() == 100);

    ExpertCache resolved_predictions(68, 1, {}, ExpertIoMode::Buffered);
    auto prediction_zero = resolved_predictions.prefetch_pair(gate_zero, down_zero, 1, "prediction-zero");
    auto prediction_one = resolved_predictions.prefetch_pair(gate_one, down_one, 1, "prediction-one");
    check(static_cast<bool>(prediction_zero && !prediction_zero.value()));
    check(static_cast<bool>(prediction_one && !prediction_one.value()));
    resolved_predictions.wait_for_background_work();
    const std::array<std::string_view, 1> demanded_prediction = {"prediction-zero"};
    resolved_predictions.resolve_predictions(1, demanded_prediction);
    check(resolved_predictions.is_ready(gate_zero, down_zero, "prediction-zero"));
    check(!resolved_predictions.is_ready(gate_one, down_one, "prediction-one"));
    check(resolved_predictions.statistics().unused_speculative_reads == 1);

    ExpertCache forward_aware(68,
                              1,
                              {},
                              ExpertIoMode::Buffered,
                              ExpertCacheForwardAwareEviction,
                              4);
    auto forward_coordinator = std::make_shared<ExpertResidencyCoordinator>();
    forward_aware.set_residency_coordinator(forward_coordinator);
    auto gpu_forward_two = forward_coordinator->register_device("forward-two");
    {
        auto group_zero = forward_aware.acquire_pair(gate_zero, down_zero, 0, "forward-zero");
        auto group_two = forward_aware.acquire_pair(gate_one, down_one, 2, "forward-two");
        check(static_cast<bool>(group_zero));
        check(static_cast<bool>(group_two));
    }
    auto group_one = forward_aware.acquire_pair(gate_zero, down_one, 1, "forward-one");
    check(static_cast<bool>(group_one));
    check(!forward_aware.is_ready(gate_zero, down_zero, "forward-zero"));
    check(forward_aware.is_ready(gate_one, down_one, "forward-two"));

    ExpertCache predicted_protection(68,
                                     1,
                                     {},
                                     ExpertIoMode::Buffered,
                                     ExpertCacheForwardAwareEviction,
                                     4);
    auto prediction_coordinator = std::make_shared<ExpertResidencyCoordinator>();
    predicted_protection.set_residency_coordinator(prediction_coordinator);
    auto gpu_prediction = prediction_coordinator->register_device("protected-prediction");
    {
        auto exact = predicted_protection.acquire_pair(gate_zero,
                                                       down_zero,
                                                       2,
                                                       "protected-exact");
        auto repeated = predicted_protection.acquire_pair(gate_zero,
                                                          down_zero,
                                                          2,
                                                          "protected-exact");
        check(static_cast<bool>(exact));
        check(static_cast<bool>(repeated));
    }
    auto predicted = predicted_protection.prefetch_pair(gate_one,
                                                        down_one,
                                                        1,
                                                        "protected-prediction");
    check(static_cast<bool>(predicted && !predicted.value()));
    predicted_protection.wait_for_background_work();
    auto incoming = predicted_protection.acquire_pair(gate_zero,
                                                      down_one,
                                                      0,
                                                      "protected-incoming");
    check(static_cast<bool>(incoming));
    check(!predicted_protection.is_ready(gate_zero,
                                         down_zero,
                                         "protected-exact"));
    check(predicted_protection.is_ready(gate_one,
                                        down_one,
                                        "protected-prediction"));

    const std::filesystem::path clustered_path = directory / "clustered.bin";
    {
        std::vector<uint8_t> clustered(112, UINT8_C(0xee));
        clustered[0] = 101;
        clustered[1] = 102;
        clustered[2] = 201;
        clustered[3] = 202;
        clustered[4] = 111;
        clustered[5] = 112;
        for (uint8_t value = 0; value < 96; ++value)
            clustered[16 + value] = value;
        std::ofstream stream(clustered_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(clustered.data()), static_cast<std::streamsize>(clustered.size()));
    }
    TensorData clustered_gate;
    clustered_gate.dtype = DType::MxFp4;
    clustered_gate.shape = {4, 32};
    auto clustered_gate_storage = std::make_shared<MxFp4FileStorage>();
    clustered_gate_storage->blocks_path = clustered_path.string();
    clustered_gate_storage->blocks_offset = 16;
    clustered_gate_storage->blocks_size = 32;
    clustered_gate_storage->scales_path = clustered_path.string();
    clustered_gate_storage->scales_offset = 0;
    clustered_gate_storage->scales_size = 2;
    clustered_gate_storage->secondary_blocks_path = clustered_path.string();
    clustered_gate_storage->secondary_blocks_offset = 80;
    clustered_gate_storage->secondary_blocks_size = 32;
    clustered_gate_storage->secondary_scales_path = clustered_path.string();
    clustered_gate_storage->secondary_scales_offset = 4;
    clustered_gate_storage->secondary_scales_size = 2;
    clustered_gate_storage->interleave_rows = true;
    clustered_gate.mxfp4_file_storage = std::move(clustered_gate_storage);
    TensorData clustered_down;
    clustered_down.dtype = DType::MxFp4;
    clustered_down.shape = {2, 32};
    auto clustered_down_storage = std::make_shared<MxFp4FileStorage>();
    clustered_down_storage->blocks_path = clustered_path.string();
    clustered_down_storage->blocks_offset = 48;
    clustered_down_storage->blocks_size = 32;
    clustered_down_storage->scales_path = clustered_path.string();
    clustered_down_storage->scales_offset = 2;
    clustered_down_storage->scales_size = 2;
    clustered_down.mxfp4_file_storage = std::move(clustered_down_storage);
    ExpertCache clustered_cache(102, 1, {}, ExpertIoMode::Buffered);
    auto clustered_pair = clustered_cache.acquire_pair(clustered_gate, clustered_down);
    check(static_cast<bool>(clustered_pair));
    check(clustered_pair.value().gate_up->mxfp4_blocks[0] == 0);
    check(clustered_pair.value().gate_up->mxfp4_blocks[16] == 64);
    check(clustered_pair.value().gate_up->mxfp4_blocks[32] == 16);
    check(clustered_pair.value().gate_up->mxfp4_blocks[48] == 80);
    check(clustered_pair.value().gate_up->mxfp4_scales[0] == 101);
    check(clustered_pair.value().gate_up->mxfp4_scales[1] == 111);
    check(clustered_pair.value().gate_up->mxfp4_scales[2] == 102);
    check(clustered_pair.value().gate_up->mxfp4_scales[3] == 112);
    check(clustered_pair.value().down->mxfp4_blocks[0] == 32);
    check(clustered_pair.value().down->mxfp4_blocks[16] == 48);
    check(clustered_pair.value().down->mxfp4_scales[0] == 201);
    check(clustered_pair.value().down->mxfp4_scales[1] == 202);
    const ExpertCacheStatistics clustered_statistics = clustered_cache.statistics();
    check(clustered_statistics.buffered_read_ranges == 2);
    check(clustered_statistics.buffered_read_bytes == 102);

    const std::filesystem::path fragmented_path = directory / "fragmented.bin";
    {
        std::vector<uint8_t> fragmented(107, UINT8_C(0xee));
        for (uint8_t value = 0; value < 32; ++value)
            fragmented[value] = value;
        fragmented[33] = 101;
        fragmented[34] = 102;
        for (uint8_t value = 0; value < 32; ++value)
            fragmented[36 + value] = static_cast<uint8_t>(64 + value);
        fragmented[69] = 111;
        fragmented[70] = 112;
        for (uint8_t value = 0; value < 32; ++value)
            fragmented[72 + value] = static_cast<uint8_t>(32 + value);
        fragmented[105] = 201;
        fragmented[106] = 202;
        std::ofstream stream(fragmented_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(fragmented.data()), static_cast<std::streamsize>(fragmented.size()));
    }
    TensorData fragmented_gate;
    fragmented_gate.dtype = DType::MxFp4;
    fragmented_gate.shape = {4, 32};
    auto fragmented_gate_storage = std::make_shared<MxFp4FileStorage>();
    fragmented_gate_storage->blocks_path = fragmented_path.string();
    fragmented_gate_storage->blocks_offset = 0;
    fragmented_gate_storage->blocks_size = 32;
    fragmented_gate_storage->scales_path = fragmented_path.string();
    fragmented_gate_storage->scales_offset = 33;
    fragmented_gate_storage->scales_size = 2;
    fragmented_gate_storage->secondary_blocks_path = fragmented_path.string();
    fragmented_gate_storage->secondary_blocks_offset = 36;
    fragmented_gate_storage->secondary_blocks_size = 32;
    fragmented_gate_storage->secondary_scales_path = fragmented_path.string();
    fragmented_gate_storage->secondary_scales_offset = 69;
    fragmented_gate_storage->secondary_scales_size = 2;
    fragmented_gate_storage->interleave_rows = true;
    fragmented_gate.mxfp4_file_storage = std::move(fragmented_gate_storage);
    TensorData fragmented_down;
    fragmented_down.dtype = DType::MxFp4;
    fragmented_down.shape = {2, 32};
    auto fragmented_down_storage = std::make_shared<MxFp4FileStorage>();
    fragmented_down_storage->blocks_path = fragmented_path.string();
    fragmented_down_storage->blocks_offset = 72;
    fragmented_down_storage->blocks_size = 32;
    fragmented_down_storage->scales_path = fragmented_path.string();
    fragmented_down_storage->scales_offset = 105;
    fragmented_down_storage->scales_size = 2;
    fragmented_down.mxfp4_file_storage = std::move(fragmented_down_storage);
    ExpertCache fragmented_cache(102, 1, {}, ExpertIoMode::Buffered);
    auto fragmented_pair = fragmented_cache.acquire_pair(fragmented_gate, fragmented_down);
    check(static_cast<bool>(fragmented_pair));
    check(fragmented_pair.value().gate_up->mxfp4_blocks[0] == 0);
    check(fragmented_pair.value().gate_up->mxfp4_blocks[16] == 64);
    check(fragmented_pair.value().gate_up->mxfp4_blocks[32] == 16);
    check(fragmented_pair.value().gate_up->mxfp4_blocks[48] == 80);
    check(fragmented_pair.value().gate_up->mxfp4_scales[0] == 101);
    check(fragmented_pair.value().gate_up->mxfp4_scales[1] == 111);
    check(fragmented_pair.value().down->mxfp4_blocks[0] == 32);
    check(fragmented_pair.value().down->mxfp4_scales[0] == 201);
    const ExpertCacheStatistics fragmented_statistics = fragmented_cache.statistics();
    check(fragmented_statistics.buffered_read_ranges == 6);
    check(fragmented_statistics.buffered_read_bytes == 102);

    TensorData invalid_secondary = packed_gate;
    auto invalid_secondary_storage = std::make_shared<MxFp4FileStorage>(*packed_gate.mxfp4_file_storage);
    invalid_secondary_storage->secondary_blocks_path = packed_path.string();
    invalid_secondary_storage->secondary_blocks_offset = 17;
    invalid_secondary_storage->secondary_blocks_size = 16;
    invalid_secondary.mxfp4_file_storage = std::move(invalid_secondary_storage);
    ExpertCache invalid_secondary_cache(50, 1, {}, ExpertIoMode::Buffered);
    auto invalid_secondary_pair = invalid_secondary_cache.acquire_pair(invalid_secondary, packed_down);
    check(!invalid_secondary_pair);
    check(invalid_secondary_pair.error().code == ErrorCode::InvalidModel);

    auto victim = std::make_shared<TestExpertVictimCache>(68);
    ExpertCache tiered(34, 1, victim, ExpertIoMode::Mmap);
    {
        auto first = tiered.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(first));
        check(static_cast<bool>(first.value().gate_up->mxfp4_blocks.front() == 0));
    }
    {
        auto second = tiered.acquire_pair(gate_one, down_one);
        check(static_cast<bool>(second));
        check(static_cast<bool>(second.value().gate_up->mxfp4_blocks.front() == 32));
    }
    {
        auto restored = tiered.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(restored));
        check(static_cast<bool>(restored.value().gate_up->mxfp4_blocks.front() == 0));
        check(static_cast<bool>(restored.value().down->mxfp4_scales.front() == 102));
    }
    const ExpertCacheStatistics tiered_statistics = tiered.statistics();
    check(static_cast<bool>(tiered_statistics.misses == 3));
    check(static_cast<bool>(tiered_statistics.evictions == 2));
    check(static_cast<bool>(tiered_statistics.bytes_read == 68));
    check(static_cast<bool>(tiered_statistics.mapped_ranges == 8));
    check(static_cast<bool>(tiered_statistics.mapped_bytes == tiered_statistics.bytes_read));
    check(static_cast<bool>(tiered_statistics.victim.hits == 1));
    check(static_cast<bool>(tiered_statistics.victim.misses == 2));
    check(static_cast<bool>(tiered_statistics.victim.admissions == 2));

    if (get_gpu_count() > 0)
    {
        auto gpu_source = cache.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(gpu_source));
        auto gpu_victim = create_vulkan_victim_cache(68,
                                                     automatic_vulkan_device_index,
                                                     vulkan_runtime,
                                                     g_test_optimization_flags);
        check(static_cast<bool>(gpu_victim));
        auto coordinator = std::make_shared<ExpertResidencyCoordinator>();
        gpu_victim->set_residency_coordinator(coordinator);
        check(!coordinator->device_resident("gpu-roundtrip"));
        auto oversized_gate = std::make_shared<TensorData>(*gpu_source.value().gate_up);
        oversized_gate->mxfp4_blocks.resize(128);
        gpu_victim->admit("gpu-rejected", oversized_gate, gpu_source.value().down);
        check(!coordinator->device_resident("gpu-rejected"));
        gpu_victim->admit("gpu-roundtrip", gpu_source.value().gate_up, gpu_source.value().down);
        gpu_victim->wait_for_background_work();
        check(coordinator->device_resident("gpu-roundtrip"));
        check(!coordinator->host_resident("gpu-roundtrip"));
        check(static_cast<bool>(gpu_victim->statistics().stores == 1));
        auto gpu_restored = gpu_victim->restore("gpu-roundtrip", gate_zero, down_zero);
        check(static_cast<bool>(gpu_restored));
        check(static_cast<bool>(gpu_restored->gate_up->mxfp4_blocks.front() == 0));
        check(static_cast<bool>(gpu_restored->down->mxfp4_blocks.front() == 16));
        check(static_cast<bool>(gpu_restored->gate_up->mxfp4_scales.front() == 101));
        check(static_cast<bool>(gpu_restored->down->mxfp4_scales.front() == 102));
        const ExpertVictimCacheStatistics gpu_statistics = gpu_victim->statistics();
        check(static_cast<bool>(gpu_statistics.bytes_uploaded >= 34));
        check(static_cast<bool>(gpu_statistics.bytes_downloaded == gpu_statistics.bytes_uploaded));
        check(static_cast<bool>(gpu_statistics.mapped_stores <= gpu_statistics.stores));
        check(static_cast<bool>(gpu_statistics.mapped_restores <= gpu_statistics.hits));
        check(static_cast<bool>(gpu_statistics.mapped_stores == gpu_statistics.mapped_restores));
        TensorData incompatible = gate_zero;
        auto incompatible_storage = std::make_shared<MxFp4FileStorage>(*gate_zero.mxfp4_file_storage);
        ++incompatible_storage->blocks_size;
        incompatible.mxfp4_file_storage = std::move(incompatible_storage);
        auto failed_restore = gpu_victim->restore("gpu-roundtrip", incompatible, down_zero);
        check(!failed_restore);
        check(!coordinator->device_resident("gpu-roundtrip"));
        gpu_victim->admit("gpu-lifetime", gpu_source.value().gate_up, gpu_source.value().down);
        gpu_victim->wait_for_background_work();
        check(coordinator->device_resident("gpu-lifetime"));
        gpu_victim.reset();
        check(!coordinator->device_resident("gpu-lifetime"));
    }

    // Coordinated pressure reclaims duplicate bytes inside ARC's selected
    // class, while keeping host-only coverage and live leases intact.
    {
        auto coordinator = std::make_shared<ExpertResidencyCoordinator>();
        ExpertCache joint(68, 1, {}, ExpertIoMode::Buffered);
        joint.set_residency_coordinator(coordinator);
        {
            auto first = joint.acquire_pair(gate_zero, down_zero, 0, "joint-zero");
            auto second = joint.acquire_pair(gate_one, down_one, 0, "joint-one");
            check(first && second);
        }
        auto gpu_one = coordinator->register_device("joint-one");
        check(joint.is_ready(gate_zero, down_zero, "joint-zero"));
        check(joint.is_ready(gate_one, down_one, "joint-one"));
        check(coordinator->host_resident("joint-one") && coordinator->device_resident("joint-one"));
        {
            auto incoming = joint.acquire_pair(gate_zero, down_one, 0, "joint-two");
            check(static_cast<bool>(incoming));
        }
        check(joint.is_ready(gate_zero, down_zero, "joint-zero"));
        check(!joint.is_ready(gate_one, down_one, "joint-one"));
        check(!coordinator->host_resident("joint-one"));
        check(coordinator->device_resident("joint-one"));
        auto joint_statistics = joint.statistics();
        check(joint_statistics.resident_size == 68);
        // A subsequent GPU eviction removes the advisory hint; exact demand
        // still reloads the original file bytes and can run on CPU normally.
        gpu_one.reset();
        const uint64_t reads_before = joint.statistics().bytes_read;
        auto restored = joint.acquire_pair(gate_one, down_one, 0, "joint-one");
        check(static_cast<bool>(restored));
        check(restored.value().gate_up->mxfp4_blocks.front() == 32);
        check(restored.value().down->mxfp4_scales.front() == 104);
        check(joint.statistics().bytes_read == reads_before + 34);
        check(joint.statistics().resident_size <= joint.capacity());

        auto replacement = std::make_shared<ExpertResidencyCoordinator>();
        joint.set_residency_coordinator(replacement);
        for (const char* key : {"joint-zero", "joint-one", "joint-two"})
            check(!coordinator->host_resident(key));
        check(replacement->host_resident("joint-one"));
        check(replacement->host_resident("joint-zero") == joint.is_ready(gate_zero, down_zero, "joint-zero"));
        check(replacement->host_resident("joint-two") == joint.is_ready(gate_zero, down_one, "joint-two"));
        joint.set_residency_coordinator({});
        for (const char* key : {"joint-zero", "joint-one", "joint-two"})
            check(!replacement->host_resident(key));
    }
    {
        auto coordinator = std::make_shared<ExpertResidencyCoordinator>();
        ExpertCache pinned_joint(68, 1, {}, ExpertIoMode::Buffered);
        pinned_joint.set_residency_coordinator(coordinator);
        {
            auto first = pinned_joint.acquire_pair(gate_zero, down_zero, 0, "pinned-zero");
            check(static_cast<bool>(first));
        }
        auto second = pinned_joint.acquire_pair(gate_one, down_one, 0, "pinned-one");
        check(static_cast<bool>(second));
        auto gpu_pin = coordinator->register_device("pinned-one");
        {
            auto incoming = pinned_joint.acquire_pair(gate_zero, down_one, 0, "pinned-two");
            check(static_cast<bool>(incoming));
        }
        check(!pinned_joint.is_ready(gate_zero, down_zero, "pinned-zero"));
        check(pinned_joint.is_ready(gate_one, down_one, "pinned-one"));
        check(second.value().gate_up_operator != nullptr);
        second.value() = {};
        {
            auto restored = pinned_joint.acquire_pair(gate_zero, down_zero, 0, "pinned-zero");
            check(static_cast<bool>(restored));
        }
        check(!pinned_joint.is_ready(gate_one, down_one, "pinned-one"));
    }
    {
        auto coordinator = std::make_shared<ExpertResidencyCoordinator>();
        ExpertCache arc_joint(68, 1, {}, ExpertIoMode::Buffered);
        arc_joint.set_residency_coordinator(coordinator);
        {
            auto hot = arc_joint.acquire_pair(gate_zero, down_zero, 0, "arc-hot");
            auto repeated = arc_joint.acquire_pair(gate_zero, down_zero, 0, "arc-hot");
            auto cold = arc_joint.acquire_pair(gate_one, down_one, 0, "arc-cold");
            check(hot && repeated && cold);
        }
        auto gpu_hot = coordinator->register_device("arc-hot");
        auto incoming = arc_joint.acquire_pair(gate_zero, down_one, 0, "arc-incoming");
        check(static_cast<bool>(incoming));
        check(arc_joint.is_ready(gate_zero, down_zero, "arc-hot"));
        check(!arc_joint.is_ready(gate_one, down_one, "arc-cold"));
    }
    {
        auto coordinator = std::make_shared<ExpertResidencyCoordinator>();
        ExpertCache no_upload(68, 1, {}, ExpertIoMode::Buffered);
        no_upload.set_residency_coordinator(coordinator);
        {
            auto first = no_upload.acquire_pair(gate_zero, down_zero, 0, "no-upload-zero");
            auto second = no_upload.acquire_pair(gate_one, down_one, 0, "no-upload-one");
            check(first && second);
        }
        // Queued/failed uploads have no ready-device registration, so host
        // ARC follows its ordinary oldest-recent choice without a duplicate
        // reclamation claim or any eager loss of CPU fallback payloads.
        check(!coordinator->device_resident("no-upload-zero") && !coordinator->device_resident("no-upload-one"));
        auto incoming = no_upload.acquire_pair(gate_zero, down_one, 0, "no-upload-two");
        check(static_cast<bool>(incoming));
        check(!no_upload.is_ready(gate_zero, down_zero, "no-upload-zero"));
        check(no_upload.is_ready(gate_one, down_one, "no-upload-one"));
    }
    {
        auto coordinator = std::make_shared<ExpertResidencyCoordinator>();
        ExpertCache failed(34, 1, {}, ExpertIoMode::Buffered);
        failed.set_residency_coordinator(coordinator);
        TensorData missing = gate_zero;
        auto missing_storage = std::make_shared<MxFp4FileStorage>(*missing.mxfp4_file_storage);
        missing_storage->blocks_path = (directory / "missing-joint-cache.bin").string();
        missing.mxfp4_file_storage = std::move(missing_storage);
        auto acquired = failed.acquire_pair(missing, down_zero, 0, "joint-failed");
        check(!acquired && acquired.error().code == ErrorCode::IoError);
        check(!coordinator->host_resident("joint-failed"));
        check(failed.statistics().resident_size == 0);
    }
    {
        auto coordinator = std::make_shared<ExpertResidencyCoordinator>();
        ExpertCacheLease surviving;
        {
            ExpertCache short_lived(34, 1, {}, ExpertIoMode::Buffered);
            short_lived.set_residency_coordinator(coordinator);
            auto acquired = short_lived.acquire_pair(gate_zero, down_zero, 0, "surviving");
            check(static_cast<bool>(acquired));
            surviving = std::move(acquired).value();
            check(coordinator->host_resident("surviving"));
        }
        check(!coordinator->host_resident("surviving"));
        check(surviving.gate_up->mxfp4_blocks.front() == 0);
        surviving = {};
    }
    {
        ExpertCache poll(34, 1, {}, ExpertIoMode::Buffered);
        auto pinned = poll.acquire_pair(gate_zero, down_zero, 0, "poll-zero");
        check(static_cast<bool>(pinned));
        const std::array<ExpertCachePairRequest, 1> requests = {{
            {&gate_one, &down_one, 0, "poll-one"},
        }};
        std::array<ExpertCacheLease, 1> leases;
        auto polled = poll.wait_acquire_ready_pairs(requests, leases, true, 1, false);
        check(polled && polled.value() == 0);
        check(!leases[0].gate_up && poll.statistics().queued_reads == 1);
        pinned.value() = {};
        auto acquired = poll.wait_acquire_ready_pairs(requests, leases);
        check(acquired && acquired.value() == 1);
        check(leases[0].gate_up->mxfp4_blocks.front() == 32);
    }
    {
        auto coordinator = std::make_shared<ExpertResidencyCoordinator>();
        auto blocked_victim = std::make_shared<TestExpertVictimCache>(68);
        auto source = cache.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(source));
        blocked_victim->admit("poll-blocked", source.value().gate_up, source.value().down, {});
        blocked_victim->block_restore("poll-blocked");
        ExpertCache poll(34, 1, blocked_victim, ExpertIoMode::Buffered);
        poll.set_residency_coordinator(coordinator);
        const std::array<ExpertCachePairRequest, 1> requests = {{
            {&gate_zero, &down_zero, 0, "poll-blocked"},
        }};
        std::array<ExpertCacheLease, 1> leases;
        auto polled = poll.wait_acquire_ready_pairs(requests, leases, true, 1, false);
        const bool blocked_entered = blocked_victim->wait_for_blocked_restore();
        auto blocked_polled = poll.wait_acquire_ready_pairs(requests, leases, true, 1, false);
        const bool marked_before_ready = coordinator->host_resident("poll-blocked");
        blocked_victim->release_blocked_restore();
        poll.wait_for_background_work();
        check(polled && polled.value() == 0 && blocked_entered);
        check(blocked_polled && blocked_polled.value() == 0 && !marked_before_ready);
        auto acquired = poll.wait_acquire_ready_pairs(requests, leases, true, 1, false);
        check(acquired && acquired.value() == 1);
        check(coordinator->host_resident("poll-blocked"));
    }

    ExpertCache concurrent(68, 2);
    check(static_cast<bool>(concurrent.request_pair(gate_zero, down_zero)));
    bool first_ok = false;
    bool second_ok = false;
    std::shared_ptr<const TensorData> first_gate_up;
    std::shared_ptr<const TensorData> first_down;
    std::shared_ptr<const TensorData> second_gate_up;
    std::shared_ptr<const TensorData> second_down;
    std::thread first_waiter([&] {
        auto lease = concurrent.acquire_pair(gate_zero, down_zero);
        first_ok = static_cast<bool>(lease);
        if (lease)
        {
            first_gate_up = lease.value().gate_up;
            first_down = lease.value().down;
        }
    });
    std::thread second_waiter([&] {
        auto lease = concurrent.acquire_pair(gate_zero, down_zero);
        second_ok = static_cast<bool>(lease);
        if (lease)
        {
            second_gate_up = lease.value().gate_up;
            second_down = lease.value().down;
        }
    });
    first_waiter.join();
    second_waiter.join();
    check(static_cast<bool>(first_ok));
    check(static_cast<bool>(second_ok));
    check(first_gate_up && first_down);
    check(second_gate_up && second_down);
    check(first_gate_up == second_gate_up);
    check(first_down == second_down);
    check(static_cast<bool>(concurrent.statistics().misses == 1));
    check(static_cast<bool>(concurrent.statistics().hits == 1));
    check(static_cast<bool>(concurrent.statistics().queued_reads == 1));
    check(static_cast<bool>(concurrent.statistics().mapped_ranges == 0));
    check(static_cast<bool>(concurrent.statistics().mapped_bytes == 0));

    {
        auto delayed = std::make_shared<DelayedExpertVictimCache>();
        ExpertCache any_ready(68, 2, delayed);
        const std::array<ExpertCachePairRequest, 2> requests = {{
            {
                &gate_zero,
                &down_zero,
                0,
                "slow",
            },
            {
                &gate_one,
                &down_one,
                0,
                "fast",
            },
        }};
        std::array<ExpertCacheLease, 2> leases;
        auto acquired = any_ready.wait_acquire_ready_pairs(requests, leases);
        check(static_cast<bool>(acquired));
        check(static_cast<bool>(acquired.value() == 1));
        check(static_cast<bool>(!leases[0].gate_up));
        check(static_cast<bool>(leases[1].gate_up));
        check(static_cast<bool>(leases[1].gate_up->mxfp4_blocks.front() == 7));
        check(static_cast<bool>(leases[1].gate_up_operator != nullptr));
        check(static_cast<bool>(leases[1].down_operator != nullptr));

        const std::array<ExpertCachePairRequest, 1> slow_request = {{
            {
                &gate_zero,
                &down_zero,
                0,
                "slow",
            },
        }};
        std::array<ExpertCacheLease, 1> slow_lease;
        acquired = any_ready.wait_acquire_ready_pairs(slow_request, slow_lease);
        check(static_cast<bool>(acquired));
        check(static_cast<bool>(acquired.value() == 1));
        check(static_cast<bool>(slow_lease[0].gate_up));
        check(static_cast<bool>(slow_lease[0].gate_up_operator != nullptr));
        check(static_cast<bool>(slow_lease[0].down_operator != nullptr));
        check(static_cast<bool>(any_ready.statistics().misses == 2));

        ExpertCache front_ready(68, 2, delayed);
        leases = {};
        acquired = front_ready.wait_acquire_ready_pairs(requests, leases, false);
        check(static_cast<bool>(acquired));
        check(static_cast<bool>(acquired.value() == 2));
        check(static_cast<bool>(leases[0].gate_up));
        check(static_cast<bool>(leases[1].gate_up));
        check(static_cast<bool>(leases[0].gate_up_operator != nullptr));
        check(static_cast<bool>(leases[0].down_operator != nullptr));
        check(static_cast<bool>(leases[1].gate_up_operator != nullptr));
        check(static_cast<bool>(leases[1].down_operator != nullptr));

        ExpertCache coalesced_ready(68, 2, delayed);
        leases = {};
        acquired = coalesced_ready.wait_acquire_ready_pairs(requests, leases, true, 8);
        check(acquired && acquired.value() == 2);
        check(leases[0].gate_up && leases[1].gate_up);
        check(coalesced_ready.statistics().queued_reads == 2);

        ExpertCache single_prefix(34, 2, delayed);
        leases = {};
        acquired = single_prefix.wait_acquire_ready_pairs(requests, leases, true, 8);
        check(acquired && acquired.value() == 1);
        check(leases[0].gate_up && !leases[1].gate_up);
        check(single_prefix.statistics().queued_reads == 1);
        check(single_prefix.statistics().resident_size == 34);
        // Reusing a lease array must release even an old pin in its tail.
        // Otherwise the first request below could wait on its own storage.
        std::swap(leases[0], leases[1]);
        const std::array<ExpertCachePairRequest, 2> reversed_requests = {{requests[1], requests[0]}};
        acquired = single_prefix.wait_acquire_ready_pairs(reversed_requests, leases, true, 8);
        check(acquired && acquired.value() == 1);
        check(leases[0].gate_up && !leases[1].gate_up);
        leases = {};

        ExpertCache bounded_span(102, 2, delayed);
        std::array<ExpertCachePairRequest, 3> bounded_requests = {{requests[0], requests[1], {&gate_zero, &down_one, 0, "unrequested"}}};
        acquired = bounded_span.wait_acquire_ready_pairs(std::span(bounded_requests).first(2), leases, true, 8);
        check(acquired && acquired.value() == 2);
        check(bounded_span.statistics().queued_reads == 2);
        check(!bounded_span.is_ready(gate_zero, down_one, "unrequested"));
        leases = {};
        auto invalid_target = bounded_span.wait_acquire_ready_pairs(std::span(bounded_requests).last(1), std::span(leases).first(1), true, 0);
        check(!invalid_target && invalid_target.error().code == ErrorCode::InvalidArgument);
        check(bounded_span.statistics().queued_reads == 2);

        leases = {};
        const std::array<ExpertCachePairRequest, 2> malformed_batch = {{
            bounded_requests[2],
            {nullptr, &down_one, 0, "malformed"},
        }};
        auto invalid_request = bounded_span.wait_acquire_ready_pairs(malformed_batch, leases);
        check(!invalid_request && invalid_request.error().code == ErrorCode::InvalidArgument);
        check(bounded_span.statistics().queued_reads == 2);
        check(std::all_of(leases.begin(), leases.end(), [](const ExpertCacheLease& lease) {
            return !lease.gate_up && !lease.down
                   && lease.gate_up_operator == nullptr && lease.down_operator == nullptr;
        }));
    }

    // A failed read must wake a coalesced wait even while a different admitted
    // read is blocked. Use a barrier rather than a timing-dependent assertion.
    {
        auto blocked_victim = std::make_shared<TestExpertVictimCache>(68);
        auto source = cache.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(source));
        blocked_victim->admit("blocked", source.value().gate_up, source.value().down, {});
        blocked_victim->block_restore("blocked");
        TensorData missing_gate = gate_one;
        auto missing_storage = std::make_shared<MxFp4FileStorage>(*gate_one.mxfp4_file_storage);
        missing_storage->blocks_path = (directory / "missing-ready-blocks.bin").string();
        missing_gate.mxfp4_file_storage = std::move(missing_storage);
        const std::array<ExpertCachePairRequest, 2> requests = {{
            {&gate_zero, &down_zero, 0, "blocked"},
            {&missing_gate, &down_one, 0, "failed"},
        }};
        ExpertCache failed_ready(68, 2, blocked_victim);
        std::array<ExpertCacheLease, 2> leases;
        std::mutex completion_mutex;
        std::condition_variable completion;
        bool completed = false;
        Result<size_t> acquired = Error{ErrorCode::InternalError, "coalesced wait did not complete"};
        std::thread waiter([&] {
            auto result = failed_ready.wait_acquire_ready_pairs(requests, leases, true, 2);
            {
                const std::lock_guard<std::mutex> lock(completion_mutex);
                acquired = std::move(result);
                completed = true;
            }
            completion.notify_all();
        });
        const bool blocked_entered = blocked_victim->wait_for_blocked_restore();
        bool failed_before_release = false;
        {
            std::unique_lock<std::mutex> lock(completion_mutex);
            failed_before_release = completion.wait_for(lock, std::chrono::seconds(5), [&] { return completed; });
        }
        blocked_victim->release_blocked_restore();
        waiter.join();
        failed_ready.wait_for_background_work();
        check(blocked_entered && failed_before_release);
        check(!acquired && acquired.error().code == ErrorCode::IoError);
        check(!leases[0].gate_up && !leases[1].gate_up);
    }

    ExpertCache speculative(68, 1);
    check(static_cast<bool>(speculative.prefetch_pair(gate_zero, down_zero)));
    speculative.wait_for_background_work();
    check(speculative.is_ready(gate_zero, down_zero));
    auto prefetched = speculative.acquire_pair(gate_zero, down_zero);
    check(static_cast<bool>(prefetched));
    check(static_cast<bool>(speculative.statistics().speculative_reads == 1));
    check(static_cast<bool>(speculative.statistics().queued_reads == 1));

    ExpertCache adaptive_replacement(68, 1);
    {
        auto first = adaptive_replacement.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(first));
    }
    check(adaptive_replacement.statistics().misses == 1);
    {
        auto reused = adaptive_replacement.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(reused));
    }
    check(adaptive_replacement.statistics().hits == 1);
    {
        auto single_use = adaptive_replacement.acquire_pair(gate_one, down_one);
        check(static_cast<bool>(single_use));
    }
    {
        auto third = adaptive_replacement.acquire_pair(gate_zero, down_one);
        check(static_cast<bool>(third));
    }
    const uint64_t hits_before_retained = adaptive_replacement.statistics().hits;
    {
        auto retained_hot = adaptive_replacement.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(retained_hot));
    }
    check(adaptive_replacement.statistics().hits == hits_before_retained + 1);
    {
        auto recent_ghost = adaptive_replacement.acquire_pair(gate_one, down_one);
        check(static_cast<bool>(recent_ghost));
        const ExpertCacheStatistics adapted = adaptive_replacement.statistics();
        check(static_cast<bool>(adapted.arc_recent_ghost_hits == 1));
        check(static_cast<bool>(adapted.arc_recent_target_size == 34));
    }
    {
        auto frequent_ghost = adaptive_replacement.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(frequent_ghost));
        const ExpertCacheStatistics adapted = adaptive_replacement.statistics();
        check(static_cast<bool>(adapted.arc_frequent_ghost_hits == 1));
        check(static_cast<bool>(adapted.arc_recent_target_size == 0));
        check(static_cast<bool>(adapted.arc_recent_size + adapted.arc_frequent_size == adapted.resident_size));
    }

    ExpertCache pressure(34, 1);
    auto pinned = pressure.acquire_pair(gate_zero, down_zero);
    check(static_cast<bool>(pinned));
    check(static_cast<bool>(pressure.prefetch_pair(gate_one, down_one)));
    check(static_cast<bool>(pressure.statistics().speculative_reads == 0));
    check(static_cast<bool>(pressure.statistics().dropped_speculative_admissions == 1));
    auto exhausted = pressure.acquire_pair(gate_one, down_one);
    check(static_cast<bool>(!exhausted));
    check(static_cast<bool>(exhausted.error().code == ErrorCode::InvalidArgument));

    ExpertCache speculative_eviction(34,
                                     1,
                                     {},
                                     ExpertIoMode::Auto,
                                     ExpertCacheAllowSpeculativeEviction);
    {
        auto resident = speculative_eviction.acquire_pair(gate_zero, down_zero);
        check(static_cast<bool>(resident));
    }
    check(static_cast<bool>(speculative_eviction.prefetch_pair(gate_one, down_one)));
    speculative_eviction.wait_for_background_work();
    check(static_cast<bool>(speculative_eviction.is_ready(gate_one, down_one)));
    check(static_cast<bool>(speculative_eviction.statistics().speculative_reads == 1));
    check(static_cast<bool>(speculative_eviction.statistics().dropped_speculative_admissions == 0));

    ExpertCache variable_size(52, 1);
    const TensorData small_gate = file_backed(0, 0, 8);
    const TensorData small_down = file_backed(8, 1, 8);
    {
        auto small = variable_size.acquire_pair(small_gate, small_down);
        check(static_cast<bool>(small));
    }
    {
        auto regular = variable_size.acquire_pair(gate_one, down_one);
        check(static_cast<bool>(regular));
    }
    const ExpertCacheStatistics variable_statistics = variable_size.statistics();
    check(static_cast<bool>(variable_statistics.bytes_read == 52));
    check(static_cast<bool>(variable_statistics.resident_size == 52));
    check(static_cast<bool>(variable_statistics.arc_recent_size + variable_statistics.arc_frequent_size == variable_statistics.resident_size));

    ExpertCache oversized_prefetch(17, 1);
    check(static_cast<bool>(oversized_prefetch.prefetch_pair(small_gate, small_down)));
    check(static_cast<bool>(oversized_prefetch.statistics().dropped_speculative_admissions == 1));

    const TensorData truncated_gate = file_backed(60, 0);
    ExpertCache retryable(34, 1);
    auto failed_read = retryable.acquire_pair(truncated_gate, down_zero);
    check(static_cast<bool>(!failed_read));
    check(static_cast<bool>(failed_read.error().code == ErrorCode::IoError));
    auto retried_read = retryable.acquire_pair(truncated_gate, down_zero);
    check(static_cast<bool>(!retried_read));
    check(static_cast<bool>(retried_read.error().code == ErrorCode::IoError));
    check(static_cast<bool>(retryable.statistics().misses == 2));
    check(static_cast<bool>(retryable.statistics().resident_size == 0));

    ExpertCache undersized(33);
    auto rejected = undersized.acquire_pair(gate_zero, down_zero);
    check(static_cast<bool>(!rejected));
    check(static_cast<bool>(rejected.error().code == ErrorCode::InvalidArgument));

    const std::filesystem::path large_blocks_path = directory / "large_blocks.bin";
    const std::filesystem::path large_scales_path = directory / "large_scales.bin";
    constexpr uint64_t large_rows = 393216;
    constexpr uint64_t large_blocks_per_tensor = large_rows * 16;
    constexpr uint64_t large_scales_per_tensor = large_rows;
    {
        std::vector<uint8_t> blocks(large_blocks_per_tensor * 2);
        for (size_t index = 0; index < blocks.size(); ++index)
            blocks[index] = static_cast<uint8_t>((index * 13) % 251);
        std::ofstream stream(large_blocks_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(blocks.data()), static_cast<std::streamsize>(blocks.size()));
    }
    {
        std::vector<uint8_t> scales(large_scales_per_tensor * 2);
        for (size_t index = 0; index < scales.size(); ++index)
            scales[index] = static_cast<uint8_t>((index * 7) % 239);
        std::ofstream stream(large_scales_path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(scales.data()), static_cast<std::streamsize>(scales.size()));
    }

    auto large_file_backed = [&](uint64_t block_offset, uint64_t scale_offset) {
        TensorData tensor;
        tensor.dtype = DType::MxFp4;
        tensor.shape = {large_rows, 32};
        auto storage = std::make_shared<MxFp4FileStorage>();
        storage->blocks_path = large_blocks_path.string();
        storage->blocks_offset = block_offset;
        storage->blocks_size = large_blocks_per_tensor;
        storage->scales_path = large_scales_path.string();
        storage->scales_offset = scale_offset;
        storage->scales_size = large_scales_per_tensor;
        tensor.mxfp4_file_storage = std::move(storage);
        return tensor;
    };
    const TensorData large_gate = large_file_backed(0, 0);
    const TensorData large_down = large_file_backed(large_blocks_per_tensor, large_scales_per_tensor);
    const uint64_t large_pair_bytes = (large_blocks_per_tensor + large_scales_per_tensor) * 2;
    ExpertCache large_reads(large_pair_bytes, 2, {}, ExpertIoMode::Mmap);
    auto large_pair = large_reads.acquire_pair(large_gate, large_down);
    check(static_cast<bool>(large_pair));
    check(static_cast<bool>(large_reads.statistics().mapped_ranges == 4));
    check(static_cast<bool>(large_reads.statistics().mapped_bytes == large_pair_bytes));
    check(static_cast<bool>(large_pair.value().gate_up->mxfp4_blocks.front() == 0));
    check(static_cast<bool>(large_pair.value().gate_up->mxfp4_blocks.back() == static_cast<uint8_t>(((large_blocks_per_tensor - 1) * 13) % 251)));
    check(static_cast<bool>(large_pair.value().down->mxfp4_scales.back() == static_cast<uint8_t>(((large_scales_per_tensor * 2 - 1) * 7) % 239)));

    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
}

} // namespace moe
} // namespace ncnn
