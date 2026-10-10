#include "engine/expert.h"
#include "engine/executor.h"
#include "backends/ncnn/modelpipeline.h"
#include "backends/ncnn/vulkan.h"
#include "graph/compiler.h"
#include "kernels/bfloat16.h"
#include "engine/expertbackend.h"
#include "engine/sessionstate.h"
#include "graph/compiledmodel.h"
#include "kernels/ops.h"
#include "ncnn/moe/session.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

static void pipeline_check(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

class PipelineTestBackend final : public ExpertBackend
{
public:
    struct Entry
    {
        std::shared_ptr<const TensorData> gate;
        std::shared_ptr<const TensorData> down;
        ExpertActivation activation = ExpertActivation::DeepSeekSwiGlu;
        float activation_limit = 0.0f;
    };

    class Demand final : public ExpertDemandSubmission
    {
    public:
        Demand(PipelineTestBackend& _owner, std::vector<std::shared_ptr<const void>> _pins, size_t _prepared, bool _invalid)
            : owner(_owner), pins(std::move(_pins)), prepared(_prepared), invalid(_invalid)
        {
            ++owner.demands_in_flight;
        }

        ~Demand() override
        {
            --owner.demands_in_flight;
        }

        size_t wait(std::span<std::shared_ptr<const void>> outputs) override
        {
            for (auto& output : outputs) output.reset();
            if (aborted || outputs.size() != pins.size()) return 0;
            if (owner.demand_wait_observer) owner.demand_wait_observer();
            ++owner.demand_waits;
            for (size_t index = 0; index < prepared; ++index) outputs[index] = pins[index];
            return invalid ? outputs.size() + 1 : prepared;
        }

        void abort() noexcept override
        {
            aborted = true;
            pins.clear();
        }

    private:
        PipelineTestBackend& owner;
        std::vector<std::shared_ptr<const void>> pins;
        size_t prepared = 0;
        bool invalid = false;
        bool aborted = false;
    };

    class Compute final : public ExpertSubmission
    {
    public:
        Compute(PipelineTestBackend& _owner, std::span<const ExpertBackendRequest> _requests)
            : owner(_owner), requests(_requests.begin(), _requests.end()), entries(requests.size()), outputs(requests.size()), planned(requests.size(), ExpertBackendExecutionResult::NotResident)
        {
            for (size_t index = 0; index < requests.size(); ++index)
            {
                const auto found = owner.resident.find(std::string(requests[index].key));
                if (found == owner.resident.end()) continue;
                entries[index] = found->second;
                planned[index] = ExpertBackendExecutionResult::Executed;
                live = true;
            }
            initial = owner.submission_count++ == 0;
            if (live)
            {
                ++owner.in_flight;
                if (initial) owner.initial_in_flight = true;
            }
        }

        ~Compute() override
        {
            if (!waited) (void)wait();
            if (!committed) abort();
        }

        std::span<const ExpertBackendExecutionResult> reservations() const noexcept override
        {
            return planned;
        }

        std::vector<ExpertBackendExecutionResult> wait() override
        {
            if (waited) return planned;
            if (live && owner.compute_wait_observer) owner.compute_wait_observer();
            for (size_t index = 0; index < entries.size(); ++index)
            {
                if (!entries[index]) continue;
                ActivationBuffer activated;
                forward_gate_up_mxfp4(*entries[index]->gate, nullptr, *requests[index].input,
                                      entries[index]->activation, entries[index]->activation_limit,
                                      activated, owner.flags);
                outputs[index] = forward_linear(*entries[index]->down, activated, owner.flags);
                ++owner.executed;
            }
            if (live)
            {
                --owner.in_flight;
                if (initial) owner.initial_in_flight = false;
            }
            waited = true;
            return planned;
        }

        bool commit() override
        {
            if (!waited) (void)wait();
            if (aborted) return false;
            if (committed) return true;
            const size_t commit_index = owner.commit_attempts++;
            if (owner.fail_commit || (owner.fail_initial_commit && commit_index == 0)) return false;
            for (size_t index = 0; index < entries.size(); ++index)
            {
                if (!entries[index]) continue;
                requests[index].output->swap(outputs[index]);
                if (requests[index].device_output) requests[index].device_output->reset();
            }
            committed = true;
            return true;
        }

        void abort() noexcept override
        {
            aborted = true;
        }

    private:
        PipelineTestBackend& owner;
        std::vector<ExpertBackendRequest> requests;
        std::vector<std::shared_ptr<Entry>> entries;
        std::vector<ActivationBuffer> outputs;
        std::vector<ExpertBackendExecutionResult> planned;
        bool initial = false;
        bool live = false;
        bool waited = false;
        bool committed = false;
        bool aborted = false;
    };

    explicit PipelineTestBackend(size_t _capacity)
        : capacity(_capacity)
    {
    }

    void admit(std::string, std::shared_ptr<const TensorData>, const TensorData*,
               std::shared_ptr<const TensorData>, const TensorData*, uint32_t, float, ExpertActivation) override
    {
    }

    size_t prepare_demand_batch(std::span<const ExpertDemandRequest> requests,
                                std::span<std::shared_ptr<const void>> pins) override
    {
        for (auto& pin : pins) pin.reset();
        if (pins.size() != requests.size()) return 0;
        maximum_wave = std::max(maximum_wave, requests.size());
        size_t prepared = 0;
        for (const auto& request : requests)
        {
            if (request.key == failed_key || capacity == 0) break;
            auto found = resident.find(std::string(request.key));
            if (found == resident.end())
            {
                while (resident.size() >= capacity)
                {
                    auto victim = std::find_if(resident.begin(), resident.end(), [](const auto& candidate) {
                        return candidate.second.use_count() == 1;
                    });
                    if (victim == resident.end()) break;
                    resident.erase(victim);
                }
                if (resident.size() >= capacity)
                {
                    ++capacity_retries;
                    break;
                }
                auto entry = std::make_shared<Entry>();
                entry->gate = request.gate_up;
                entry->down = request.down;
                entry->activation = request.activation;
                entry->activation_limit = request.activation_limit;
                found = resident.emplace(std::string(request.key), std::move(entry)).first;
            }
            pins[prepared++] = found->second;
        }
        return prepared;
    }

    std::unique_ptr<ExpertDemandSubmission> begin_demand_batch(std::span<const ExpertDemandRequest> requests) override
    {
        if (in_flight != 0) ++overlap_begins;
        if (initial_in_flight) ++initial_overlap_begins;
        if (synchronous_default) return ExpertBackend::begin_demand_batch(requests);
        std::vector<std::shared_ptr<const void>> pins(requests.size());
        const size_t prepared = prepare_demand_batch(requests, pins);
        return std::make_unique<Demand>(*this, std::move(pins), prepared, invalid_demand);
    }

    std::unique_ptr<ExpertSubmission> submit_batch(std::span<const ExpertBackendRequest> requests) override
    {
        return std::make_unique<Compute>(*this, requests);
    }

    void wait_for_background_work() override
    {
    }

    ExpertBackendStatistics statistics() const override
    {
        ExpertBackendStatistics stats;
        stats.executions = executed;
        return stats;
    }

    const uint64_t flags = OptimizationDefaultFlags;
    size_t in_flight = 0;
    size_t submission_count = 0;
    size_t commit_attempts = 0;
    size_t initial_overlap_begins = 0;
    bool initial_in_flight = false;
    size_t overlap_begins = 0;
    size_t capacity_retries = 0;
    size_t maximum_wave = 0;
    size_t demands_in_flight = 0;
    size_t demand_waits = 0;
    uint64_t executed = 0;
    std::string failed_key;
    bool fail_commit = false;
    bool fail_initial_commit = false;
    bool invalid_demand = false;
    bool synchronous_default = false;
    std::function<void()> demand_wait_observer;
    std::function<void()> compute_wait_observer;

private:
    size_t capacity = 0;
    std::unordered_map<std::string, std::shared_ptr<Entry>> resident;
};

static TensorData pipeline_weight(uint32_t rows, uint32_t columns, uint32_t seed)
{
    TensorData result;
    result.dtype = DType::MxFp4;
    result.shape = {rows, columns};
    result.mxfp4_blocks.resize(static_cast<size_t>(rows) * columns / 2);
    result.mxfp4_scales.resize(static_cast<size_t>(rows) * columns / 32);
    std::fill_n(result.mxfp4_scales.data(), result.mxfp4_scales.size(), uint8_t(123));
    for (size_t index = 0; index < result.mxfp4_blocks.size(); ++index)
        result.mxfp4_blocks[index] = static_cast<uint8_t>((index * 7 + seed * 3) % 16
                                                          | ((index * 11 + seed * 5) % 16) << 4);
    return result;
}

static void pipeline_engine(size_t capacity, bool fail_commit = false, bool invalid_demand = false,
                            bool synchronous_default = false, bool reject_one = false,
                            bool resident_first = false, bool fail_initial_commit = false,
                            bool independent_shared = false, bool throw_shared = false,
                            bool resident_all = false)
{
    static constexpr size_t expert_count = 12;
    auto backend = std::make_shared<PipelineTestBackend>(capacity);
    backend->fail_commit = fail_commit;
    backend->fail_initial_commit = fail_initial_commit;
    backend->invalid_demand = invalid_demand;
    backend->synchronous_default = synchronous_default;
    if (reject_one) backend->failed_key = "pipeline-2";
    CompiledModel model;
    model.descriptor.hidden_size = 32;
    model.opt.hybrid_mode = HybridMode::HybridExperts;
    model.opt.optimization_flags = backend->flags;
    model.expert_backend = backend;
    auto gate = model.weights.add("gate", pipeline_weight(64, 32, 1));
    auto down = model.weights.add("down", pipeline_weight(32, 32, 3));
    pipeline_check(gate && down, "pipeline fixture weights");
    model.operators.bind_weight_count(model.weights.size());
    MoeBlockPlan moe;
    moe.experts.resize(expert_count);
    LayerState state;
    state.normalized.reset(3, 32, false);
    state.resize_experts(expert_count);
    for (size_t row = 0; row < 3; ++row)
        for (uint32_t column = 0; column < 32; ++column)
            state.normalized.row(row)[column] = static_cast<float>(static_cast<int>((row * 3 + column) % 13) - 6) * 0.015625f;
    for (size_t index = 0; index < expert_count; ++index)
    {
        auto& expert = moe.experts[index];
        expert.gate_up_weight = gate.value();
        expert.down_weight = down.value();
        expert.activation = ExpertActivation::DeepSeekSwiGlu;
        expert.activation_limit = 4.0f;
        expert.cache_key = "pipeline-" + std::to_string(index);
        auto& active = state.active_experts()[index];
        active.batch.expert_id = static_cast<uint32_t>(index);
        active.batch.routes = {{static_cast<uint32_t>(index % 3), 0, 1.0f}};
        active.output.reset(1, 32, false);
        std::fill_n(active.output.row(0), 32, -123.0f);
    }
    if (resident_first || resident_all)
    {
        const auto resident_gate = std::make_shared<const TensorData>(model.weights.at(gate.value()));
        const auto resident_down = std::make_shared<const TensorData>(model.weights.at(down.value()));
        for (size_t index = 0; index < (resident_all ? expert_count : size_t(1)); ++index)
        {
            const std::string key = "pipeline-" + std::to_string(index);
            const ExpertDemandRequest resident_request{key, resident_gate, nullptr, resident_down,
                                                       nullptr, 0, 4.0f, ExpertActivation::DeepSeekSwiGlu};
            std::array<std::shared_ptr<const void>, 1> pins;
            pipeline_check(backend->prepare_demand_batch(std::span<const ExpertDemandRequest>(&resident_request, 1), pins) == 1,
                           "resident fixture seeds the requested GPU pairs");
        }
    }
    ExpertScratch scratch;
    SessionStatistics statistics;
    size_t shared_calls = 0;
    std::function<void()> shared_work;
    if (independent_shared)
    {
        shared_work = [&] {
            ++shared_calls;
            pipeline_check(backend->demands_in_flight != 0 && backend->demand_waits == 0,
                           "Shared starts with an issued demand ticket before its completion wait");
            pipeline_check(!state.experts_executed,
                           "Shared overlaps routed supply instead of following the completed Expert group");
            if (throw_shared) throw std::runtime_error("injected Shared failure");
            state.shared_expert_output.reset(state.normalized.rows(), state.normalized.columns(), false);
            for (size_t row = 0; row < state.normalized.rows(); ++row)
                for (uint32_t column = 0; column < state.normalized.columns(); ++column)
                    state.shared_expert_output.row(row)[column] = state.normalized.row(row)[column] * 0.25f;
        };
    }
    bool threw = false;
    try
    {
        auto result = forward_moe(model, moe, state, statistics, scratch, 0, ExecutionBackend::Vulkan, false, shared_work);
        pipeline_check(static_cast<bool>(result) && state.experts_executed, "pipeline forward completed");
    }
    catch (const std::runtime_error& exception)
    {
        if (!throw_shared || std::string(exception.what()) != "injected Shared failure") throw;
        threw = true;
    }
    pipeline_check(backend->demands_in_flight == 0 && backend->in_flight == 0,
                   "Shared completion or exception releases upload and compute ticket lifetimes");
    pipeline_check(threw == throw_shared, "Shared exceptions preserve execution failure semantics");
    if (independent_shared)
        pipeline_check(shared_calls == (resident_all ? 0 : 1),
                       "independent Shared runs once only when a cold demand ticket exists");
    if (throw_shared) return;
    for (const auto& active : state.active_experts())
    {
        ActivationBuffer activated;
        forward_gate_up_mxfp4(model.weights.at(gate.value()), nullptr, active.input,
                              ExpertActivation::DeepSeekSwiGlu, 4.0f, activated, backend->flags);
        const auto expected = forward_linear(model.weights.at(down.value()), activated, backend->flags);
        pipeline_check(active.output.rows() == 1 && active.output.columns() == 32, "pipeline output shape");
        for (size_t index = 0; index < expected.values().size(); ++index)
            pipeline_check(std::isfinite(active.output.values()[index])
                               && std::abs(active.output.values()[index] - expected.values()[index]) < 0.000001f,
                           "pipeline output matches independent CPU oracle including fallback");
    }
    pipeline_check(backend->in_flight == 0 && backend->maximum_wave <= 8, "pipeline drains bounded compute and upload waves");
    const uint64_t expected_gpu = capacity == 0 || invalid_demand
                                      ? (resident_first ? 1 : 0)
                                      : expert_count - (reject_one ? 1 : 0) + (resident_first && fail_initial_commit ? 1 : 0);
    pipeline_check(backend->executed == expected_gpu, "transient capacity pressure cannot cause CPU fallback");
    if (capacity != 0 && !invalid_demand && !resident_all)
        pipeline_check(backend->overlap_begins != 0, "next upload begins before current compute wait");
    if (capacity == 1 && !invalid_demand)
        pipeline_check(backend->capacity_retries != 0, "one-pair GPU cache exercises release and retry");
    if (resident_first)
        pipeline_check(backend->initial_overlap_begins == 1,
                       "first bounded cold upload starts before initial resident compute wait");
}

#if NCNN_MOE_WITH_VULKAN
static TensorData pipeline_bfloat16(std::vector<uint32_t> shape, float diagonal, bool vector = false)
{
    TensorData result;
    result.dtype = DType::BFloat16;
    result.shape = std::move(shape);
    result.bfloat16_data.assign(result.element_count(), 0);
    if (vector)
        std::fill(result.bfloat16_data.begin(), result.bfloat16_data.end(), float_to_bfloat16(diagonal));
    else
        for (uint32_t row = 0; row < result.shape[0]; ++row)
            result.bfloat16_data[static_cast<size_t>(row) * result.shape[1] + row % result.shape[1]] = float_to_bfloat16(diagonal);
    return result;
}

static TensorData pipeline_shared_weight(uint32_t size, float diagonal, bool gpu_shared)
{
    if (gpu_shared) return pipeline_bfloat16({size, size}, diagonal);
    TensorData result;
    result.dtype = DType::Float32;
    result.shape = {size, size};
    result.float32_data.assign(result.element_count(), 0.0f);
    for (uint32_t row = 0; row < size; ++row)
        result.float32_data[static_cast<size_t>(row) * size + row] = diagonal;
    return result;
}

static CompiledModel pipeline_host_shared_model(bool enabled, bool gpu_shared, bool cpu_only)
{
    constexpr uint32_t hidden = 128;
    constexpr uint32_t vocabulary = 4;
    MoeModelDescriptor descriptor;
    descriptor.model_type = "host_shared_pipeline_test";
    descriptor.vocabulary_size = vocabulary;
    descriptor.hidden_size = hidden;
    descriptor.intermediate_size = hidden;
    descriptor.expert_count = 2;
    descriptor.experts_per_token = 1;
    descriptor.hash_routing_layer_count = 1;
    descriptor.activation_dtype = DType::BFloat16;
    descriptor.final_norm = NormType::RmsNorm;
    descriptor.layers.resize(1);
    descriptor.layers.front().attention.kind = AttentionKind::None;
    descriptor.layers.front().pre_ffn_norm = NormType::RmsNorm;
    auto& moe = descriptor.layers.front().moe;
    moe.expert_count = 2;
    moe.top_k = 1;
    moe.intermediate_size = hidden;
    moe.activation = ExpertActivation::Silu;
    moe.layout = ExpertLayout::PackedGateUpDown;
    moe.expert_weight_dtype = DType::MxFp4;
    moe.shared_expert_count = 1;
    moe.shared_expert_weight_dtype = gpu_shared ? DType::BFloat16 : DType::Float32;
    moe.flags = MoeDescriptorSharedExpertGate;
    WeightMapping weights;
    auto embedding = pipeline_bfloat16({vocabulary, hidden}, 0.0f);
    for (uint32_t row = 0; row < vocabulary; ++row)
        for (uint32_t column = 0; column < hidden; ++column)
            embedding.bfloat16_data[static_cast<size_t>(row) * hidden + column] = float_to_bfloat16(static_cast<float>(static_cast<int>((row * 5 + column) % 13) - 6) * 0.015625f);
    weights.emplace("token_embedding.weight", std::move(embedding));
    weights.emplace("layers.0.pre_ffn_norm.weight", pipeline_bfloat16({hidden}, 1.0f, true));
    weights.emplace("layers.0.router.weight", pipeline_bfloat16({2, hidden}, 0.0f));
    TensorData routes;
    routes.dtype = DType::Int64;
    routes.shape = {vocabulary, 1};
    routes.int64_data = {0, 1, 0, 1};
    weights.emplace("layers.0.router.token_experts", std::move(routes));
    weights.emplace("final_norm.weight", pipeline_bfloat16({hidden}, 1.0f, true));
    weights.emplace("lm_head.weight", pipeline_bfloat16({vocabulary, hidden}, 1.0f));
    for (uint32_t expert = 0; expert < 2; ++expert)
    {
        const std::string prefix = "layers.0.experts." + std::to_string(expert) + ".";
        weights.emplace(prefix + "gate_up.weight", pipeline_weight(hidden * 2, hidden, expert + 1));
        weights.emplace(prefix + "down.weight", pipeline_weight(hidden, hidden, expert + 3));
    }
    weights.emplace("layers.0.shared_expert.gate.weight", pipeline_shared_weight(hidden, 0.5f, gpu_shared));
    weights.emplace("layers.0.shared_expert.up.weight", pipeline_shared_weight(hidden, 0.75f, gpu_shared));
    weights.emplace("layers.0.shared_expert.down.weight", pipeline_shared_weight(hidden, 0.25f, gpu_shared));
    weights.emplace("layers.0.shared_expert.router_gate.weight", pipeline_bfloat16({1, hidden}, 0.125f));
    CompilerOption option;
    if (!enabled) option.optimization_flags &= ~OptimizationVulkanSharedExpertOverlap;
    // Keep routed GPU eligibility independent of optional Shared capability.
    if (!cpu_only) option.flags |= BackendVulkanExperts | BackendVulkanDense;
    option.device_index = 0;
    option.vulkan_runtime = create_vulkan_runtime();
    CompiledModel model;
    const auto compiled = compile_model(std::move(descriptor), std::move(weights), model,
                                        cpu_only ? HybridMode::CpuOnly : HybridMode::HybridExperts, option);
    pipeline_check(static_cast<bool>(compiled), "host Shared executor model compiles");
    pipeline_check(support_vulkan_shared_experts(model.operators, model.graph.layer_plans.front().moe) == gpu_shared,
                   "host Shared capability is backed by the BF16 fused operator");
    return model;
}

static std::vector<float> pipeline_host_shared_case(bool staged, bool enabled, bool resident,
                                                    bool gpu_shared = true, bool cpu_only = false)
{
    auto model = pipeline_host_shared_model(enabled, gpu_shared, cpu_only);
    auto backend = std::make_shared<PipelineTestBackend>(2);
    model.expert_backend = backend;
    auto& moe = model.graph.layer_plans.front().moe;
    for (size_t index = 0; index < moe.experts.size(); ++index)
    {
        auto& expert = moe.experts[index];
        expert.cache_key = "host-shared-" + std::to_string(index);
        if (resident)
        {
            const ExpertDemandRequest request{expert.cache_key,
                                              std::make_shared<TensorData>(model.weights.at(expert.gate_up_weight)), nullptr,
                                              std::make_shared<TensorData>(model.weights.at(expert.down_weight)), nullptr,
                                              0, expert.activation_limit, expert.activation};
            std::array<std::shared_ptr<const void>, 1> pins;
            pipeline_check(backend->prepare_demand_batch(std::span<const ExpertDemandRequest>(&request, 1), pins) == 1,
                           "host Shared resident fixture prepares both pairs");
        }
    }
    std::array<SessionState, 2> states;
    std::array<SessionStatistics, 2> statistics;
    const bool early = enabled && gpu_shared && !resident && !cpu_only;
    bool observed_host_input = true;
    bool timing_valid = true;
    size_t demand_observations = 0;
    size_t compute_observations = 0;
    const auto observe = [&] {
        observed_host_input = observed_host_input && !states[0].execution_state.normalized_device;
        timing_valid = timing_valid
                       && (states[0].execution_state.shared_expert_output.rows() != 0) == early;
        if (staged)
            timing_valid = timing_valid && (states[1].execution_state.shared_expert_output.rows() != 0) == early;
    };
    backend->demand_wait_observer = [&] {
        ++demand_observations;
        observe();
    };
    backend->compute_wait_observer = [&] {
        ++compute_observations;
        observe();
    };
    std::vector<float> actual;
    if (staged)
    {
        BatchWorkspace workspace;
        const std::array<DecodeBatchEntry, 2> entries = {{
            {0, &statistics[0], &states[0], 0},
            {1, &statistics[1], &states[1], 0},
        }};
        auto result = forward_decode_batch(model, entries, workspace);
        pipeline_check(result && result.value().size() == 2, "host Shared staged executor completes");
        for (const auto& row : result.value()) actual.insert(actual.end(), row.begin(), row.end());
    }
    else
    {
        const std::array<int32_t, 2> tokens = {0, 1};
        const auto result = forward_model(model, tokens, statistics[0], states[0], 0);
        pipeline_check(static_cast<bool>(result), "host Shared single executor completes");
        actual.assign(states[0].logits.values().begin(), states[0].logits.values().end());
    }
    const size_t expected_demands = resident || cpu_only ? 0 : 1;
    const size_t expected_compute = cpu_only ? 0 : 1;
    const std::string case_info = std::string(staged ? "staged" : "single")
                                  + " enabled=" + std::to_string(enabled)
                                  + " resident=" + std::to_string(resident)
                                  + " gpu_shared=" + std::to_string(gpu_shared)
                                  + " cpu_only=" + std::to_string(cpu_only)
                                  + " demand=" + std::to_string(demand_observations)
                                  + "/" + std::to_string(expected_demands)
                                  + " compute=" + std::to_string(compute_observations)
                                  + "/" + std::to_string(expected_compute);
    std::cout << "host Shared case: " << case_info << '\n';
    pipeline_check(observed_host_input && timing_valid,
                   ("host-normalized Shared completes after demand issue and before routed completion only when eligible: " + case_info).c_str());
    pipeline_check(demand_observations == expected_demands,
                   ("host Shared demand timing observation: " + case_info).c_str());
    pipeline_check(compute_observations == expected_compute,
                   ("host Shared resident execution timing observation: " + case_info).c_str());
    pipeline_check(backend->demands_in_flight == 0 && backend->in_flight == 0,
                   "host Shared executor drains both ticket lifetimes");
    return actual;
}

static void pipeline_host_shared_tests()
{
    for (bool staged : {false, true})
    {
        const auto expected = pipeline_host_shared_case(staged, false, false);
        for (const auto& actual : {
                 pipeline_host_shared_case(staged, true, false),
                 pipeline_host_shared_case(staged, true, true),
                 pipeline_host_shared_case(staged, true, false, false),
                 pipeline_host_shared_case(staged, true, false, false, true),
             })
        {
            pipeline_check(actual.size() == expected.size(), "host Shared executor logits shape");
            for (size_t index = 0; index < expected.size(); ++index)
                pipeline_check(std::isfinite(actual[index]) && std::abs(actual[index] - expected[index]) < 0.003f,
                               "host Shared flag, residency and capability cases preserve logits");
        }
    }
}
#endif

static void pipeline_multidevice_prefix()
{
    auto first = std::make_shared<PipelineTestBackend>(1);
    auto second = std::make_shared<PipelineTestBackend>(2);
    auto backend = std::make_shared<MultiDeviceExpertBackend>(std::vector<std::shared_ptr<ExpertBackend>>{first, second},
                                                              std::vector<uint32_t>{0, 1}, std::vector<uint32_t>{0, 1}, false);
    auto gate = std::make_shared<TensorData>(pipeline_weight(64, 32, 1));
    auto down = std::make_shared<TensorData>(pipeline_weight(32, 32, 3));
    const std::array<ExpertDemandRequest, 4> requests = {{
        {"first-A", gate, nullptr, down, nullptr, 0, 4.0f, ExpertActivation::DeepSeekSwiGlu},
        {"first-B", gate, nullptr, down, nullptr, 0, 4.0f, ExpertActivation::DeepSeekSwiGlu},
        {"second-A", gate, nullptr, down, nullptr, 1, 4.0f, ExpertActivation::DeepSeekSwiGlu},
        {"second-B", gate, nullptr, down, nullptr, 1, 4.0f, ExpertActivation::DeepSeekSwiGlu},
    }};
    auto upload = backend->begin_demand_batch(requests);
    std::array<std::shared_ptr<const void>, 3> wrong_shape;
    pipeline_check(upload && upload->wait(wrong_shape) == 0, "demand shape mismatch cannot publish pins");
    std::array<std::shared_ptr<const void>, 4> pins;
    pipeline_check(upload->wait(pins) == 1 && pins[0] && !pins[1] && !pins[2] && !pins[3],
                   "multi-device async demand returns only continuous prefix");
    auto repeated = pins[0];
    pipeline_check(upload->wait(pins) == 1 && pins[0] == repeated, "completed demand wait is idempotent");
    upload->abort();
    pipeline_check(upload->wait(pins) == 0 && !pins[0] && !pins[1] && !pins[2] && !pins[3],
                   "aborted demand cannot republish a resident pin");
}

} // namespace moe
} // namespace ncnn

int main()
{
    try
    {
        ncnn::moe::pipeline_engine(16);
        ncnn::moe::pipeline_engine(1);
        ncnn::moe::pipeline_engine(0);
        ncnn::moe::pipeline_engine(16, true);
        ncnn::moe::pipeline_engine(16, false, true);
        ncnn::moe::pipeline_engine(16, false, false, true);
        ncnn::moe::pipeline_engine(16, false, false, false, true);
        ncnn::moe::pipeline_engine(16, false, false, false, false, true);
        ncnn::moe::pipeline_engine(16, false, false, false, false, true, true);
        ncnn::moe::pipeline_engine(1, false, false, false, false, true);
        ncnn::moe::pipeline_engine(1, false, false, false, false, true, true);
        ncnn::moe::pipeline_engine(16, false, false, false, false, false, false, true);
        ncnn::moe::pipeline_engine(16, false, false, false, false, true, false, true);
        ncnn::moe::pipeline_engine(1, false, false, false, false, true, true, true);
        ncnn::moe::pipeline_engine(16, false, true, false, false, false, false, true);
        ncnn::moe::pipeline_engine(16, false, false, false, false, true, false, true, true);
        ncnn::moe::pipeline_engine(16, false, false, false, false, false, false, true, true);
        ncnn::moe::pipeline_engine(16, false, false, false, false, false, false, true, false, true);
        ncnn::moe::pipeline_multidevice_prefix();
#if NCNN_MOE_WITH_VULKAN
        if (ncnn::moe::get_gpu_count() != 0) ncnn::moe::pipeline_host_shared_tests();
#endif
        std::cout << "Expert transfer/compute pipeline tests passed\n";
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
