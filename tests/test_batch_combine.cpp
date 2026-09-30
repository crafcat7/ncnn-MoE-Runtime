#include "engine/executor.h"

#include "engine/expert.h"
#include "engine/expertbackend.h"
#include "engine/sessionstate.h"
#include "graph/compiler.h"
#include "kernels/bfloat16.h"
#include "kernels/fastmath.h"
#include "ncnn/moe/session.h"
#include "kernels/ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <stdexcept>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ncnn {
namespace moe {

static constexpr uint32_t hidden_size = 128;
static constexpr uint32_t vocabulary_size = 4;
static constexpr uint32_t expert_count = 2;

static void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

static TensorData make_float_tensor(std::vector<uint32_t> shape, float value)
{
    TensorData tensor;
    tensor.dtype = DType::Float32;
    tensor.shape = std::move(shape);
    tensor.float32_data.assign(tensor.element_count(), value);
    return tensor;
}

static CompiledModel make_cpu_model()
{
    MoeModelDescriptor descriptor;
    descriptor.model_type = "batch_combine_test";
    descriptor.vocabulary_size = vocabulary_size;
    descriptor.hidden_size = hidden_size;
    descriptor.intermediate_size = hidden_size;
    descriptor.expert_count = expert_count;
    descriptor.experts_per_token = 1;
    descriptor.hash_routing_layer_count = 1;
    descriptor.final_norm = NormType::RmsNorm;
    descriptor.layers.resize(1);
    descriptor.layers.front().pre_ffn_norm = NormType::RmsNorm;
    MoeDescriptor& moe = descriptor.layers.front().moe;
    moe.expert_count = expert_count;
    moe.top_k = 1;
    moe.intermediate_size = hidden_size;
    moe.activation = ExpertActivation::Silu;
    moe.layout = ExpertLayout::PackedGateUpDown;
    moe.expert_weight_dtype = DType::BFloat16;
    moe.normalization = RouterNormalization::SelectedExperts;

    WeightMapping mapping;
    TensorData embedding = make_float_tensor({vocabulary_size, hidden_size}, 0.0f);
    for (uint32_t token = 0; token < vocabulary_size; ++token)
        for (uint32_t column = 0; column < hidden_size; ++column)
            embedding.float32_data[static_cast<size_t>(token) * hidden_size + column] = static_cast<float>((token + 1) * (column % 5 + 1)) * (0.015625f * static_cast<float>((column % 7) + 1));
    mapping.emplace("token_embedding.weight", std::move(embedding));
    mapping.emplace("layers.0.pre_ffn_norm.weight", make_float_tensor({hidden_size}, 1.0f));
    // Explicit token routes select the experts; router logits are immaterial.
    mapping.emplace("layers.0.router.weight", make_float_tensor({expert_count, hidden_size}, 0.0f));
    TensorData token_experts;
    token_experts.dtype = DType::Int64;
    token_experts.shape = {vocabulary_size, 1};
    token_experts.int64_data = {0, 1, 0, 1};
    mapping.emplace("layers.0.router.token_experts", std::move(token_experts));
    mapping.emplace("final_norm.weight", make_float_tensor({hidden_size}, 1.0f));
    mapping.emplace("lm_head.weight", make_float_tensor({vocabulary_size, hidden_size}, 0.0f));
    for (uint32_t token = 0; token < vocabulary_size; ++token)
    {
        TensorData& lm_head = mapping.at("lm_head.weight");
        lm_head.float32_data[static_cast<size_t>(token) * hidden_size + token] = 1.0f;
    }
    for (uint32_t expert = 0; expert < expert_count; ++expert)
    {
        const std::string prefix = "layers.0.experts." + std::to_string(expert) + ".";
        const float scale = expert == 0 ? 0.75f : 1.25f;
        TensorData gate_up;
        gate_up.dtype = DType::BFloat16;
        gate_up.shape = {hidden_size * 2, hidden_size};
        gate_up.bfloat16_data.resize(gate_up.element_count());
        TensorData down;
        down.dtype = DType::BFloat16;
        down.shape = {hidden_size, hidden_size};
        down.bfloat16_data.resize(down.element_count());
        for (uint32_t column = 0; column < hidden_size; ++column)
        {
            gate_up.bfloat16_data[static_cast<size_t>(column) * hidden_size + column] = float_to_bfloat16(scale);
            gate_up.bfloat16_data[static_cast<size_t>(column + hidden_size) * hidden_size + column] = float_to_bfloat16(1.0f + 0.25f * expert);
            down.bfloat16_data[static_cast<size_t>(column) * hidden_size + column] = float_to_bfloat16(1.0f - 0.125f * expert);
        }
        mapping.emplace(prefix + "gate_up.weight", std::move(gate_up));
        mapping.emplace(prefix + "down.weight", std::move(down));
    }

    CompiledModel model;
    auto compiled = compile_model(std::move(descriptor), std::move(mapping), model, HybridMode::CpuOnly);
    if (!compiled)
        throw std::runtime_error("batch combine test model compilation failed: " + compiled.error().message);
    return model;
}

struct ExpertMatrices
{
    const TensorData* gate_up = nullptr;
    const TensorData* down = nullptr;
};

struct FakeBackendState
{
    size_t submit_count = 0;
    size_t wait_count = 0;
    size_t commit_count = 0;
    size_t abort_count = 0;
    size_t request_count = 0;
    size_t aggregate_published = 0;
    size_t raw_published = 0;
    bool saw_route_aggregation = false;
    bool saw_require_all_requests = false;
};

static ActivationBuffer expert_output(const ExpertMatrices& expert,
                                      const ActivationBuffer& input,
                                      uint64_t optimization_flags)
{
    const ActivationBuffer gate_up = linear_batch(*expert.gate_up, input, optimization_flags);
    const uint32_t intermediate = expert.gate_up->shape.front() / 2;
    ActivationBuffer activated(input.rows(), intermediate);
    for (size_t row = 0; row < input.rows(); ++row)
    {
        const float* source = gate_up.row(row);
        float* destination = activated.row(row);
        for (uint32_t column = 0; column < intermediate; ++column)
            destination[column] = scaled_silu(source[column], 1.0f, optimization_flags)
                                  * source[intermediate + column];
    }
    return linear_batch(*expert.down, activated, optimization_flags);
}

class FakeSubmission final : public ExpertSubmission
{
public:
    FakeSubmission(std::shared_ptr<FakeBackendState> state,
                   std::span<const ExpertBackendRequest> requests,
                   const std::unordered_map<std::string, ExpertMatrices>& experts,
                   uint64_t optimization_flags,
                   bool commit_success,
                   size_t aggregate_count)
        : state_(std::move(state)),
          requests_(requests.begin(), requests.end()),
          planned_(requests_.size(), ExpertBackendExecutionResult::Executed),
          final_(planned_),
          private_outputs_(requests_.size()),
          experts_(experts),
          optimization_flags_(optimization_flags),
          commit_success_(commit_success),
          aggregate_count_(aggregate_count)
    {
        for (size_t index = 0; index < requests_.size(); ++index)
        {
            const ExpertBackendRequest& request = requests_[index];
            if (!request.input || !request.output)
            {
                final_[index] = ExpertBackendExecutionResult::Failed;
                continue;
            }
            const auto expert = experts_.find(std::string(request.key));
            if (expert == experts_.end())
            {
                final_[index] = ExpertBackendExecutionResult::Failed;
                continue;
            }
            private_outputs_[index] = expert_output(expert->second, *request.input, optimization_flags_);
        }
    }

    ~FakeSubmission() override
    {
        if (!committed_ && !aborted_)
            abort();
    }

    std::span<const ExpertBackendExecutionResult> reservations() const noexcept override
    {
        return planned_;
    }

    std::vector<ExpertBackendExecutionResult> wait() override
    {
        if (!waited_)
        {
            waited_ = true;
            ++state_->wait_count;
        }
        return final_;
    }

    bool commit() override
    {
        if (!waited_)
            (void)wait();
        ++state_->commit_count;
        if (!commit_success_
            || std::any_of(final_.begin(), final_.end(), [](ExpertBackendExecutionResult result) {
                   return result != ExpertBackendExecutionResult::Executed;
               }))
        {
            return false;
        }

        for (size_t index = 0; index < aggregate_count_; ++index)
        {
            const ExpertBackendRequest& request = requests_[index];
            require(request.route_aggregation.output != nullptr, "fake aggregate output is missing");
            require(request.route_aggregation.completed != nullptr, "fake aggregate completion is missing");
            for (size_t row = 0; row < request.route_aggregation.routes.size(); ++row)
            {
                const ExpertRoute& route = request.route_aggregation.routes[row];
                float* destination = request.route_aggregation.output->row(route.token_index);
                const float* source = private_outputs_[index].row(row);
                for (uint32_t column = 0; column < request.route_aggregation.output->columns(); ++column)
                    destination[column] += route.weight * source[column];
            }
            *request.route_aggregation.completed = 1;
            ++state_->aggregate_published;
        }
        for (size_t index = aggregate_count_; index < requests_.size(); ++index)
        {
            requests_[index].output->swap(private_outputs_[index]);
            ++state_->raw_published;
        }
        committed_ = true;
        return true;
    }

    void abort() noexcept override
    {
        if (committed_ || aborted_)
            return;
        aborted_ = true;
        ++state_->abort_count;
    }

private:
    std::shared_ptr<FakeBackendState> state_;
    std::vector<ExpertBackendRequest> requests_;
    std::vector<ExpertBackendExecutionResult> planned_;
    std::vector<ExpertBackendExecutionResult> final_;
    std::vector<ActivationBuffer> private_outputs_;
    const std::unordered_map<std::string, ExpertMatrices>& experts_;
    uint64_t optimization_flags_ = 0;
    bool commit_success_ = false;
    size_t aggregate_count_ = 0;
    bool waited_ = false;
    bool committed_ = false;
    bool aborted_ = false;
};

class FakeExpertBackend final : public ExpertBackend
{
public:
    FakeExpertBackend(std::unordered_map<std::string, ExpertMatrices> experts,
                      uint64_t optimization_flags,
                      bool commit_success,
                      size_t aggregate_count,
                      std::shared_ptr<FakeBackendState> state)
        : experts_(std::move(experts)),
          optimization_flags_(optimization_flags),
          commit_success_(commit_success),
          aggregate_count_(aggregate_count),
          state_(std::move(state))
    {
    }

    void admit(std::string,
               std::shared_ptr<const TensorData>,
               const TensorData*,
               std::shared_ptr<const TensorData>,
               const TensorData*,
               uint32_t,
               float,
               ExpertActivation) override
    {
    }

    std::unique_ptr<ExpertSubmission> submit_batch(std::span<const ExpertBackendRequest> requests) override
    {
        ++state_->submit_count;
        state_->request_count = requests.size();
        for (const ExpertBackendRequest& request : requests)
        {
            state_->saw_route_aggregation = state_->saw_route_aggregation || request.route_aggregation.output != nullptr;
            state_->saw_require_all_requests = state_->saw_require_all_requests
                                               || request.route_aggregation.require_all_requests;
        }
        return std::make_unique<FakeSubmission>(state_,
                                                requests,
                                                experts_,
                                                optimization_flags_,
                                                commit_success_,
                                                aggregate_count_);
    }

    void wait_for_background_work() override
    {
    }

    ExpertBackendStatistics statistics() const override
    {
        return {};
    }

    uint64_t capacity() const noexcept override
    {
        return std::numeric_limits<uint64_t>::max();
    }

private:
    std::unordered_map<std::string, ExpertMatrices> experts_;
    uint64_t optimization_flags_ = 0;
    bool commit_success_ = false;
    size_t aggregate_count_ = 0;
    std::shared_ptr<FakeBackendState> state_;
};

static CompiledModel make_backend_model(const CompiledModel& cpu_model)
{
    CompiledModel model = cpu_model;
    model.opt.hybrid_mode = HybridMode::HybridExperts;
    size_t expert_nodes = 0;
    for (ExecutionNode& node : model.graph.nodes)
    {
        if (node.type == ExecutionNodeType::Expert || node.type == ExecutionNodeType::ExpertGroup)
        {
            node.backend = ExecutionBackend::Vulkan;
            node.backend_mask |= ExecutionBackendVulkan;
            ++expert_nodes;
        }
    }
    require(expert_nodes == 1, "batch combine test expected one Expert node");
    require(static_cast<bool>(model.schedule.validate(model.graph)), "batch combine backend schedule is invalid");

    MoeBlockPlan& moe = model.graph.layer_plans.front().moe;
    for (uint32_t expert_id = 0; expert_id < moe.experts.size(); ++expert_id)
    {
        ExpertPlan& expert = moe.experts[expert_id];
        require(expert.layout == ExpertLayout::PackedGateUpDown, "batch combine test expected packed experts");
        require(expert.gate_up_weight != invalid_tensor_handle && expert.down_weight != invalid_tensor_handle,
                "batch combine test expert weights are incomplete");
        expert.cache_key = "batch-combine-expert-" + std::to_string(expert_id);
        require(support_vulkan_expert(expert,
                                      model.weights.at(expert.gate_up_weight),
                                      model.weights.at(expert.down_weight),
                                      model.opt.optimization_flags),
                "batch combine test expert is not Vulkan eligible");
    }
    return model;
}

static void compare_logits(const std::vector<std::vector<float>>& actual,
                           const std::vector<std::vector<float>>& expected)
{
    require(actual.size() == expected.size(), "batch combine logits row count differs");
    for (size_t row = 0; row < actual.size(); ++row)
    {
        require(actual[row].size() == expected[row].size(), "batch combine logits column count differs");
        for (size_t column = 0; column < actual[row].size(); ++column)
        {
            require(std::abs(actual[row][column] - expected[row][column]) <= 3e-4f,
                    "batch combine logits differ from CPU reference");
        }
    }
}

static void run_batch_case(const CompiledModel& cpu_model,
                           const CompiledModel& backend_template,
                           bool commit_success,
                           size_t aggregate_count)
{
    CompiledModel backend_model = backend_template;
    std::unordered_map<std::string, ExpertMatrices> matrices;
    for (const ExpertPlan& expert : backend_model.graph.layer_plans.front().moe.experts)
    {
        matrices.emplace(expert.cache_key,
                         ExpertMatrices{&backend_model.weights.at(expert.gate_up_weight),
                                        &backend_model.weights.at(expert.down_weight)});
    }
    auto state = std::make_shared<FakeBackendState>();
    backend_model.expert_backend = std::make_shared<FakeExpertBackend>(matrices,
                                                                       backend_model.opt.optimization_flags,
                                                                       commit_success,
                                                                       aggregate_count,
                                                                       state);

    // Reverse first-seen order: combined slot 0 is expert 1, while Combine
    // must still visit the original expert-id order when accumulating.
    constexpr std::array<int32_t, 2> input_ids = {1, 0};
    std::vector<std::vector<float>> expected;
    expected.reserve(input_ids.size());
    for (int32_t input_id : input_ids)
    {
        SessionState session;
        SessionStatistics statistics;
        const std::array<int32_t, 1> input = {input_id};
        auto executed = forward_model(cpu_model, input, statistics, session, 0, LogitsOutput::All);
        require(static_cast<bool>(executed), "single-session CPU reference failed");
        const auto logits = batch_to_vectors(session.logits);
        require(logits.size() == 1, "single-session CPU reference has an invalid row count");
        expected.push_back(logits.front());
    }

    BatchWorkspace workspace;
    std::array<SessionState, input_ids.size()> sessions;
    std::array<SessionStatistics, input_ids.size()> statistics;
    std::array<DecodeBatchEntry, input_ids.size()> entries = {{
        {input_ids[0], &statistics[0], &sessions[0], 0},
        {input_ids[1], &statistics[1], &sessions[1], 0},
    }};
    auto actual = forward_decode_batch(backend_model, entries, workspace);
    require(static_cast<bool>(actual), "staged batch execution failed");
    compare_logits(actual.value(), expected);
    require(statistics[0].expert_batches != 0 && statistics[1].expert_batches != 0,
            "staged batch did not record expert execution statistics");

    require(state->submit_count == 1, "fake backend was not submitted exactly once");
    require(state->wait_count == 1, "fake backend was not waited exactly once");
    require(state->commit_count == 1, "fake backend was not committed exactly once");
    require(state->request_count == expert_count, "staged batch did not submit all active experts");
    require(state->saw_route_aggregation, "staged batch did not request route aggregation");
    require(state->saw_require_all_requests, "staged batch did not require complete backend coverage");
    if (commit_success)
    {
        require(state->aggregate_published == aggregate_count, "successful backend did not publish aggregation");
        require(state->raw_published == expert_count - aggregate_count, "successful backend published an invalid raw output count");
        require(state->abort_count == 0, "successful backend submission was aborted");
    }
    else
    {
        require(state->aggregate_published == 0, "failed backend published an aggregate");
        require(state->raw_published == 0, "failed backend published raw output");
        require(state->abort_count == 1, "failed backend submission was not aborted");
    }
}

void test_batch_combine_backend()
{
    const CompiledModel cpu_model = make_cpu_model();
    const CompiledModel backend_template = make_backend_model(cpu_model);
    // The executor accepts aggregate and raw outputs in one complete commit.
    run_batch_case(cpu_model, backend_template, true, 1);
    // Vulkan's require-all policy publishes every route in the aggregate.
    run_batch_case(cpu_model, backend_template, true, expert_count);
    run_batch_case(cpu_model, backend_template, false, expert_count);
}

} // namespace moe
} // namespace ncnn
