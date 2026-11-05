#include "test_runtime_decls.h"
#include "test_runtime_support.h"

int main(int argc, char** argv)
{
    try
    {
        if (argc > 2)
        {
            std::cerr << "expected at most one test argument\n";
            return 2;
        }
        const std::string argument = argc > 1 ? std::string(argv[1]) : std::string{};
        if (argument == "--mapped-file")
        {
#if !defined(_WIN32)
            struct rlimit limits{};
            if (getrlimit(RLIMIT_NOFILE, &limits) != 0 || limits.rlim_max < 16)
            {
                std::cerr << "cannot set mapped-file test descriptor limit\n";
                return 3;
            }
            limits.rlim_cur = 16;
            if (setrlimit(RLIMIT_NOFILE, &limits) != 0)
            {
                std::cerr << "cannot set mapped-file test descriptor limit\n";
                return 3;
            }
            ncnn::moe::test_mapped_file_range_and_shared_buffer(true);
            return 0;
#else
            std::cerr << "mapped-file descriptor test requires POSIX\n";
            return 3;
#endif
        }
        if (argument == "--benchmark-bfloat16-attention")
        {
            ncnn::moe::benchmark_bfloat16_attention_kernels();
            return 0;
        }
        if (argument == "--benchmark-qnk")
        {
            ncnn::moe::benchmark_qnk_gemm();
            return 0;
        }
        if (argument == "--benchmark-vulkan-qnk")
        {
            ncnn::moe::benchmark_vulkan_qnk();
            return 0;
        }
        if (argument == "--benchmark-vulkan-qnk-expert")
        {
            ncnn::moe::benchmark_vulkan_qnk_expert();
            return 0;
        }
        if (!argument.empty())
        {
            std::cerr << "unknown test argument: " << argument << '\n';
            return 2;
        }
        ncnn::moe::test_flag_defaults();
        ncnn::moe::test_cpu_task_worker();
        ncnn::moe::test_cpu_resource_coordination();
        ncnn::moe::test_float_scaled_add();
        ncnn::moe::test_float_scale_inplace_and_scaled_add();
        ncnn::moe::test_float_scale_add();
        ncnn::moe::test_float_dot();
        ncnn::moe::test_dense_gemm_tails();
        ncnn::moe::test_float_exp_inplace();
        ncnn::moe::test_float_approximate_exp();
        ncnn::moe::test_int8_float_dot();
        ncnn::moe::test_weighted_scale();
        ncnn::moe::test_rms_vector_kernels();
        ncnn::moe::test_l2_vector_kernel();
        ncnn::moe::test_hyper_connection_vector_kernels();
        ncnn::moe::test_float_rope_kernel();
        ncnn::moe::test_float_silu_mul();
        ncnn::moe::test_float_sigmoid_mul();
        ncnn::moe::test_dense_expert_thread_budget();
        ncnn::moe::test_bfloat16_vector_kernels();
        ncnn::moe::test_bfloat16_dot_reduction();
        ncnn::moe::test_float_to_bfloat16_array();
        ncnn::moe::test_activation_buffer_reset();
        ncnn::moe::test_bfloat16_batched_linear_kernel();
        ncnn::moe::test_linear_operator();
        ncnn::moe::test_vulkan_device_tensor_residency();
        ncnn::moe::test_dense_mxn_tiles();
        ncnn::moe::test_released_dense_host_storage_guard();
        ncnn::moe::test_ncnn_vulkan_bfloat16_operator();
        ncnn::moe::test_ncnn_vulkan_float8_operator();
        ncnn::moe::test_gguf_mxfp4_row_unpack();
        ncnn::moe::test_mxfp4_weight_storage_padding();
        ncnn::moe::test_weightstore_finalize();
        ncnn::moe::test_mxfp4_cpu_kernel_and_fused_gate_up();
        ncnn::moe::test_qnk_cpu_kernel();
        ncnn::moe::test_qnk_graph_cpu_projections();
        ncnn::moe::test_ncnn_vulkan_qnk_operator();
        ncnn::moe::test_ncnn_vulkan_qnk_expert_operator();
        ncnn::moe::test_adaptive_cache_wait_policy();
        ncnn::moe::test_sharded_expert_victim_cache();
        ncnn::moe::test_mapped_file_range_and_shared_buffer();
        ncnn::moe::test_safetensors_dense_mmap();
        ncnn::moe::test_safetensors_packed_mxfp4_expert();
        ncnn::moe::test_safetensors_qnk_source_binding();
        ncnn::moe::test_expert_residency_coordinator();
        ncnn::moe::test_file_backed_bfloat16_expert_cache();
        ncnn::moe::test_file_backed_qnk_expert_cache();
        ncnn::moe::test_file_backed_mxfp4_expert_cache();
        ncnn::moe::test_cross_session_batch_scheduler();
        ncnn::moe::test_batch_scheduler_submission_contract();
        ncnn::moe::test_independent_batch_scheduler();
        ncnn::moe::test_staged_bfloat16_dispatch_telemetry();
        ncnn::moe::test_chunked_prefill_statistics_commit_on_success();
        ncnn::moe::test_chunked_prefill_invalid_token_preserves_attention_state();
        ncnn::moe::test_staged_invalid_token_preserves_attention_state();
        ncnn::moe::test_session_execution_failure_requires_reset();
        ncnn::moe::test_staged_execution_failure_requires_reset();
        ncnn::moe::test_speculative_generation();
        ncnn::moe::test_prefill_decode_and_reset();
        ncnn::moe::test_invalid_token_is_transactional();
        ncnn::moe::test_chunked_prefill_matches_single_batch();
        ncnn::moe::test_topk_selected_weight_normalization_and_combine();
        ncnn::moe::test_int8_expert_linear();
        ncnn::moe::test_invalid_int8_scale_is_rejected();
        ncnn::moe::test_attention_kv_cache_and_reset();
        ncnn::moe::test_bfloat16_ring_kv_cache();
        ncnn::moe::test_generic_mtp_int4_cache_key();
        ncnn::moe::test_attention_graph_without_bias_or_sink();
        ncnn::moe::test_shared_expert_descriptor();
        ncnn::moe::test_execution_graph_and_scheduler();
        ncnn::moe::test_expert_dispatcher_groups_routes();
        ncnn::moe::test_deepseek_router_and_hyper_connection_kernels();
        ncnn::moe::test_deepseek_v4_descriptors();
        ncnn::moe::test_latent_attention_scratch_reuse();
        ncnn::moe::test_qwen3_5_moe_descriptors();
        ncnn::moe::test_model_neutral_expert_storage();
        ncnn::moe::test_qwen4_exp_descriptors();
        ncnn::moe::test_qwen4_exp_compile_and_execute();
        ncnn::moe::test_qwen4_exp_vulkan_projections();
        ncnn::moe::test_gated_delta_vulkan_budget_admission();
        ncnn::moe::test_gated_residual_kernels();
        ncnn::moe::test_ple_prefill_decode_continuation();
        ncnn::moe::test_quantized_attention_scratch_reuse();
        ncnn::moe::test_attention_output_aliases();
        ncnn::moe::test_qsa_prefill_decode_continuation();
        ncnn::moe::test_gated_delta_net_continuation();
        ncnn::moe::test_automatic_expert_memory_planning();
        ncnn::moe::test_sampling_and_streaming_generation();
        ncnn::moe::test_batch_workspace_reuse();
        ncnn::moe::test_batch_combine_backend();
        ncnn::moe::test_dense_ffn();
        ncnn::moe::test_int4();
        ncnn::moe::test_quant_archive();
        ncnn::moe::test_native_awq();
        ncnn::moe::test_generation_logits_ownership();
        ncnn::moe::test_generation_callback_errors();
        ncnn::moe::test_manifest_readers();
        ncnn::moe::test_model_adapter_scopes();
        ncnn::moe::test_loader_reports_adapter_and_weight_errors();
        ncnn::moe::test_backend_capabilities_and_hybrid_execution();
        std::cout << "all ncnn_moe tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
