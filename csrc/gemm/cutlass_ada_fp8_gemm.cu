// ================================================================
// FlashRT — CUTLASS FP8 GEMM implementations for SM89 (Ada)
// 
// Based on CUTLASS example 58_ada_fp8_gemm
// FP8 E4M3 inputs → BF16/FP16 output
// Per-tensor scaling with device scale pointers (CUDA Graph compatible)
// ================================================================

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/gemm/device/gemm_universal_with_absmax.h>
#include <cutlass/epilogue/thread/linear_combination_generic_with_scaling.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/device_memory.h>
#include <cstdio>
#include <mutex>

// ── Ada FP8 GEMM Types ──
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementAccumulator = float;
using ElementCompute = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;  // FP8 MMA requires B in ColumnMajor
using LayoutC = cutlass::layout::RowMajor;

static int const kStages = 3;
static int const kAlignmentA = 16;
static int const kAlignmentB = 16;

// ── BF16 Output GEMM (standard for Pi0.5) ──
using ElementOutputBF16 = cutlass::bfloat16_t;
using ElementAuxOutputBF16 = ElementOutputBF16;

using EpilogueOutputOpBF16 = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity,  // No activation
    ElementOutputBF16,
    ElementAuxOutputBF16,
    8,  // vector width
    ElementAccumulator,
    ElementAccumulator
>;

template <typename MathOperator>
using GemmAdaBF16_ = cutlass::gemm::device::GemmUniversalWithAbsMax<
    ElementA, LayoutA, ElementB, LayoutB, ElementOutputBF16, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 64, 128>,  // Threadblock shape
    cutlass::gemm::GemmShape<64, 32, 128>,   // Warp shape
    cutlass::gemm::GemmShape<16, 8, 32>,     // Instruction shape (FP8 TC)
    EpilogueOutputOpBF16,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages,
    kAlignmentA, kAlignmentB,
    MathOperator
>;

using GemmAdaBF16 = GemmAdaBF16_<cutlass::arch::OpMultiplyAdd>;

// ── FP16 Output GEMM ──
using ElementOutputFP16 = cutlass::half_t;
using ElementAuxOutputFP16 = ElementOutputFP16;

using EpilogueOutputOpFP16 = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    cutlass::epilogue::thread::Identity,
    ElementOutputFP16,
    ElementAuxOutputFP16,
    8,
    ElementAccumulator,
    ElementAccumulator
>;

template <typename MathOperator>
using GemmAdaFP16_ = cutlass::gemm::device::GemmUniversalWithAbsMax<
    ElementA, LayoutA, ElementB, LayoutB, ElementOutputFP16, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 64, 128>,
    cutlass::gemm::GemmShape<64, 32, 128>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    EpilogueOutputOpFP16,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages,
    kAlignmentA, kAlignmentB,
    MathOperator
>;

using GemmAdaFP16 = GemmAdaFP16_<cutlass::arch::OpMultiplyAdd>;

// ── Workspace cache (thread-safe) ──
static std::mutex g_ada_mutex;
static cutlass::device_memory::allocation<uint8_t> g_ada_workspace(0);

// ── Generic runner with device scale pointers ──
// Follows the exact format from CUTLASS example 58_ada_fp8_gemm
template <typename GemmOp>
static int cutlass_ada_run_impl(
    const void* A_fp8, const void* B_fp8, void* D_out,
    int M, int N, int K,
    const float* d_scale_a, const float* d_scale_b,
    cudaStream_t stream) {
    
    using ElementA = typename GemmOp::ElementA;
    using ElementB = typename GemmOp::ElementB;
    using ElementC = typename GemmOp::ElementC;  // ElementC is the output type
    
    // Create dummy pointers for optional params (C, Aux, Vector)
    static void* dummy_ptr = nullptr;
    
    // Epilogue parameters following CUTLASS Ada FP8 format
    typename GemmOp::EpilogueOutputOp::Params::ActivationParams activation_params{
        ElementCompute(1.0f),  // alpha
        ElementCompute(0.0f)   // beta
    };
    
    typename GemmOp::EpilogueOutputOp::Params epilogue_params{
        activation_params,
        d_scale_a,       // scale_A (activation scale)
        d_scale_b,       // scale_B (weight scale)
        nullptr,         // scale_C
        nullptr,         // scale_D
        nullptr,         // scale_Aux
        nullptr,         // abs_max_Aux
        nullptr          // abs_max_D
    };
    
    // Arguments following CUTLASS Ada FP8 format
    typename GemmOp::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},      // problem_size
        1,              // batch_count
        epilogue_params,
        (ElementA*)A_fp8,  // tensor_A
        (ElementB*)B_fp8,  // tensor_B
        (ElementC*)dummy_ptr,  // tensor_C (source, not used with beta=0)
        (ElementC*)D_out,   // tensor_D (output) - ElementC is output type
        nullptr,           // tensor_Aux
        nullptr,           // tensor_Vector (bias)
        M * K,             // batch_stride_A
        K * N,             // batch_stride_B (ColumnMajor layout)
        M * N,             // batch_stride_C
        M * N,             // batch_stride_D
        (int)M,            // batch_stride_vector
        K,                 // stride_A (RowMajor: ld = K)
        K,                 // stride_B (ColumnMajor: ld = K)
        N,                 // stride_C
        N,                 // stride_D
        (int64_t)0         // ld_vector
    };
    
    GemmOp gemm_op;
    
    // Check if implementable
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS Ada] cannot implement: M=%d N=%d K=%d\n", M, N, K);
        return -1;
    }
    
    // Get workspace size
    size_t ws_size = GemmOp::get_workspace_size(arguments);
    
    // Allocate/reallocate workspace if needed (thread-safe)
    {
        std::lock_guard<std::mutex> lock(g_ada_mutex);
        if (ws_size > g_ada_workspace.size()) {
            g_ada_workspace = cutlass::device_memory::allocation<uint8_t>(ws_size);
        }
    }
    
    // Initialize
    status = gemm_op.initialize(arguments, g_ada_workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS Ada] init failed: M=%d N=%d K=%d\n", M, N, K);
        return -2;
    }
    
    // Run
    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS Ada] run failed: M=%d N=%d K=%d\n", M, N, K);
        return -3;
    }
    
    return 0;
}

// ── Exported C functions ──
extern "C" {

// BF16 output (primary for Pi0.5 decoder)
int cutlass_ada_fp8_gemm_bf16(
    const void* A_fp8, const void* B_fp8, void* D_bf16,
    int M, int N, int K,
    const float* d_scale_a, const float* d_scale_b,
    cudaStream_t stream) {
    return cutlass_ada_run_impl<GemmAdaBF16>(
        A_fp8, B_fp8, D_bf16, M, N, K, d_scale_a, d_scale_b, stream);
}

// FP16 output (for vision encoder, alternative)
int cutlass_ada_fp8_gemm_fp16(
    const void* A_fp8, const void* B_fp8, void* D_fp16,
    int M, int N, int K,
    const float* d_scale_a, const float* d_scale_b,
    cudaStream_t stream) {
    return cutlass_ada_run_impl<GemmAdaFP16>(
        A_fp8, B_fp8, D_fp16, M, N, K, d_scale_a, d_scale_b, stream);
}

}  // extern "C"