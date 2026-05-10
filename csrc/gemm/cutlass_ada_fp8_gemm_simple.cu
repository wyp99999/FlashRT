// ================================================================
// FlashRT — CUTLASS FP8 GEMM implementations for SM89 (Ada)
// Simple LinearCombination version (no AbsMax, for BF16/FP16 output)
// ================================================================

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/epilogue/thread/linear_combination.h>
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

// ── BF16 Output GEMM (simple linear combination) ──
using ElementOutputBF16 = cutlass::bfloat16_t;

using EpilogueOutputOpBF16 = cutlass::epilogue::thread::LinearCombination<
    ElementOutputBF16,      // ElementOutput
    8,                      // Count (vector width)
    ElementAccumulator,     // ElementAccumulator
    ElementCompute,         // ElementCompute
    cutlass::epilogue::thread::ScaleType::Default  // Scale
>;

template <typename MathOperator>
using GemmAdaBF16_Simple_ = cutlass::gemm::device::GemmUniversal<
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

using GemmAdaBF16_Simple = GemmAdaBF16_Simple_<cutlass::arch::OpMultiplyAdd>;

// ── FP16 Output GEMM ──
using ElementOutputFP16 = cutlass::half_t;

using EpilogueOutputOpFP16 = cutlass::epilogue::thread::LinearCombination<
    ElementOutputFP16,      // ElementOutput
    8,                      // Count
    ElementAccumulator,     // ElementAccumulator
    ElementCompute,         // ElementCompute
    cutlass::epilogue::thread::ScaleType::Default  // Scale
>;

template <typename MathOperator>
using GemmAdaFP16_Simple_ = cutlass::gemm::device::GemmUniversal<
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

using GemmAdaFP16_Simple = GemmAdaFP16_Simple_<cutlass::arch::OpMultiplyAdd>;

// ── Workspace cache (thread-safe) ──
static std::mutex g_ada_simple_mutex;
static cutlass::device_memory::allocation<uint8_t> g_ada_simple_workspace(0);

// ── Generic runner with simple linear combination ──
template <typename GemmOp>
static int cutlass_ada_simple_run_impl(
    const void* A_fp8, const void* B_fp8, void* D_out,
    int M, int N, int K,
    float scale_a, float scale_b,
    cudaStream_t stream) {
    
    using ElementC = typename GemmOp::ElementC;  // ElementC is the output type
    
    // Epilogue parameters for LinearCombination
    // alpha * accumulator + beta * source
    // Since beta=0: alpha * (A @ B)
    // alpha = scale_a * scale_b (combined scaling factor)
    typename GemmOp::EpilogueOutputOp::Params epilogue_params{
        scale_a * scale_b,  // alpha
        0.0f                // beta
    };
    
    // Arguments for GemmUniversal
    typename GemmOp::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},      // problem_size
        1,              // batch_count
        epilogue_params,
        (ElementA*)A_fp8,  // tensor_A
        (ElementB*)B_fp8,  // tensor_B
        (ElementC*)nullptr,  // tensor_C (source, not used with beta=0)
        (ElementC*)D_out,  // tensor_D (output)
        M * K,             // batch_stride_A
        K * N,             // batch_stride_B (ColumnMajor: K*N)
        M * N,             // batch_stride_C
        M * N,             // batch_stride_D
        K,                 // stride_A (row-major: ld = K)
        K,                 // stride_B (column-major: ld = K)
        N,                 // stride_C
        N                  // stride_D
    };
    
    GemmOp gemm_op;
    
    // Check if implementable
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS Ada Simple] cannot implement: M=%d N=%d K=%d\n", M, N, K);
        return -1;
    }
    
    // Get workspace size
    size_t ws_size = GemmOp::get_workspace_size(arguments);
    
    // Allocate/reallocate workspace if needed (thread-safe)
    {
        std::lock_guard<std::mutex> lock(g_ada_simple_mutex);
        if (ws_size > g_ada_simple_workspace.size()) {
            g_ada_simple_workspace = cutlass::device_memory::allocation<uint8_t>(ws_size);
        }
    }
    
    // Initialize
    status = gemm_op.initialize(arguments, g_ada_simple_workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS Ada Simple] init failed: M=%d N=%d K=%d\n", M, N, K);
        return -2;
    }
    
    // Run
    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[CUTLASS Ada Simple] run failed: M=%d N=%d K=%d\n", M, N, K);
        return -3;
    }
    
    return 0;
}

// ── Exported C functions ──
extern "C" {

// BF16 output (primary for Pi0.5 decoder) - simple version
int cutlass_ada_fp8_gemm_bf16_simple(
    const void* A_fp8, const void* B_fp8, void* D_bf16,
    int M, int N, int K,
    float scale_a, float scale_b,
    cudaStream_t stream) {
    return cutlass_ada_simple_run_impl<GemmAdaBF16_Simple>(
        A_fp8, B_fp8, D_bf16, M, N, K, scale_a, scale_b, stream);
}

// FP16 output - simple version
int cutlass_ada_fp8_gemm_fp16_simple(
    const void* A_fp8, const void* B_fp8, void* D_fp16,
    int M, int N, int K,
    float scale_a, float scale_b,
    cudaStream_t stream) {
    return cutlass_ada_simple_run_impl<GemmAdaFP16_Simple>(
        A_fp8, B_fp8, D_fp16, M, N, K, scale_a, scale_b, stream);
}

}  // extern "C"