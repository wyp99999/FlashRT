// ================================================================
// FlashRT — FP8 Storage + FP16 Compute GEMM for SM89 (Ada)
// 
// Strategy: FP8存储 + FP16 Tensor Core计算
// - 权重以FP8格式存储 (节省50%显存和带宽)
// - 运行时反量化为FP16
// - 使用SM89支持的FP16 Tensor Core (16x8x16指令形状)
// - 输出FP16/BF16结果
// 
// This kernel works on SM89 where FP8 Tensor Core MMA is NOT supported.
// ================================================================

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/device_memory.h>
#include <cstdio>
#include <mutex>

// ── Kernel Configuration for SM89 FP16 Tensor Core ──
// SM89 supports FP16 Tensor Core with 16x8x16 instruction shape
using ElementA_FP8 = cutlass::float_e4m3_t;  // FP8 input (stored)
using ElementB_FP8 = cutlass::float_e4m3_t;  // FP8 input (stored)
using ElementA_FP16 = cutlass::half_t;       // FP16 for computation
using ElementB_FP16 = cutlass::half_t;       // FP16 for computation
using ElementOutput = cutlass::bfloat16_t;   // BF16 output
using ElementAccumulator = float;            // FP32 accumulator
using ElementCompute = float;                // FP32 compute

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;  // Weight matrix typically column-major
using LayoutC = cutlass::layout::RowMajor;

static int const kStages = 3;
static int const kAlignmentA = 16;
static int const kAlignmentB = 16;

// ── FP16 Tensor Core GEMM (SM89 supported) ──
using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // Vector width
    ElementAccumulator,
    ElementCompute,
    cutlass::epilogue::thread::ScaleType::Default
>;

using GemmFP16TC = cutlass::gemm::device::GemmUniversal<
    ElementA_FP16, LayoutA,
    ElementB_FP16, LayoutB,
    ElementOutput, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    cutlass::gemm::GemmShape<128, 128, 64>,    // Threadblock shape
    cutlass::gemm::GemmShape<64, 64, 64>,      // Warp shape  
    cutlass::gemm::GemmShape<16, 8, 16>,       // FP16 TC instruction shape (SM89 supports)
    EpilogueOutputOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages,
    kAlignmentA,
    kAlignmentB
>;

// ── FP8 to FP16 conversion kernel (optimized) ──
__global__ void fp8_to_fp16_kernel(
    const __nv_fp8_e4m3* __restrict__ input,
    __half* __restrict__ output,
    float scale,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = static_cast<float>(input[idx]);
        output[idx] = __float2half(val * scale);
    }
}

// ── Batched FP8 to FP16 conversion (for GEMM inputs) ──
__global__ void fp8_to_fp16_batched_kernel(
    const __nv_fp8_e4m3* __restrict__ input,
    __half* __restrict__ output,
    const float* __restrict__ scale,  // Per-tensor scale
    int rows,
    int cols,
    int stride  // Row stride (for row-major)
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * stride + col;
        float val = static_cast<float>(input[idx]);
        float s = scale ? *scale : 1.0f;
        output[idx] = __float2half(val * s);
    }
}

// ── Workspace management ──
static std::mutex g_fp8_fp16_mutex;
static cutlass::device_memory::allocation<uint8_t> g_fp8_fp16_workspace(0);
static cutlass::device_memory::allocation<__half> g_fp8_fp16_buffer_A(0);
static cutlass::device_memory::allocation<__half> g_fp8_fp16_buffer_B(0);

// ── Main GEMM function: FP8 input -> FP16 compute -> BF16 output ──
extern "C" {

int fp8_storage_fp16_compute_gemm(
    const void* A_fp8,      // FP8 input matrix A (M x K)
    const void* B_fp8,      // FP8 input matrix B (K x N)
    void* D_out,            // BF16 output matrix (M x N)
    int M, int N, int K,
    const float* scale_a,   // Per-tensor scale for A
    const float* scale_b,   // Per-tensor scale for B
    cudaStream_t stream
) {
    // Step 1: Allocate temporary FP16 buffers
    size_t size_A = M * K * sizeof(__half);
    size_t size_B = K * N * sizeof(__half);
    
    {
        std::lock_guard<std::mutex> lock(g_fp8_fp16_mutex);
        
        if (size_A > g_fp8_fp16_buffer_A.size()) {
            g_fp8_fp16_buffer_A = cutlass::device_memory::allocation<__half>(size_A);
        }
        if (size_B > g_fp8_fp16_buffer_B.size()) {
            g_fp8_fp16_buffer_B = cutlass::device_memory::allocation<__half>(size_B);
        }
    }
    
    __half* A_fp16 = g_fp8_fp16_buffer_A.get();
    __half* B_fp16 = g_fp8_fp16_buffer_B.get();
    
    // Step 2: Convert FP8 to FP16 (with scaling)
    // A matrix conversion (M x K)
    int block_size = 256;
    int grid_x_A = (K + block_size - 1) / block_size;
    dim3 grid_A(grid_x_A, M);
    fp8_to_fp16_batched_kernel<<<grid_A, block_size, 0, stream>>>(
        (const __nv_fp8_e4m3*)A_fp8, A_fp16,
        scale_a, M, K, K
    );
    
    // B matrix conversion (K x N) - column-major for GEMM
    int grid_x_B = (N + block_size - 1) / block_size;
    dim3 grid_B(grid_x_B, K);
    fp8_to_fp16_batched_kernel<<<grid_B, block_size, 0, stream>>>(
        (const __nv_fp8_e4m3*)B_fp8, B_fp16,
        scale_b, K, N, N
    );
    
    // Step 3: Run FP16 Tensor Core GEMM
    float alpha = 1.0f;
    float beta = 0.0f;
    
    typename GemmFP16TC::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        1,  // batch_count
        {alpha, beta},
        A_fp16,
        B_fp16,
        (ElementOutput*)nullptr,  // C (source, not used with beta=0)
        (ElementOutput*)D_out,
        K,    // lda (row-major)
        N,    // ldb (column-major for B)
        N,    // ldc
        N,    // ldd
        0,    // batch_stride_A
        0,    // batch_stride_B
        0,    // batch_stride_C
        0     // batch_stride_D
    };
    
    GemmFP16TC gemm_op;
    
    // Check if implementable
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[FP8-FP16] cannot implement: M=%d N=%d K=%d\n", M, N, K);
        return -1;
    }
    
    // Get workspace size
    size_t ws_size = GemmFP16TC::get_workspace_size(arguments);
    
    {
        std::lock_guard<std::mutex> lock(g_fp8_fp16_mutex);
        if (ws_size > g_fp8_fp16_workspace.size()) {
            g_fp8_fp16_workspace = cutlass::device_memory::allocation<uint8_t>(ws_size);
        }
    }
    
    // Initialize and run
    status = gemm_op.initialize(arguments, g_fp8_fp16_workspace.get(), stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[FP8-FP16] init failed: M=%d N=%d K=%d\n", M, N, K);
        return -2;
    }
    
    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "[FP8-FP16] run failed: M=%d N=%d K=%d\n", M, N, K);
        return -3;
    }
    
    return 0;
}

// ── Alternative: In-place FP8 to FP16 GEMM with fused conversion ──
// This uses CUTLASS's built-in numeric conversion
__global__ void fp8_to_fp16_fused_kernel(
    const __nv_fp8_e4m3* __restrict__ input,
    __half* __restrict__ output,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Direct conversion without scaling (scale applied in epilogue)
        output[idx] = __half(static_cast<float>(input[idx]));
    }
}

// ── Get workspace size for pre-allocation ──
size_t fp8_storage_fp16_compute_gemm_workspace_size(int M, int N, int K) {
    size_t buffer_A = M * K * sizeof(__half);
    size_t buffer_B = K * N * sizeof(__half);
    
    // CUTLASS workspace
    typename GemmFP16TC::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        1,
        {1.0f, 0.0f},
        nullptr, nullptr, nullptr, nullptr,
        K, N, N, N, 0, 0, 0, 0
    };
    size_t ws = GemmFP16TC::get_workspace_size(arguments);
    
    return buffer_A + buffer_B + ws;
}

}  // extern "C"