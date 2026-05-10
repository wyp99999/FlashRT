// ================================================================
// FlashRT FP8 GEMM SM89 Fix
// 添加FP8 Tensor Core支持所需的scale配置
// ================================================================

#include "gemm_runner.h"
#include <iostream>

// SM89 FP8 GEMM需要per-tensor scale mode
// 修改get_or_create_cached中的FP8_NN_DEV分支

// 原始代码在FP8_NN_DEV分支缺少scale mode设置
// 修复: 添加CUBLASLT_MATMUL_DESC_A_SCALE_MODE和B_SCALE_MODE

// 关键修复点:
// 1. FP8 GEMM需要设置scale mode为per-tensor
// 2. 需要设置scale pointer（在运行时）
// 3. cuBLASLt需要这些配置才能找到合适的算法

// 修复后的FP8_NN_DEV descriptor创建:
void fix_fp8_nn_dev_descriptor(GemmRunner::CachedGemm& entry, int M, int N, int K) {
    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    cublasOperation_t op_N = CUBLAS_OP_N;
    
    // 1. 创建matmul descriptor
    cublasLtMatmulDescCreate(&entry.matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N));
    cublasLtMatmulDescSetAttribute(entry.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N));
    
    // 2. 关键修复: 设置per-tensor scale mode
    // SM89 FP8 GEMM需要per-tensor scaling
    int32_t per_tensor_scale = CUBLASLT_MATMUL_MATRIX_SCALE_VEC1;  // per-tensor
    cublasLtMatmulDescSetAttribute(entry.matmul_desc,
        CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &per_tensor_scale, sizeof(per_tensor_scale));
    cublasLtMatmulDescSetAttribute(entry.matmul_desc,
        CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &per_tensor_scale, sizeof(per_tensor_scale));
    
    // 3. 创建matrix layout
    cublasLtMatrixLayoutCreate(&entry.A_desc, CUDA_R_8F_E4M3, M, K, K);
    cublasLtMatrixLayoutSetAttribute(entry.A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order));
    
    cublasLtMatrixLayoutCreate(&entry.B_desc, CUDA_R_8F_E4M3, K, N, N);
    cublasLtMatrixLayoutSetAttribute(entry.B_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order));
    
    cublasLtMatrixLayoutCreate(&entry.D_desc, CUDA_R_16BF, M, N, N);
    cublasLtMatrixLayoutSetAttribute(entry.D_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order));
}

// 注意: 如果cuBLASLt仍然找不到算法，可能是因为:
// 1. SM89的FP8 GEMM需要特定的矩阵尺寸alignment (如16或32)
// 2. 需要使用cublasLtMatmulAlgoSearch而不是AlgoGetHeuristic
// 3. SM89可能不支持某些FP8 GEMM配置

// 替代方案: 使用FP8量化 + FP16 GEMM
// 这在测试中显示比纯FP16更快（内存带宽优势）