#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                          \
    cudaError_t err = (call);                          \
    if (err != cudaSuccess) {                          \
        fprintf(stderr, "CUDA error %s:%d: %s\n",      \
                __FILE__, __LINE__,                    \
                cudaGetErrorString(err));              \
        exit(1);                                       \
    }                                                  \
} while (0)

static inline int ceil_div(int a, int b) {
    return (a+b-1)/b;
}

double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

#define BR 32
#define BC 32
// #define BR 32
// #define BC 64
#define MAX_D 64

int Batch, N, d;
float *Q, *K, *V, *O;

void input(const char *input_filename) {
    FILE *file = fopen(input_filename, "rb");
    if (!file) {
        perror("fopen input");
        exit(1);
    }
    fread(&Batch, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc((size_t)Batch * N * d * sizeof(float));
    K = (float *)malloc((size_t)Batch * N * d * sizeof(float));
    V = (float *)malloc((size_t)Batch * N * d * sizeof(float));
    O = (float *)malloc((size_t)Batch * N * d * sizeof(float));
    if (!Q || !K || !V || !O) {
        fprintf(stderr, "Host malloc failed\n");
        exit(1);
    }
    for (int i = 0; i < Batch; i++) {
        fread(Q + (size_t)i * N * d, sizeof(float), (size_t)N * d, file);
        fread(K + (size_t)i * N * d, sizeof(float), (size_t)N * d, file);
        fread(V + (size_t)i * N * d, sizeof(float), (size_t)N * d, file);
    }
    memset(O, 0x00, (size_t)Batch * N * d * sizeof(float));
    fclose(file);
}

void output(const char *output_filename) {
    FILE *file = fopen(output_filename, "wb");
    if (!file) {
        perror("fopen output");
        exit(1);
    }
    fwrite(O, sizeof(float), (size_t)Batch * N * d, file);
    free(Q);
    free(K);
    free(V);
    free(O);
    fclose(file);
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}


template<int D>
__global__ void flash_attention_forward_kernel_t(
    const float * __restrict__ Q,
    const float * __restrict__ K,
    const float * __restrict__ V,
    float * __restrict__ O,
    int N,
    int Batch
) {
    const int lane = threadIdx.x;
    const int r_in_block = threadIdx.y;

    const int b = blockIdx.y;  // batch index
    const int row_in_batch = blockIdx.x * BR + r_in_block;
    if (row_in_batch >= N || b >= Batch) return;

    const int row = b*N + row_in_batch;

    __shared__ float sQ[BR * D];
    __shared__ float sK[BC * D];
    __shared__ float sV[BC * D];

    float *sQ_row = sQ + r_in_block*D;
    const int vec_d = D/4;
    const float4 *gQ4 = reinterpret_cast<const float4*>(Q + (size_t)row * D);
    float4 *sQ4 = reinterpret_cast<float4*>(sQ_row);

    for (int vidx=lane; vidx<vec_d; vidx+=32) {
        sQ4[vidx] = gQ4[vidx];
    }
    __syncthreads();

    float mi = -1e30f;
    float li = 0.0f;
    float acc[(D + 31) / 32];
    #pragma unroll
    for (int u=0; u<(D+31)/32; ++u) acc[u]=0.0f;
    const float scale = 1.0f/sqrtf((float)D);

    const int num_k_blocks = (N + BC - 1) / BC;

    for (int kb=0; kb<num_k_blocks; ++kb) {
        const int k_start = kb * BC;
        int cols = N - k_start;
        if (cols > BC) cols = BC;

        // load K, V tile for this batch
        for (int kk=r_in_block; kk<cols; kk+=BR) {
            const int key_row = b*N + (k_start + kk);
            const float *gK_row = K + (size_t)key_row * D;
            const float *gV_row = V + (size_t)key_row * D;
            float *sK_row = sK + kk*D;
            float *sV_row = sV + kk*D;

            const float4 *gK4 = reinterpret_cast<const float4*>(gK_row);
            const float4 *gV4 = reinterpret_cast<const float4*>(gV_row);
            float4 *sK4 = reinterpret_cast<float4*>(sK_row);
            float4 *sV4 = reinterpret_cast<float4*>(sV_row);
            for (int vidx=lane; vidx<vec_d; vidx+=32) {
                sK4[vidx] = gK4[vidx];
                sV4[vidx] = gV4[vidx];
            }
        }
        __syncthreads();

        for (int kk=0; kk<cols; ++kk) {
            float *k_vec = sK + kk*D;
            float *v_vec = sV + kk*D;

            float partial = 0.0f;
            #pragma unroll
            for (int t=lane; t<D; t+=32) {
                partial += sQ_row[t] * k_vec[t];
            }
            float dot = warp_reduce_sum(partial);

            float score = 0.0f;
            if (lane == 0) score = dot * scale;
            score = __shfl_sync(0xffffffffu, score, 0);

            float m_new = fmaxf(mi, score);
            float alpha = __expf(mi - m_new);
            float beta  = __expf(score - m_new);
            float l_new = alpha*li + beta;

            float coeff_prev = (li>0.0f) ? (alpha*li/l_new) : 0.0f;
            float coeff_new  = beta / l_new;
            #pragma unroll
            for (int u=0; u<(D+31)/32; ++u) {
                const int t = lane + u*32;
                if (t < D) {
                    float v_val = v_vec[t];
                    acc[u] = coeff_prev*acc[u] + coeff_new*v_val;
                }
            }
            mi = m_new;
            li = l_new;
        }
        __syncthreads();
    }
    // write O
    #pragma unroll
    for (int u=0; u<(D+31)/32; ++u) {
        const int t = lane + u*32;
        if (t < D) {
            O[(size_t)row*D+t] = acc[u];
        }
    }
}
int main(int argc, char *argv[]) {
    double t_io_start = getTimeStamp();
    input(argv[1]);
    double t_io_end = getTimeStamp();

    float *dQ, *dK, *dV, *dO;
    size_t total_elems = (size_t)Batch*N*d;
    size_t total_bytes = total_elems*sizeof(float);

    CHECK_CUDA(cudaMalloc(&dQ, total_bytes));
    CHECK_CUDA(cudaMalloc(&dK, total_bytes));
    CHECK_CUDA(cudaMalloc(&dV, total_bytes));
    CHECK_CUDA(cudaMalloc(&dO, total_bytes));

    dim3 blockDim(32, BR);
    dim3 gridDim(ceil_div(N, BR), Batch);

    double t_h2d_start = getTimeStamp();
    CHECK_CUDA(cudaMemcpy(dQ, Q, total_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, K, total_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, V, total_bytes, cudaMemcpyHostToDevice));
    double t_h2d_end = getTimeStamp();

    double t_kernel_start = getTimeStamp();
    if (d == 32) {
        flash_attention_forward_kernel_t<32><<<gridDim, blockDim>>>(dQ, dK, dV, dO, N, Batch);
    } else if (d == 64) {
        flash_attention_forward_kernel_t<64><<<gridDim, blockDim>>>(dQ, dK, dV, dO, N, Batch);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    double t_kernel_end = getTimeStamp();

    double t_d2h_start = getTimeStamp();
    CHECK_CUDA(cudaMemcpy(O, dO, total_bytes, cudaMemcpyDeviceToHost));
    double t_d2h_end = getTimeStamp();

    printf("\n=== Time Distribution ===\n");
    printf("I/O (read input):        %.3f s\n", t_io_end - t_io_start);
    printf("H2D transfer:            %.3f s\n", t_h2d_end - t_h2d_start);
    printf("Kernel compute:          %.3f s\n", t_kernel_end - t_kernel_start);
    printf("D2H transfer:            %.3f s\n", t_d2h_end - t_d2h_start);
    printf("Total:                   %.3f s\n",
           (t_io_end - t_io_start) +
           (t_h2d_end - t_h2d_start) +
           (t_kernel_end - t_kernel_start) +
           (t_d2h_end - t_d2h_start));

    output(argv[2]);

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
    return 0;
}