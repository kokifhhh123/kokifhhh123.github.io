#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define BSZ 64
const int INF = (1<<30)-1;

int original_n, m, n;

inline int ceil_div(int a, int b) {return (a+b-1)/b;}

// Phase 1
__global__ void phase1_kernel(int* __restrict__ d, const int N, const int Round) {
    __shared__ int s[BSZ][BSZ];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int half = BSZ >> 1;
    int base = Round * BSZ;

    int i0 = base + ty;
    int j0 = base + tx;

    s[ty][tx]               = d[i0*N + j0];
    s[ty][tx+half]        = d[i0*N + (j0 + half)];
    s[ty + half][tx]        = d[(i0+half)*N + j0];
    s[ty + half][tx + half] = d[(i0+half)*N + (j0 + half)];
    __syncthreads();

    // #pragma unroll 8
for (int k=0; k<BSZ; k++) {
    s[ty][tx]               = min(s[ty][tx],               s[ty][k] + s[k][tx]);
    s[ty][tx + half]        = min(s[ty][tx + half],        s[ty][k] + s[k][tx + half]);
    s[ty + half][tx]        = min(s[ty + half][tx],        s[ty + half][k] + s[k][tx]);
    s[ty + half][tx + half] = min(s[ty + half][tx + half], s[ty + half][k] + s[k][tx + half]);
    __syncthreads();
}
    d[i0*N + j0]         = s[ty][tx];
    d[i0*N + (j0+half)]  = s[ty][tx+half];
    d[(i0+half)*N + j0]  = s[ty+half][tx];
    d[(i0+half)*N + (j0+half)] = s[ty+half][tx + half];
}

// Phase 2
__global__ void phase2_row_kernel(int* __restrict__ d, const int N, const int Round) {
    __shared__ int sPivot[BSZ][BSZ];
    __shared__ int sSelf[BSZ][BSZ];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int half = BSZ >> 1;

    int p = Round * BSZ;

    int bid = blockIdx.x;
    if (bid >= Round) bid++;

    int i = p + ty;
    int j = bid * BSZ + tx;

    sSelf[ty][tx]               = d[i*N + j];
    sSelf[ty][tx + half]        = d[i*N + (j + half)];
    sSelf[ty + half][tx]        = d[(i+half)*N + j];
    sSelf[ty + half][tx + half] = d[(i+half)*N + (j+half)];

    int pi = p + ty;
    int pj = p + tx;

    sPivot[ty][tx]               = d[pi*N + pj];
    sPivot[ty][tx + half]        = d[pi*N + (pj+half)];
    sPivot[ty + half][tx]        = d[(pi+half)*N + pj];
    sPivot[ty + half][tx + half] = d[(pi+half)*N + (pj+half)];

    __syncthreads();

    // #pragma unroll 8
    for (int k=0; k<BSZ; k++) {
        sSelf[ty][tx]               = min(sSelf[ty][tx],               sPivot[ty][k]        + sSelf[k][tx]);
        sSelf[ty][tx + half]        = min(sSelf[ty][tx + half],        sPivot[ty][k]        + sSelf[k][tx + half]);
        sSelf[ty + half][tx]        = min(sSelf[ty + half][tx],        sPivot[ty + half][k] + sSelf[k][tx]);
        sSelf[ty + half][tx + half] = min(sSelf[ty + half][tx + half], sPivot[ty + half][k] + sSelf[k][tx + half]);
        __syncthreads();
    }

    d[i * N + j]               = sSelf[ty][tx];
    d[i * N + (j + half)]      = sSelf[ty][tx + half];
    d[(i + half)*N + j]        = sSelf[ty + half][tx];
    d[(i + half)*N + (j + half)] = sSelf[ty + half][tx + half];
}

__global__ void phase2_col_kernel(int* __restrict__ d, const int N, const int Round) {
    __shared__ int sPivot[BSZ][BSZ];
    __shared__ int sSelf[BSZ][BSZ];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int half = BSZ >> 1;

    int p = Round * BSZ;

    int bid = blockIdx.x;
    if (bid >= Round) bid++;

    int i = bid * BSZ + ty;
    int j = p + tx;

    sSelf[ty][tx]               = d[i * N + j];
    sSelf[ty][tx + half]        = d[i * N + (j + half)];
    sSelf[ty + half][tx]        = d[(i + half) * N + j];
    sSelf[ty + half][tx + half] = d[(i + half) * N + (j + half)];

    int pi = p + ty;
    int pj = p + tx;

    sPivot[ty][tx]               = d[pi * N + pj];
    sPivot[ty][tx + half]        = d[pi * N + (pj + half)];
    sPivot[ty + half][tx]        = d[(pi + half)*N + pj];
    sPivot[ty + half][tx + half] = d[(pi + half)*N + (pj + half)];

    __syncthreads();

    // #pragma unroll 8
    for (int k=0; k<BSZ; k++) {
        sSelf[ty][tx]               = min(sSelf[ty][tx],           sSelf[ty][k] + sPivot[k][tx]);
        sSelf[ty][tx + half]        = min(sSelf[ty][tx + half],    sSelf[ty][k] + sPivot[k][tx + half]);
        sSelf[ty + half][tx]        = min(sSelf[ty+half][tx],      sSelf[ty+half][k] + sPivot[k][tx]);
        sSelf[ty + half][tx + half] = min(sSelf[ty+half][tx+half], sSelf[ty+half][k] + sPivot[k][tx+half]);
        __syncthreads();
    }
    d[i * N + j]               = sSelf[ty][tx];
    d[i * N + (j + half)]      = sSelf[ty][tx + half];
    d[(i + half)*N + j]        = sSelf[ty + half][tx];
    d[(i + half)*N + (j + half)] = sSelf[ty + half][tx + half];
}

// Phase 3
__global__ void phase3_kernel(int* __restrict__ d, const int N, const int Round) {
    __shared__ int sRow[BSZ][BSZ];
    __shared__ int sCol[BSZ][BSZ];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int half = BSZ>>1;

    int p0 = Round * BSZ;

    int bx = blockIdx.x + (blockIdx.x >= Round);
    int by = blockIdx.y + (blockIdx.y >= Round);

    int row_i0 = bx * BSZ;
    int col_j0 = by * BSZ;

    int i = row_i0 + ty;
    int j = col_j0 + tx;

    sRow[ty][tx]             = d[i*N + (p0 + tx)];
    sRow[ty][tx + half]      = d[i*N + (p0 + tx + half)];
    sRow[ty + half][tx]      = d[(i+half)*N + (p0 + tx)];
    sRow[ty + half][tx+half] = d[(i+half)*N + (p0 + tx + half)];

    sCol[ty][tx]             = d[(p0+ty)*N + j];
    sCol[ty][tx + half]      = d[(p0+ty)*N + (j + half)];
    sCol[ty + half][tx]      = d[(p0 + ty + half)*N + j];
    sCol[ty + half][tx+half] = d[(p0 + ty + half)*N + (j + half)];

    __syncthreads();

    int cur00 = d[i*N + j];
    int cur01 = d[i*N + (j + half)];
    int cur10 = d[(i + half)*N + j];
    int cur11 = d[(i + half)*N + (j + half)];

    // #pragma unroll 8
    for (int k=0; k<BSZ; k++) {
        cur00 = min(cur00, sRow[ty][k]        + sCol[k][tx]);
        cur01 = min(cur01, sRow[ty][k]        + sCol[k][tx + half]);
        cur10 = min(cur10, sRow[ty + half][k] + sCol[k][tx]);
        cur11 = min(cur11, sRow[ty + half][k] + sCol[k][tx + half]);
    }
    d[i*N + j]                 = cur00;
    d[i*N + (j + half)]        = cur01;
    d[(i+half)*N + j]          = cur10;
    d[(i+half)*N + (j + half)] = cur11;
}


int* input(const char* file) {
    FILE* f = fopen(file, "rb");
    fread(&original_n, sizeof(int), 1, f);
    fread(&m, sizeof(int), 1, f);
    n = ceil_div(original_n, BSZ) * BSZ;
    int* Dist = (int*)malloc((size_t)n * n * sizeof(int));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Dist[i*n + j] = (i == j && i < original_n ? 0 : INF);
    int e[3];
    for (int i = 0; i < m; i++) {
        fread(e, sizeof(int), 3, f);
        Dist[e[0]*n + e[1]] = e[2];
    }
    fclose(f);
    return Dist;
}

void output(const char* file, int* Dist) {
    FILE* f = fopen(file, "wb");
    for (int i = 0; i < original_n; i++)
        fwrite(&Dist[i*n], sizeof(int), original_n, f);
    fclose(f);
}


void block_FW(int* Dist) {
    int* d_Dist;
    size_t size = (size_t)n*n*sizeof(int);

    cudaMalloc(&d_Dist, size);
    cudaMemcpy(d_Dist, Dist, size, cudaMemcpyHostToDevice);

    int R = n/BSZ;
    dim3 block(32, 32);
    dim3 grid_row(R-1);
    dim3 grid_col(R-1);
    dim3 grid3(R-1, R-1);
    for (int r=0; r<R;r++) {
        phase1_kernel<<<1, block>>>(d_Dist, n, r);
        phase2_row_kernel<<<grid_row, block>>>(d_Dist, n, r);
        phase2_col_kernel<<<grid_col, block>>>(d_Dist, n, r);
        if (R>1)
            phase3_kernel<<<grid3, block>>>(d_Dist, n, r);
    }
    cudaMemcpy(Dist, d_Dist, size, cudaMemcpyDeviceToHost);
    cudaFree(d_Dist);
}

int main(int argc, char* argv[]) {
    if (argc!=3) {
        printf("Usage: %s <input> <output>\n", argv[0]);
        return 1;
    }
    int* Dist = input(argv[1]);
    block_FW(Dist);
    output(argv[2], Dist);
    free(Dist);
    return 0;
}