#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BSZ 64
const int INF = (1<<30)-1;

int original_n, m, n;

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

__global__ void phase1_kernel(int* __restrict__ d, int N, int Round) {
    __shared__ int s[BSZ][BSZ];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int half = BSZ >> 1;
    int base = Round * BSZ;

    int i0 = base + ty;
    int j0 = base + tx;

    int row0 = i0 * N;
    int row1 = (i0 + half) * N;

    // load pivot block
    s[ty][tx]                 = d[row0 + j0];
    s[ty][tx + half]          = d[row0 + (j0 + half)];
    s[ty + half][tx]          = d[row1 + j0];
    s[ty + half][tx + half]   = d[row1 + (j0 + half)];

    __syncthreads();

    // #pragma unroll 64
    for (int k=0; k<BSZ; k++) {
        s[ty][tx]               = min(s[ty][tx],               s[ty][k] + s[k][tx]);
        s[ty][tx + half]        = min(s[ty][tx + half],        s[ty][k] + s[k][tx + half]);
        s[ty + half][tx]        = min(s[ty + half][tx],        s[ty + half][k] + s[k][tx]);
        s[ty + half][tx + half] = min(s[ty + half][tx + half], s[ty + half][k] + s[k][tx + half]);
        __syncthreads();
    }

    d[row0 + j0]                 = s[ty][tx];
    d[row0 + (j0 + half)]        = s[ty][tx + half];
    d[row1 + j0]                 = s[ty + half][tx];
    d[row1 + (j0 + half)]        = s[ty + half][tx + half];
}


__global__ void phase2_row_kernel(int* __restrict__ d, const int N, const int Round) {
    __shared__ int sPivot[BSZ][BSZ];
    __shared__ int sSelf[BSZ][BSZ];

    int tx   = threadIdx.x;
    int ty   = threadIdx.y;
    int half = BSZ >> 1;

    int p = Round * BSZ;

    int bid = blockIdx.x;
    if (bid >= Round) bid++;

    int i = p + ty;
    int j = bid * BSZ + tx;

    int iN      = i * N;
    int iHalfN  = (i + half) * N;

    sSelf[ty][tx]               = d[iN + j];
    sSelf[ty][tx + half]        = d[iN + (j + half)];
    sSelf[ty + half][tx]        = d[iHalfN + j];
    sSelf[ty + half][tx + half] = d[iHalfN + (j + half)];

    int pi = p + ty;
    int pj = p + tx;

    int piN      = pi * N;
    int piHalfN  = (pi + half) * N;

    sPivot[ty][tx]               = d[piN + pj];
    sPivot[ty][tx + half]        = d[piN + (pj + half)];
    sPivot[ty + half][tx]        = d[piHalfN + pj];
    sPivot[ty + half][tx + half] = d[piHalfN + (pj + half)];

    __syncthreads();

    // #pragma unroll 64
    for (int k=0; k<BSZ; k++) {
        sSelf[ty][tx]               = min(sSelf[ty][tx],               sPivot[ty][k]        + sSelf[k][tx]);
        sSelf[ty][tx + half]        = min(sSelf[ty][tx + half],        sPivot[ty][k]        + sSelf[k][tx + half]);
        sSelf[ty + half][tx]        = min(sSelf[ty + half][tx],        sPivot[ty + half][k] + sSelf[k][tx]);
        sSelf[ty + half][tx + half] = min(sSelf[ty + half][tx + half], sPivot[ty + half][k] + sSelf[k][tx + half]);
        __syncthreads();
    }

    d[iN + j]                 = sSelf[ty][tx];
    d[iN + (j + half)]        = sSelf[ty][tx + half];
    d[iHalfN + j]             = sSelf[ty + half][tx];
    d[iHalfN + (j + half)]    = sSelf[ty + half][tx + half];
}


__global__ void phase2_col_kernel(int* __restrict__ d, const int N, const int Round,
                                  int rowStart, int rowEnd) {
    __shared__ int sPivot[BSZ][BSZ];
    __shared__ int sSelf[BSZ][BSZ];

    int tx   = threadIdx.x;
    int ty   = threadIdx.y;
    int half = BSZ >> 1;

    int p = Round * BSZ;

    int bid = blockIdx.x;
    if (bid >= Round) bid++;     // skip pivot block row

    int i = bid * BSZ + ty;       // global row index
    int j = p        + tx;       // global col index (pivot column band)


    if (i < rowStart || i >= rowEnd) return;

    int iN      = i * N;
    int iHalfN  = (i + half) * N;

    sSelf[ty][tx]               = d[iN + j];
    sSelf[ty][tx + half]        = d[iN + (j + half)];
    sSelf[ty + half][tx]        = d[iHalfN + j];
    sSelf[ty + half][tx + half] = d[iHalfN + (j + half)];

    int pi = p + ty;
    int pj = p + tx;

    int piN      = pi * N;
    int piHalfN  = (pi + half) * N;

    sPivot[ty][tx]               = d[piN + pj];
    sPivot[ty][tx + half]        = d[piN + (pj + half)];
    sPivot[ty + half][tx]        = d[piHalfN + pj];
    sPivot[ty + half][tx + half] = d[piHalfN + (pj + half)];

    __syncthreads();

    // #pragma unroll 64
    for (int k=0; k<BSZ; k++) {
        sSelf[ty][tx]               = min(sSelf[ty][tx],           sSelf[ty][k] + sPivot[k][tx]);
        sSelf[ty][tx + half]        = min(sSelf[ty][tx + half],    sSelf[ty][k] + sPivot[k][tx + half]);
        sSelf[ty + half][tx]        = min(sSelf[ty+half][tx],      sSelf[ty+half][k] + sPivot[k][tx]);
        sSelf[ty + half][tx + half] = min(sSelf[ty+half][tx+half], sSelf[ty+half][k] + sPivot[k][tx+half]);
        __syncthreads();
    }

    d[iN + j]                = sSelf[ty][tx];
    d[iN + (j + half)]       = sSelf[ty][tx + half];
    d[iHalfN + j]            = sSelf[ty + half][tx];
    d[iHalfN + (j + half)]   = sSelf[ty + half][tx + half];
}


__global__ void phase3_kernel(int* __restrict__ d, int N, int Round,
                              int rowStart, int rowEnd) {
    __shared__ int sRow[BSZ][BSZ];
    __shared__ int sCol[BSZ][BSZ];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int half = BSZ >> 1;
    int p0 = Round * BSZ;

    int bx = blockIdx.x + (blockIdx.x >= Round);
    int by = blockIdx.y + (blockIdx.y >= Round);

    int row_i0 = bx * BSZ;
    int col_j0 = by * BSZ;

    int i = row_i0 + ty;
    int j = col_j0 + tx;


    if (i<rowStart || i>=rowEnd) return;

    sRow[ty][tx]             = d[i*N + (p0 + tx)];
    sRow[ty][tx + half]      = d[i*N + (p0 + tx + half)];
    sRow[ty + half][tx]      = d[(i+half)*N + (p0 + tx)];
    sRow[ty + half][tx+half] = d[(i+half)*N + (p0 + tx + half)];

    sCol[ty][tx]             = d[(p0+ty)*N + j];
    sCol[ty][tx + half]      = d[(p0+ty)*N + (j + half)];
    sCol[ty + half][tx]      = d[(p0 + ty + half)*N + j];
    sCol[ty + half][tx+half] = d[(p0 + ty + half)*N + (j + half)];

    __syncthreads();

    int row0 = i * N;
    int row1 = (i + half) * N;
    int cur00 = d[row0 + j];
    int cur01 = d[row0 + j + half];
    int cur10 = d[row1 + j];
    int cur11 = d[row1 + j + half];

    // #pragma unroll 64
    for (int k=0; k<BSZ; k++) {
        cur00 = min(cur00, sRow[ty][k]        + sCol[k][tx]);
        cur01 = min(cur01, sRow[ty][k]        + sCol[k][tx + half]);
        cur10 = min(cur10, sRow[ty + half][k] + sCol[k][tx]);
        cur11 = min(cur11, sRow[ty + half][k] + sCol[k][tx + half]);
    }

    d[row0 + j]        = cur00;
    d[row0 + j + half] = cur01;
    d[row1 + j]        = cur10;
    d[row1 + j + half] = cur11;
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
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    int R = n / BSZ;
    dim3 block(32, 32);
    dim3 grid_row(R - 1);
    dim3 grid_col(R - 1);
    dim3 grid3(R - 1, R - 1);
    size_t size = (size_t)n * n * sizeof(int);

    const int GPU0 = 0;
    const int GPU1 = 1;

    int* d_Dist[2];

int splitBlock = R / 2;        // block-row
int splitRow   = splitBlock * BSZ;
int rowStart[2] = {0,        splitRow};
int rowEnd  [2] = {splitRow, n       };


    int can01 = 0, can10 = 0;
    cudaDeviceCanAccessPeer(&can01, GPU0, GPU1);
    cudaDeviceCanAccessPeer(&can10, GPU1, GPU0);
    if (can01) {
        cudaSetDevice(GPU0);
        cudaDeviceEnablePeerAccess(GPU1, 0);
    }
    if (can10) {
        cudaSetDevice(GPU1);
        cudaDeviceEnablePeerAccess(GPU0, 0);
    }

    for (int g = 0; g < 2; ++g) {
        cudaSetDevice(g);
        cudaMalloc(&d_Dist[g], size);
        cudaMemcpy(d_Dist[g], Dist, size, cudaMemcpyHostToDevice);
    }

    size_t bandBytes = (size_t)BSZ * n * sizeof(int);

    for (int r=0; r<R;r++) {
        int p = r * BSZ;

        int pivotOwner = (p < splitRow) ? GPU0 : GPU1;
        int other      = pivotOwner ^ 1;

        cudaSetDevice(pivotOwner);
        phase1_kernel<<<1, block>>>(d_Dist[pivotOwner], n, r);
        phase2_row_kernel<<<grid_row, block>>>(d_Dist[pivotOwner], n, r);
        cudaDeviceSynchronize();

        cudaMemcpyPeer(
            d_Dist[other] + (size_t)p * n, other,
            d_Dist[pivotOwner] + (size_t)p * n, pivotOwner,
            bandBytes
        );

        for (int g=0; g<2; ++g) {
            cudaSetDevice(g);
            phase2_col_kernel<<<grid_col, block>>>(d_Dist[g], n, r,
                                                   rowStart[g], rowEnd[g]);
        }
        for (int g=0; g<2; ++g) {
            cudaSetDevice(g);
            cudaDeviceSynchronize();
        }
        if (R > 1) {
            for (int g=0; g<2; ++g) {
                cudaSetDevice(g);
                phase3_kernel<<<grid3, block>>>(d_Dist[g], n, r,
                                                rowStart[g], rowEnd[g]);
            }
            for (int g=0; g<2; ++g) {
                cudaSetDevice(g);
                cudaDeviceSynchronize();
            }
        }
    }
    cudaSetDevice(GPU0);
    cudaMemcpy(Dist + (size_t)rowStart[0] * n,
               d_Dist[GPU0] + (size_t)rowStart[0] * n,
               (size_t)(rowEnd[0] - rowStart[0]) * n * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaSetDevice(GPU1);
    cudaMemcpy(Dist + (size_t)rowStart[1] * n,
               d_Dist[GPU1] + (size_t)rowStart[1] * n,
               (size_t)(rowEnd[1] - rowStart[1]) * n * sizeof(int),
               cudaMemcpyDeviceToHost);

    for (int g=0; g<2; ++g) {
        cudaSetDevice(g);
        cudaFree(d_Dist[g]);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input> <output>\n", argv[0]);
        return 1;
    }
    int* Dist = input(argv[1]);
    block_FW(Dist);
    output(argv[2], Dist);
    free(Dist);
    return 0;
}