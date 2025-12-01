#include <bits/stdc++.h>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>
#include <cuda_runtime.h>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define SCALE 8

// ---------------- PNG I/O (host) ----------------
int read_png(const char* filename, unsigned char** image, unsigned* h,
             unsigned* w, unsigned* ch) {
    unsigned char sig[8];
    FILE* infile = fopen(filename, "rb");
    if (!infile) return 2;
    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8)) return 1;
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) return 4;
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) { png_destroy_read_struct(&png_ptr, NULL, NULL); return 4; }
    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, w, h, &bit_depth, &color_type, NULL, NULL, NULL);
    png_read_update_info(png_ptr, info_ptr);
    png_uint_32 rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *ch = png_get_channels(png_ptr, info_ptr);
    *image = (unsigned char*)malloc(rowbytes * (*h));
    if (!*image) { png_destroy_read_struct(&png_ptr, &info_ptr, NULL); return 3; }
    std::vector<png_bytep> row_pointers(*h);
    for (unsigned i = 0; i < *h; ++i) row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers.data());
    png_read_end(png_ptr, NULL);
    fclose(infile);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned h, const unsigned w,
               const unsigned ch) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, w, h, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_set_compression_level(png_ptr, 1);
    png_write_info(png_ptr, info_ptr);
    std::vector<png_bytep> row_ptr(h);
    for (unsigned i = 0; i < h; ++i) row_ptr[i] = image + i * w * ch * sizeof(unsigned char);
    png_write_image(png_ptr, row_ptr.data());
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

// ---------------- CUDA utils ----------------
#define CUDA_CHECK(x) do { cudaError_t err=(x); if (err!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s failed @%s:%d: %s\n", #x,__FILE__,__LINE__,cudaGetErrorString(err)); exit(1);} } while(0)

__constant__ int d_mask[MASK_N][MASK_X][MASK_Y];

// tile config
#ifndef TILE_X
#define TILE_X 32
#endif
#ifndef TILE_Y
#define TILE_Y 32
#endif

__global__ void sobel(const unsigned char* __restrict__ src,
                      unsigned char* __restrict__ dst,
                      unsigned H, unsigned W, unsigned C)
{
    // shared tile stores RGB only (3 chans), with 2-pixel halo (5x5 kernel)
    const int HALO = 2;
    const int tileW = TILE_X + 2*HALO;
    const int tileH = TILE_Y + 2*HALO;
    extern __shared__ unsigned char smem[]; // size = tileW*tileH*3

    auto clampi = [](int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); };

    // global coord of output pixel this thread computes
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;

    const int gx0 = blockIdx.x * blockDim.x - HALO;
    const int gy0 = blockIdx.y * blockDim.y - HALO;

    for (int sy = threadIdx.y; sy < tileH; sy += blockDim.y) {
        int gy = clampi(gy0 + sy, 0, (int)H - 1);

        // coalesced along x (threads move in x)
        if (C == 4) {
            // aligned 4B loads when possible
            const uchar4* row4 = reinterpret_cast<const uchar4*>(src + (size_t)gy * W * C);
            for (int sx = threadIdx.x; sx < tileW; sx += blockDim.x) {
                int gx = clampi(gx0 + sx, 0, (int)W - 1);
                uchar4 p = row4[gx]; // B,G,R,A
                int off = (sy * tileW + sx) * 3;
                smem[off + 0] = p.x; // B
                smem[off + 1] = p.y; // G
                smem[off + 2] = p.z; // R
            }
        } else { // C == 3 (or other, default read first 3)
            const unsigned char* row = src + (size_t)gy * W * C;
            for (int sx = threadIdx.x; sx < tileW; sx += blockDim.x) {
                int gx = clampi(gx0 + sx, 0, (int)W - 1);
                const unsigned char* p = row + (size_t)gx * C;
                int off = (sy * tileW + sx) * 3;
                smem[off + 0] = p[0];
                smem[off + 1] = p[1];
                smem[off + 2] = p[2];
            }
        }
    }
    __syncthreads();

    if (ox >= (int)W || oy >= (int)H) return;

    // compute on shared memory window centered at (threadIdx + HALO)
    const int sx0 = threadIdx.x + HALO;
    const int sy0 = threadIdx.y + HALO;

    float accR[MASK_N] = {0.f, 0.f};
    float accG[MASK_N] = {0.f, 0.f};
    float accB[MASK_N] = {0.f, 0.f};

#pragma unroll
    for (int mi = 0; mi < MASK_N; ++mi) {
#pragma unroll
        for (int ky = 0; ky < MASK_Y; ++ky) {
            const int sy = sy0 + ky - HALO;
#pragma unroll
            for (int kx = 0; kx < MASK_X; ++kx) {
                const int sx = sx0 + kx - HALO;
                const int off = (sy * tileW + sx) * 3;
                const float B = (float)smem[off + 0];
                const float G = (float)smem[off + 1];
                const float R = (float)smem[off + 2];
                const int k = d_mask[mi][kx][ky];
                accR[mi] += R * k;
                accG[mi] += G * k;
                accB[mi] += B * k;
            }
        }
    }

    float r = (accR[0]*accR[0] + accR[1]*accR[1]);
    float g = (accG[0]*accG[0] + accG[1]*accG[1]);
    float b = (accB[0]*accB[0] + accB[1]*accB[1]);

    r = __fsqrt_rn(r) * (1.0f / SCALE);
    g = __fsqrt_rn(g) * (1.0f / SCALE);
    b = __fsqrt_rn(b) * (1.0f / SCALE);

    unsigned char R8 = (unsigned char)min(255.0f, r);
    unsigned char G8 = (unsigned char)min(255.0f, g);
    unsigned char B8 = (unsigned char)min(255.0f, b);

    const size_t outIdx = (size_t)(oy) * W * C + (size_t)ox * C;
    dst[outIdx + 2] = R8;
    dst[outIdx + 1] = G8;
    dst[outIdx + 0] = B8;
    if (C == 4) dst[outIdx + 3] = src[outIdx + 3];
}

int main(int argc, char** argv) {
    assert(argc == 3);
    unsigned H, W, C;
    unsigned char* h_src = nullptr;
    int err = read_png(argv[1], &h_src, &H, &W, &C);
    if (err) { fprintf(stderr, "read_png error %d\n", err); return 1; }
    unsigned char* h_dst = (unsigned char*)malloc((size_t)H * W * C);
    if (!h_dst) { fprintf(stderr, "malloc failed\n"); return 1; }

    // masks to constant
    int h_mask[MASK_N][MASK_X][MASK_Y] = {
        { { -1, -4, -6, -4, -1 },
          { -2, -8,-12, -8, -2 },
          {  0,  0,  0,  0,  0 },
          {  2,  8, 12,  8,  2 },
          {  1,  4,  6,  4,  1 } },
        { { -1, -2,  0,  2,  1 },
          { -4, -8,  0,  8,  4 },
          { -6,-12,  0, 12,  6 },
          { -4, -8,  0,  8,  4 },
          { -1, -2,  0,  2,  1 } }
    };
    CUDA_CHECK(cudaMemcpyToSymbol(d_mask, h_mask, sizeof(h_mask)));

    const size_t bytes = (size_t)H * W * C;
    unsigned char *d_src = nullptr, *d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE_X, TILE_Y);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    // prefer shared memory
    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

    // shared bytes: (tileW*tileH*3)
    size_t smem_bytes = (size_t)(TILE_X + 4) * (TILE_Y + 4) * 3;

    sobel<<<grid, block, smem_bytes>>>(d_src, d_dst, H, W, C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, bytes, cudaMemcpyDeviceToHost));

    write_png(argv[2], h_dst, H, W, C);

    cudaFree(d_src); cudaFree(d_dst);
    free(h_src); free(h_dst);
    return 0;
}
