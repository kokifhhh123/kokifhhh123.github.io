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

// ---- PNG I/O (host) ----
int read_png(const char* filename, unsigned char** image, unsigned* height,
             unsigned* width, unsigned* channels) {
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
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_read_update_info(png_ptr, info_ptr);
    png_uint_32 rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = png_get_channels(png_ptr, info_ptr);

    *image = (unsigned char*)malloc(rowbytes * (*height));
    if (!*image) { png_destroy_read_struct(&png_ptr, &info_ptr, NULL); return 3; }

    std::vector<png_bytep> row_pointers(*height);
    for (unsigned i = 0; i < *height; ++i) row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers.data());
    png_read_end(png_ptr, NULL);
    fclose(infile);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width,
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_set_compression_level(png_ptr, 1);
    png_write_info(png_ptr, info_ptr);

    std::vector<png_bytep> row_ptr(height);
    for (unsigned i = 0; i < height; ++i)
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    png_write_image(png_ptr, row_ptr.data());
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

// ---- CUDA bits ----
#define CUDA_CHECK(stmt) do { \
    cudaError_t err = (stmt); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #stmt, __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Same masks as CPU version; placed in constant memory.
__constant__ int d_mask[MASK_N][MASK_X][MASK_Y];

__global__ void sobel_kernel(const unsigned char* __restrict__ s,
                             unsigned char* __restrict__ t,
                             unsigned height, unsigned width, unsigned channels) {
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int xBound = MASK_X / 2;
    const int yBound = MASK_Y / 2;
    const int adjustX = (MASK_X % 2) ? 1 : 0;
    const int adjustY = (MASK_Y % 2) ? 1 : 0;

    double valR[MASK_N] = {0.0}, valG[MASK_N] = {0.0}, valB[MASK_N] = {0.0};

    for (int i = 0; i < MASK_N; ++i) {
        for (int v = -yBound; v < yBound + adjustY; ++v) {
            int yy = int(y) + v;
            if (yy < 0 || yy >= int(height)) continue;
            for (int u = -xBound; u < xBound + adjustX; ++u) {
                int xx = int(x) + u;
                if (xx < 0 || xx >= int(width)) continue;

                size_t idx = channels * (size_t(width) * yy + xx);
                // Keep same B,G,R indexing convention as your code (+0=B, +1=G, +2=R)
                int B = s[idx + 0];
                int G = s[idx + 1];
                int R = s[idx + 2];

                int k = d_mask[i][u + xBound][v + yBound];
                valR[i] += R * k;
                valG[i] += G * k;
                valB[i] += B * k;
            }
        }
    }

    double totalR = 0.0, totalG = 0.0, totalB = 0.0;
    for (int i = 0; i < MASK_N; ++i) {
        totalR += valR[i] * valR[i];
        totalG += valG[i] * valG[i];
        totalB += valB[i] * valB[i];
    }
    totalR = sqrt(totalR) / SCALE;
    totalG = sqrt(totalG) / SCALE;
    totalB = sqrt(totalB) / SCALE;

    unsigned char cR = (totalR > 255.0) ? 255 : (unsigned char)totalR;
    unsigned char cG = (totalG > 255.0) ? 255 : (unsigned char)totalG;
    unsigned char cB = (totalB > 255.0) ? 255 : (unsigned char)totalB;

    size_t outIdx = channels * (size_t(width) * y + x);
    t[outIdx + 2] = cR;
    t[outIdx + 1] = cG;
    t[outIdx + 0] = cB;

    // If input has alpha (channels==4), pass it through.
    if (channels == 4) t[outIdx + 3] = s[outIdx + 3];
}

int main(int argc, char** argv) {
    assert(argc == 3);
    unsigned H, W, C;
    unsigned char* h_src = nullptr;
    int err = read_png(argv[1], &h_src, &H, &W, &C);
    if (err) { fprintf(stderr, "read_png error %d\n", err); return 1; }

    // libpng may give RGBA; our writer forces RGB. We'll keep C as-is in device,
    // but write_png uses 'channels' to compute row stride; we pass same C back.
    unsigned char* h_dst = (unsigned char*)malloc((size_t)H * W * C);
    if (!h_dst) { fprintf(stderr, "malloc failed\n"); return 1; }

    // Copy mask to constant memory
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

    // Device buffers
    const size_t bytes = (size_t)H * W * C * sizeof(unsigned char);
    unsigned char *d_src = nullptr, *d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));

    // Launch
    dim3 block(32, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);

    sobel_kernel<<<grid, block>>>(d_src, d_dst, H, W, C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, bytes, cudaMemcpyDeviceToHost));

    // Output (writer expects RGB layout; we preserved original channel count for stride)
    write_png(argv[2], h_dst, H, W, C);

    cudaFree(d_src);
    cudaFree(d_dst);
    free(h_src);
    free(h_dst);
    return 0;
}