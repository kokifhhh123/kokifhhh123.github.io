// hybrid_mandel_dynamic_block.cc
#define PNG_NO_SETJMP
#include <mpi.h>
#include <omp.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <emmintrin.h>

#ifndef OMP_BLOCK_W
#define OMP_BLOCK_W 64   // width of a dynamic x-block each thread grabs
#endif
#ifndef OMP_SCHEDULE_CHUNK
#define OMP_SCHEDULE_CHUNK 1
#endif

typedef struct {
    int iters, width, height;
    double left, right, lower, upper;
    double dx, dy;
} SharedData;

static inline void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb"); assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL); assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr); assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    size_t row_size = 3u * (size_t)width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size); assert(row);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * (size_t)width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = (p & 15) * 16;
                } else {
                    color[0] = (p & 15) * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

// ===== SSE2 block kernel (process [i0, i0+cnt)) =====
static inline void compute_block_sse2(const SharedData *s, int j, int i0, int cnt, int *row_out) {
    const double y0d = j * s->dy + s->lower;
    const double dx  = s->dx;

    __m128d y0_vec   = _mm_set1_pd(y0d);
    __m128d four_vec = _mm_set1_pd(4.0);

    int i = i0;
    const int i_end = i0 + cnt;

    // scalar prologue to align to even index for 2-wide SIMD
    if ((i & 1) && i < i_end) {
        double x0_0 = i * dx + s->left;
        double x = 0.0, y = 0.0;
        int k = 0;
        for (; k < s->iters; ++k) {
            double x2 = x*x, y2 = y*y;
            if (x2 + y2 >= 4.0) break;
            double xy = x*y;
            double nx = x2 - y2 + x0_0;
            double ny = 2.0*xy + y0d;
            x = nx; y = ny;
        }
        row_out[i] = k;
        ++i;
    }

    // main 2-wide SIMD
    for (; i + 1 < i_end; i += 2) {
        double x0_0 = i * dx + s->left;
        double x0_1 = (i + 1) * dx + s->left;

        __m128d x0v = _mm_setr_pd(x0_0, x0_1);
        __m128d x   = _mm_setzero_pd();
        __m128d y   = _mm_setzero_pd();

        int cnt0 = 0, cnt1 = 0;
        int mask_active = 3; // both lanes active initially

        for (int k = 0; k < s->iters && mask_active; ++k) {
            __m128d x2 = _mm_mul_pd(x, x);
            __m128d y2 = _mm_mul_pd(y, y);
            __m128d xy = _mm_mul_pd(x, y);

            __m128d nx = _mm_add_pd(_mm_sub_pd(x2, y2), x0v);
            __m128d ny = _mm_add_pd(_mm_add_pd(xy, xy), y0_vec);

            __m128d len2 = _mm_add_pd(_mm_mul_pd(nx, nx), _mm_mul_pd(ny, ny));

            if (mask_active & 1) ++cnt0;
            if (mask_active & 2) ++cnt1;

            int stay = _mm_movemask_pd(_mm_cmplt_pd(len2, four_vec));
            mask_active &= stay;

            x = nx; y = ny;
        }
        row_out[i]   = cnt0;
        row_out[i+1] = cnt1;
    }

    // scalar epilogue (last odd pixel if needed)
    if (i < i_end) {
        double x0_0 = i * dx + s->left;
        double x = 0.0, y = 0.0;
        int k = 0;
        for (; k < s->iters; ++k) {
            double x2 = x*x, y2 = y*y;
            if (x2 + y2 >= 4.0) break;
            double xy = x*y;
            double nx = x2 - y2 + x0_0;
            double ny = 2.0*xy + y0d;
            x = nx; y = ny;
        }
        row_out[i] = k;
    }
}

static inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    if (argc != 9) {
        if (rank == 0) fprintf(stderr, "Usage: %s out.png iters left right lower upper width height\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const char* filename = argv[1];
    SharedData S;
    S.iters = (int)strtol(argv[2], 0, 10);
    S.left  = strtod(argv[3], 0);
    S.right = strtod(argv[4], 0);
    S.lower = strtod(argv[5], 0);
    S.upper = strtod(argv[6], 0);
    S.width = (int)strtol(argv[7], 0, 10);
    S.height= (int)strtol(argv[8], 0, 10);
    S.dx = (S.right - S.left) / (double)S.width;
    S.dy = (S.upper - S.lower) / (double)S.height;

    const int H = S.height, W = S.width;

    // Interleaved rows by ranks
    const int local_rows = (H > rank) ? ceil_div_int(H - rank, world) : 0;

    int *rows = NULL;
    if (local_rows > 0) {
        rows = (int*)malloc((size_t)local_rows * sizeof(int)); assert(rows);
        for (int lr = 0; lr < local_rows; ++lr) rows[lr] = rank + lr * world;
    }

    int *local_img = NULL;
    if (local_rows > 0) {
        local_img = (int*)malloc((size_t)local_rows * (size_t)W * sizeof(int));
        assert(local_img);
    }

    omp_lock_t lk;
    omp_init_lock(&lk);
    int sched_lr = 0;
    int sched_x  = 0;

    #pragma omp parallel
    {
        while (1) {
            int lr, x0, cnt, j_global;
            omp_set_lock(&lk);
            while (sched_lr < local_rows && sched_x >= W) {
                ++sched_lr;
                sched_x = 0;
            }
            if (sched_lr >= local_rows) {
                omp_unset_lock(&lk);
                break;
            }
            lr = sched_lr;
            x0 = sched_x;
            cnt = OMP_BLOCK_W;
            if (x0 + cnt > W) cnt = W - x0;
            sched_x += cnt;
            j_global = rows[lr];
            omp_unset_lock(&lk);

            int *row_out = local_img + (size_t)lr * (size_t)W;
            compute_block_sse2(&S, j_global, x0, cnt, row_out);
        }
    }
    omp_destroy_lock(&lk);

    int send_count = local_rows * W;
    int *recv_counts = NULL, *displs = NULL;
    int *tmp_packed = NULL;
    int *full_img = NULL;

    if (rank == 0) {
        recv_counts = (int*)malloc((size_t)world * sizeof(int)); assert(recv_counts);
        displs = (int*)malloc((size_t)world * sizeof(int)); assert(displs);
    }

    MPI_Gather(&send_count, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int total_ints = 0;
        for (int r = 0; r < world; ++r) {
            displs[r] = total_ints;
            int r_local_rows = (H > r) ? ceil_div_int(H - r, world) : 0;
            recv_counts[r] = r_local_rows * W;
            total_ints += recv_counts[r];
        }
        tmp_packed = (int*)malloc((size_t)total_ints * sizeof(int)); assert(tmp_packed);
    }

    MPI_Gatherv(local_img, send_count, MPI_INT,
                tmp_packed, recv_counts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        full_img = (int*)malloc((size_t)H * (size_t)W * sizeof(int)); assert(full_img);
        for (int r = 0; r < world; ++r) {
            int r_local_rows = (H > r) ? ceil_div_int(H - r, world) : 0;
            if (r_local_rows <= 0) continue;
            int *src = tmp_packed + displs[r];
            for (int lr = 0; lr < r_local_rows; ++lr) {
                int j = r + lr * world;
                if (j >= H) break;
                memcpy(full_img + (size_t)j * (size_t)W,
                       src + (size_t)lr * (size_t)W,
                       (size_t)W * sizeof(int));
            }
        }
        write_png(filename, S.iters, W, H, full_img);
    }
    if (rank == 0) {
        free(full_img);
        free(tmp_packed);
        free(recv_counts);
        free(displs);
    }
    free(local_img);
    free(rows);
    MPI_Finalize();
    return 0;
}