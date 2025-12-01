#define PNG_NO_SETJMP
#include <sched.h>
#include <emmintrin.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <atomic>

std::atomic<int> next_row(0);
#define CHUNK 2   // 可調整行分配大小

typedef struct {
    int iters, width, height;
    double left, right, lower, upper;
    double dx, dy;
    int *image;
} SharedData;

typedef struct {
    int id;
    int nthreads;
    SharedData *shared;
} ThreadArg;

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
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

void compute_row_sse2(int j, SharedData *s) {
    double y0 = j * s->dy + s->lower;
    __m128d y0_vec = _mm_set1_pd(y0);
    __m128d four_vec = _mm_set1_pd(4.0);

    int base = j*s->width;
    int* row_out = s->image+base;

    for (int i=0; i<s->width; i+=2) {
        int valid = (s->width - i < 2) ? (s->width - i) : 2;

        __m128d x0 = _mm_setr_pd(i * s->dx + s->left,
                                 (i + 1) * s->dx + s->left);
        __m128d x = _mm_setzero_pd();
        __m128d y = _mm_setzero_pd();

        int cnt0 = 0, cnt1 = 0;
        int active_mask = (valid == 2) ? 3 : 1; // lane 0/1 是否仍需計算

        for (int k = 0; k < s->iters && active_mask; ++k) {
            __m128d x2 = _mm_mul_pd(x, x);
            __m128d y2 = _mm_mul_pd(y, y);
            __m128d xy = _mm_mul_pd(x, y);

            __m128d new_x = _mm_add_pd(_mm_sub_pd(x2, y2), x0);
            __m128d new_y = _mm_add_pd(_mm_add_pd(xy, xy), y0_vec);

            __m128d len2 = _mm_add_pd(_mm_mul_pd(new_x, new_x),
                                      _mm_mul_pd(new_y, new_y));
            cnt0 += (active_mask & 1) ? 1 : 0;
            cnt1 += (active_mask & 2) ? 1 : 0;
            int stay = _mm_movemask_pd(_mm_cmplt_pd(len2, four_vec));
            active_mask &= stay;
            x = new_x;
            y = new_y;
        }

        row_out[i] = cnt0;
        if (valid == 2) row_out[i + 1] = cnt1;
    }
}

void* worker(void* arg) {
    ThreadArg *t = (ThreadArg*)arg;
    SharedData *s = t->shared;

    while (1) {
        int j_start = next_row.fetch_add(CHUNK);
        if (j_start >= s->height) break;

        int j_end = j_start + CHUNK;
        if (j_end > s->height) j_end = s->height;

        for (int j = j_start; j < j_end; ++j)
            compute_row_sse2(j, s);
    }
    return NULL;
}

int main(int argc, char** argv) {
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int ncpus = CPU_COUNT(&cpu_set);
    printf("%d cpus available\n", ncpus);

    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    int *image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    SharedData shared = {
        iters, width, height, left, right, lower, upper,
        (right - left) / width,
        (upper - lower) / height,
        image
    };

    pthread_t threads[ncpus];
    ThreadArg args[ncpus];

    next_row.store(0);
    for (int i = 0; i < ncpus; ++i) {
        args[i].id = i;
        args[i].nthreads = ncpus;
        args[i].shared = &shared;
        pthread_create(&threads[i], NULL, worker, &args[i]);
    }

    for (int i = 0; i < ncpus; ++i)
        pthread_join(threads[i], NULL);

    write_png(filename, iters, width, height, image);
    free(image);
    return 0;
}