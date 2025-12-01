#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

typedef unsigned long long ull;

int num_threads;
ull r, k;
ull* partial_sum;

void* worker(void* arg) {
    int tid = *(int*)arg;
    ull local_sum = 0;

    ull chunk = r / num_threads;
    ull start = tid * chunk;
    ull end = (tid == num_threads - 1) ? r : start + chunk;

    for (ull x = start; x < end; x++) {
        ull y = ceil(sqrtl((long double)r * r - (long double)x * x));
        local_sum += y;
    }

    partial_sum[tid] = local_sum;
    return NULL;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }

    r = atoll(argv[1]);
    k = atoll(argv[2]);

    num_threads = 4;

    pthread_t threads[num_threads];
    int tid[num_threads];
    partial_sum = (ull*)calloc(num_threads, sizeof(ull));

    for (int i = 0; i < num_threads; i++) {
        tid[i] = i;
        pthread_create(&threads[i], NULL, worker, &tid[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    ull total = 0;
    for (int i = 0; i < num_threads; i++) {
        total += partial_sum[i];
    }

    printf("%llu\n", (4 * (total % k)) % k);
    free(partial_sum);
    return 0;
}
