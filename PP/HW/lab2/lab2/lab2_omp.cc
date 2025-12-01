#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }

    unsigned long long r = atoll(argv[1]);
    unsigned long long k = atoll(argv[2]);
    unsigned long long pixels = 0;

	unsigned long long rr = r * r;
	#pragma omp parallel for reduction(+:pixels) schedule(static)
	for (unsigned long long x = 0; x < r; x++) {
		long double val = sqrtl((long double)rr - (long double)x * x);
		pixels += (unsigned long long)ceil(val);
	}

    printf("%llu\n", (4 * (pixels % k)) % k);
    return 0;
}