#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0, world_size = 12;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 3) {
        if (world_rank == 0) fprintf(stderr, "must provide exactly 2 arguments!\n");
        MPI_Finalize();
        return 1;
    }

    unsigned long long r = strtoull(argv[1], NULL, 10);
    unsigned long long k = strtoull(argv[2], NULL, 10);

    unsigned long long chunk = (r + world_size - 1) / world_size; // ceiling
    unsigned long long start = world_rank * chunk;
    unsigned long long end   = (start + chunk > r) ? r : (start + chunk);

    unsigned long long local_pixels_mod = 0;
    // unsigned long long buffer = 0;

    double t0 = MPI_Wtime();
    // double rf = (double)r;
    
    long double rf = (long double)r;
    long double r2 = rf * rf;

    #pragma omp parallel for simd reduction(+:local_pixels_mod) schedule(static)
    for (unsigned long long x = start; x < end; ++x) {
        long double xf = (long double)x;
        long double yy = sqrtl(r2 - xf * xf);
        unsigned long long y = (unsigned long long)yy;
        if ((long double)y < yy) y++;  // long double
        local_pixels_mod += (y % k);
        if (local_pixels_mod >= k) local_pixels_mod -= k;
    }

    unsigned long long global_pixels_mod = 0;
    MPI_Reduce(&local_pixels_mod, &global_pixels_mod, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        global_pixels_mod %= k;
        unsigned long long ans = (4 * global_pixels_mod) % k;
        double t1 = MPI_Wtime();
        printf("%llu\n", ans);
        // fprintf(stderr, "[hybrid] time = %.6f s, ranks=%d, OMP_THREADS=%d\n",
        //         t1 - t0, world_size, omp_get_max_threads());
    }

    MPI_Finalize();
    return 0;
}
