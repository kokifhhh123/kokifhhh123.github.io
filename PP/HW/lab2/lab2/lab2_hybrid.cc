#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

typedef unsigned long long ull;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 3) {
        if (world_rank == 0)
            fprintf(stderr, "must provide exactly 2 arguments!\n");
        MPI_Finalize();
        return 1;
    }

    ull r = atoll(argv[1]);
    ull k = atoll(argv[2]);
    ull local_sum = 0, global_sum = 0;

    ull chunk = r / world_size;
    ull start = world_rank * chunk;
    ull end = (world_rank == world_size - 1) ? r : start + chunk;

    #pragma omp parallel for reduction(+:local_sum) schedule(static)
    for (ull x = start; x < end; x++) {
        long double y = ceil(sqrtl((long double)r * r - (long double)x * x));
        local_sum += (ull)y;
    }

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("%llu\n", (4 * (global_sum % k)) % k);
    }

    MPI_Finalize();
    return 0;
}
