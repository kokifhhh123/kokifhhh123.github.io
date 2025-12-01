#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

const int INF = (1 << 30) - 1;
int n, m;
int **Dist;

static inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

void input(const char *file) {
    FILE *f = fopen(file, "rb");
    fread(&n, sizeof(int), 1, f);
    fread(&m, sizeof(int), 1, f);

    Dist = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; ++i) {
        Dist[i] = (int*)malloc(n * sizeof(int));
        for (int j = 0; j < n; ++j)
            Dist[i][j] = (i == j ? 0 : INF);
    }

    int p[3];
    for (int i = 0; i < m; ++i) {
        fread(p, sizeof(int), 3, f);
        Dist[p[0]][p[1]] = p[2];
    }
    fclose(f);
}

void output(const char *file) {
    FILE *f = fopen(file, "wb");
    for (int i = 0; i < n; ++i)
        fwrite(Dist[i], sizeof(int), n, f);
    fclose(f);
}

static inline void update_block(int i_start, int i_end,
                                int j_start, int j_end,
                                int k_start, int k_end)
{
    for (int k = k_start; k < k_end; ++k) {
        int *Dk = Dist[k];
        for (int i = i_start; i < i_end; ++i) {
            int dik = Dist[i][k];
            if (dik == INF) continue;
            int *Di = Dist[i];
            for (int j = j_start; j < j_end; ++j) {
                int nd = dik + Dk[j];
                if (nd < Di[j]) Di[j] = nd;
            }
        }
    }
}

void block_FW_fast(int B) {
    int R = ceil_div(n, B);

    for (int r = 0; r < R; ++r) {

        int k_start = r * B;
        int k_end   = (r + 1) * B;
        if (k_end > n) k_end = n;

        update_block(
            r * B, (r + 1) * B > n ? n : (r + 1) * B,
            r * B, (r + 1) * B > n ? n : (r + 1) * B,
            k_start, k_end
        );

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < R; ++i) {

            int i_start = i * B;
            int i_end   = (i + 1) * B;
            if (i_end > n) i_end = n;

            if (i == r) continue;
            update_block(
                i_start, i_end,
                r * B, (r + 1) * B > n ? n : (r + 1) * B,
                k_start, k_end
            );
        }

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < R; ++j) {

            int j_start = j * B;
            int j_end   = (j + 1) * B;
            if (j_end > n) j_end = n;

            if (j == r) continue;

            update_block(
                r * B, (r + 1) * B > n ? n : (r + 1) * B,
                j_start, j_end,
                k_start, k_end
            );
        }

        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < R; ++j) {

                if (i == r || j == r) continue;

                int i_start = i * B;
                int i_end   = (i + 1) * B;
                if (i_end > n) i_end = n;

                int j_start = j * B;
                int j_end   = (j + 1) * B;
                if (j_end > n) j_end = n;

                update_block(
                    i_start, i_end,
                    j_start, j_end,
                    k_start, k_end
                );
            }
        }
    }
}

int main(int argc, char *argv[]) {
    input(argv[1]);
    int B = 512;
    block_FW_fast(B);
    output(argv[2]);
    return 0;
}