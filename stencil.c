#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define IDX(i, j) (i) * n + (j)


void compute_stencil(const double *A, double *B, int n) {
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            B[IDX(i, j)] = 0.25 * (A[IDX(i - 1, j)] + A[IDX(i + 1, j)] + A[IDX(i, j - 1)] + A[IDX(i, j + 1)]);
        }
    }
}

void compute_stencil_parallel_buffer(const double *A, double *B, int n) {
    #pragma omp parallel for
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            B[IDX(i, j)] = 0.25 * (A[IDX(i - 1, j)] + A[IDX(i + 1, j)] + A[IDX(i, j - 1)] + A[IDX(i, j + 1)]);
        }
    }
}

void compute_stencil_parallel_red_black_ordering_v1(double *A, int n) {
    // Problema, forkjoin. Cria uma região paralela, depois junta todas as threads numa só,
    // por fim cria threads de novo
    #pragma omp parallel for
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            if ((i + j) % 2 == 0) {
                A[IDX(i, j)] = 0.25 * (A[IDX(i - 1, j)] + A[IDX(i + 1, j)] + A[IDX(i, j - 1)] + A[IDX(i, j + 1)]);
            }
        }
    }

    #pragma omp parallel for
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            if ((i + j) % 2 == 1) {
                A[IDX(i, j)] = 0.25 * (A[IDX(i - 1, j)] + A[IDX(i + 1, j)] + A[IDX(i, j - 1)] + A[IDX(i, j + 1)]);
            }
        }
    }
}

void compute_stencil_parallel_red_black_ordering_v2(double *A, int n) {
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                if ((i + j) % 2 == 0) {
                    A[IDX(i, j)] = 0.25 * (A[IDX(i - 1, j)] + A[IDX(i + 1, j)] + A[IDX(i, j - 1)] + A[IDX(i, j + 1)]);
                }
            }
        }

        #pragma omp barrier

        #pragma omp for
        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                if ((i + j) % 2 == 1) {
                    A[IDX(i, j)] = 0.25 * (A[IDX(i - 1, j)] + A[IDX(i + 1, j)] + A[IDX(i, j - 1)] + A[IDX(i, j + 1)]);
                }
            }
        }
    }
}

void initialize(double *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[IDX(i, j)] = (double) (i * n + j);
        }
    }
}

void clear(double *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[IDX(i, j)] = 0.0;
        }
    }
}

int main(int argc, char **argv) {
    int n = atoi(argv[1]);

    double *A = (double *) malloc(sizeof(double) * n * n);
    double *B = (double *) malloc(sizeof(double) * n * n);

    double t_start;
    double t_end;

    t_start = omp_get_wtime();
    compute_stencil(A, B, n);
    t_end = omp_get_wtime();
    printf("Sequential: %fs\n", t_end - t_start);

    initialize(A, n);
    initialize(A, n);

    t_start = omp_get_wtime();
    compute_stencil_parallel_red_black_ordering_v1(A, n);
    t_end = omp_get_wtime();
    printf("Parallel (Red-Black Ordering): %fs\n", t_end - t_start);


    t_start = omp_get_wtime();
    compute_stencil_parallel_red_black_ordering_v2(A, n);
    t_end = omp_get_wtime();
    printf("Parallel (Red-Black Ordering): %fs\n", t_end - t_start);


    t_start = omp_get_wtime();
    compute_stencil_parallel_buffer(A, B, n);
    t_end = omp_get_wtime();
    printf("Parallel (Buffer): %fs\n", t_end - t_start);

    free(A);
    free(B);

    return 0;
}
