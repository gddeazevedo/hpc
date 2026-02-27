#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define IDX(i, j) i*n + j

typedef void (*gemm)(double *C, const double *A, const double *B, const double alpha, const double beta, const uint32_t n);

static inline double wtime() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

double *tranpose(const double *M, const uint32_t n) {
    double *Mt = malloc(sizeof(double) * n * n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Mt[IDX(i, j)] = M[IDX(j, i)];
        }
    }

    return Mt;
}

void naive_gemm(double *C, const double *A, const double *B, const double alpha, const double beta, const uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < n; j++) {
            double sum = 0.0;

            for (uint32_t k = 0; k < n; k++) {
                sum += A[IDX(i, k)] * B[IDX(k, j)];
            }

            C[IDX(i, j)] = alpha * sum + beta * C[IDX(i, j)];
        }
    }
}

void pro_gemm(double *C, const double *A, const double *B, const double alpha, const double beta, const uint32_t n) {
    double *Bt = tranpose(B, n);

    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < n; j++) {
            double sum = 0.0;

            for (uint32_t k = 0; k < n; k++) {
                sum += A[IDX(i, k)] * Bt[IDX(j, k)];
            }

            C[IDX(i, j)] = alpha * sum + beta * C[IDX(i, j)];
        }
    }

    free(Bt);
}

void pro_gemm_parallel(double *C, const double *A, const double *B, const double alpha, const double beta, const uint32_t n) {
    double *Bt = tranpose(B, n);

    #pragma omp parallel for
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < n; j++) {
            double sum = 0.0;

            for (uint32_t k = 0; k < n; k++) {
                sum += A[IDX(i, k)] * Bt[IDX(j, k)];
            }

            C[IDX(i, j)] = alpha * sum + beta * C[IDX(i, j)];
        }
    }

    free(Bt);
}

void pro_gemm_parallel_simd(double *C, const double *A, const double *B, const double alpha, const double beta, const uint32_t n) {
    double *Bt = tranpose(B, n);

    #pragma omp parallel for
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < n; j++) {
            double sum = 0.0;

            #pragma omp simd reduction(+:sum)
            for (uint32_t k = 0; k < n; k++) {
                sum += A[IDX(i, k)] * Bt[IDX(j, k)];
            }

            C[IDX(i, j)] = alpha * sum + beta * C[IDX(i, j)];
        }
    }

    free(Bt);
}

void run_gemm_benchmark(uint32_t n, gemm func, char *label) {
    double *A = malloc(n * n * sizeof(double));
    double *B = malloc(n * n * sizeof(double));
    double *C = malloc(n * n * sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[IDX(i, j)] = 1.0;
            B[IDX(i, j)] = 1.0;
            C[IDX(i, j)] = 1.0;
        }
    }

    double start = wtime();
    func(C, A, B, 1.0, 1.0, n);
    double end = wtime();
    printf("%s: %lf seconds\n", label, end - start);

    free(A);
    free(B);
    free(C);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);

    run_gemm_benchmark(n, naive_gemm, "Naive GEMM");
    run_gemm_benchmark(n, pro_gemm, "Pro GEMM");
    run_gemm_benchmark(n, pro_gemm_parallel, "Pro GEMM Parallel");
    run_gemm_benchmark(n, pro_gemm_parallel_simd, "Pro GEMM Parallel SIMD");

    return 0;
}
