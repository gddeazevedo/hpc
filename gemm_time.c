#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#define IDX(i, j) i * n + j
#define BS 64

typedef void (*gemm_func_t)(double *C, const double *A, const double *B, const double alpha, const double beta, const uint32_t n);


void print_matrix(double *M, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2lf ", M[IDX(i, j)]);
        }
        printf("\n");
    }
    printf("\n");
}

double *transpose(const double *M, const uint32_t n) {
    double *Mt = malloc(sizeof(double) * n * n);

    #pragma omp parallel for
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

void naive_gemm_parallel(double *C, const double *A, const double *B, const double alpha, const double beta, const uint32_t n) {
    #pragma omp parallel for
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
    double *Bt = transpose(B, n);

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

void pro_gemm_parallel(double *C, const double *A, const double *B, const double alpha, const double beta, const uint32_t n) {
    double *Bt = transpose(B, n);

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

void pro_gemm_simd(double *C, const double *A, const double *B, const double alpha, const double beta, const uint32_t n) {
    double *Bt = transpose(B, n);

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

void pro_gemm_parallel_simd(double *C, const double *A, const double *B, const double alpha, const double beta, const uint32_t n) {
    double *Bt = transpose(B, n);

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

void pro_gemm_tiles(double *C, const double *A, const double *B, const double alpha, const double beta, const uint32_t n) {
    double *Bt = transpose(B, n);

    #pragma omp parallel for
    for (uint32_t i = 0; i < n * n; i++) {
        C[i] = beta * C[i];
    }

    #pragma omp parallel for
    for (uint32_t i = 0; i < n; i += BS) {
        for (uint32_t j = 0; j < n; j += BS) {
            for (uint32_t k = 0; k < n; k += BS) {

                for (uint32_t ii = i; ii < i + BS && ii < n; ii++) {
                    for (uint32_t jj = j; jj < j + BS && jj < n; jj++) {

                        double sum = 0.0;

                        for (uint32_t kk = k; kk < k + BS && kk < n; kk++) {
                            sum += A[IDX(ii, kk)] * Bt[IDX(jj, kk)];
                        }

                        C[IDX(ii, jj)] += alpha * sum;
                    }
                }
            }
        }
    }

    free(Bt);
}


// TODO: Armazenar B em blocos para melhorar a localidade de cache
// void pro_gemm_tiles_blocked(double *C, const double *A, const double *B, const double alpha, const double beta, const uint32_t n) {
//     for (uint32_t i = 0; i < n * n; i++) {
//         C[i] = beta * C[i];
//     }

//     for (uint32_t i = 0; i < n; i += BS) {
//         for (uint32_t j = 0; j < n; j += BS) {
//             for (uint32_t k = 0; k < n; k += BS) {

//                 for (uint32_t ii = i; ii < i + BS && ii < n; ii++) {
//                     for (uint32_t jj = j; jj < j + BS && jj < n; jj++) {

//                         double sum = 0.0;

//                         for (uint32_t kk = k; kk < k + BS && kk < n; kk++) {
//                             sum += A[IDX(ii, kk)] * B[IDX(jj, kk)];
//                         }

//                         C[IDX(ii, jj)] += alpha * sum;
//                     }
//                 }
//             }
//         }
//     }
// }


void run_gemm_benchmark(uint32_t n, gemm_func_t gemm, char *label) {
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

    gemm(C, A, B, 2.0, 2.0, n); // cache warmup

    double start = omp_get_wtime();
    gemm(C, A, B, 2.0, 2.0, n);
    double end = omp_get_wtime();

    // print_matrix(C, n);

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
    run_gemm_benchmark(n, naive_gemm_parallel, "Naive GEMM Parallel");
    run_gemm_benchmark(n, pro_gemm, "Pro GEMM");
    run_gemm_benchmark(n, pro_gemm_parallel, "Pro GEMM Parallel");
    run_gemm_benchmark(n, pro_gemm_simd, "Pro GEMM SIMD");
    run_gemm_benchmark(n, pro_gemm_parallel_simd, "Pro GEMM Parallel SIMD");
    run_gemm_benchmark(n, pro_gemm_tiles, "Pro GEMM Tiles");

    return 0;
}
