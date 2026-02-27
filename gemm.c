#include <stdio.h>
#include <stdlib.h>

// n is the number of rows
#define IDX(i, j, n) i*n +j


void naive_gemm(double *C, double *A, double *B, double alpha, double beta, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;

            for (int k = 0; k < n; k++) {
                sum += A[IDX(i, k, n)] * B[IDX(k, j, n)];
            }

            C[IDX(i, j, n)] = alpha * sum + beta * C[IDX(i, j, n)];
        }
    }
}

void pro_gemm(double *C, double *A, double *B, double alpha, double beta, int n) {
    double *Bt = malloc(sizeof(double) * n * n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Bt[IDX(i, j, n)] = B[IDX(j, i, n)];
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;

            for (int k = 0; k < n; k++) {
                sum += A[IDX(i, k, n)] * Bt[IDX(j, k, n)];
            }

            C[IDX(i, j, n)] = alpha * sum + beta * C[IDX(i, j, n)];
        }
    }

    free(Bt);
}

void pro_gemm_parallel_simd(double *C, double *A, double *B, double alpha, double beta, int n) {
    double *Bt = malloc(sizeof(double) * n * n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Bt[IDX(i, j, n)] = B[IDX(j, i, n)];
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;

            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < n; k++) {
                sum += A[IDX(i, k, n)] * Bt[IDX(j, k, n)];
            }

            C[IDX(i, j, n)] = alpha * sum + beta * C[IDX(i, j, n)];
        }
    }

    free(Bt);
}

void pro_gemm_parallel(double *C, double *A, double *B, double alpha, double beta, int n) {
    double *Bt = malloc(sizeof(double) * n * n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Bt[IDX(i, j, n)] = B[IDX(j, i, n)];
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;

            for (int k = 0; k < n; k++) {
                sum += A[IDX(i, k, n)] * Bt[IDX(j, k, n)];
            }

            C[IDX(i, j, n)] = alpha * sum + beta * C[IDX(i, j, n)];
        }
    }

    free(Bt);
}

void print_matrix(double *M, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2lf ", M[IDX(i, j, n)]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);

    double *A = malloc(n * n * sizeof(double));
    double *B = malloc(n * n * sizeof(double));
    double *C = malloc(n * n * sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[IDX(i, j, n)] = 1.0;
            B[IDX(i, j, n)] = 1.0;
            C[IDX(i, j, n)] = 1.0;
        }
    }

    // print_matrix(A, n);
    // print_matrix(B, n);
    // print_matrix(C, n);

    // naive_gemm(C, A, B, 1.0, 1.0, n);
    // pro_gemm(C, A, B, 1.0, 1.0, n);
    pro_gemm_parallel_simd(C, A, B, 1.0, 1.0, n);
    // pro_gemm_parallel(C, A, B, 1.0, 1.0, n);


    // print_matrix(C, n);

    return 0;
}