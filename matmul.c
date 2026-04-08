#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double randr(double a, double b) {
    return ((rand() % 10000000) / 10000000.)*(b-a)+a;
}

double **new_matrix(int n) {
    double **matrix = (double **) malloc(sizeof(double *) * n);

    for (int i = 0; i < n; i++) {
        matrix[i] = (double *) malloc(sizeof(double) * n);
    }

    return matrix;
}

double **gen_matrix(int n) {
    double **matrix = (double **) malloc(sizeof(double *) * n);

    for (int i = 0; i < n; i++) {
        matrix[i] = (double *) malloc(sizeof(double) * n);

        for (int j = 0; j < n; j++) {
            matrix[i][j] = randr(1.0, 10.0);
        }
    }

    return matrix;
}

void freemat(double **M, int n) {
    for (int i = 0; i < n; i++) {
        free(M[i]);
    }

    free(M);
}

void matmul(double **C, double **A, double **B, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;

            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }

            C[i][j] = sum;
        }
    }
}

void matmul_v2(double **C, double **A, double **B, int n) {
    double ** Bt = newmat(n);

    for( int i = 0; i < n; i++ ) {
        for( int j = 0; j < n; j++ ) {
            Bt[i][j] = B[j][i];
        }
    } 

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;

            for (int k = 0; k < n; k++) {
                sum += A[i][k] * Bt[j][k];
            }

            C[i][j] = sum;
        }
    }
}

int main(int arc, char **argv) {
    int n = atoi(argv[1]);

    double **A = gen_matrix(n);
    double **B = gen_matrix(n);
    double **C = gen_matrix(n);

    double t0, t1;

    t0 = omp_get_wtime();
    matmul(C, A, B, n);
    t1 = omp_get_wtime();

    printf("Sequential (no coalescence): %.2fs\n", t1 - t0);


    return 0;
}
