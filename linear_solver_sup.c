#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define IDX(i, j) i * n + j
#define M(i, j) A[IDX(i, j)]


void gauss_elimination(double *A, double *b, int n) {
    for (int k = 0; k < n - 1; k++) {
        for (int i = k + 1; i < n; i++) {
            double pivot = A[IDX(i, k)] / A[IDX(k, k)];
            b[i] -= b[k] * pivot;

            for (int j = k; j < n; j++) {
                M(i, j) -= M(k, j) * pivot;
            }
        }
    }
}

void gauss_elimination_parallel(double *A, double *b, int n) {
    for (int k = 0; k < n - 1; k++) {

        #pragma omp parallel for num_threads(2)
        for (int i = k + 1; i < n; i++) {
            double pivot = M(i, k) / M(k, k);
            b[i] -= b[k] * pivot;

            for (int j = k; j < n; j++) {
                M(i, j) -= M(k, j) * pivot;
            }
        }
    }
}

double *solve_sup(const double *A, const double *b, const int n) {
    double *x = (double *) malloc(sizeof(double) * n);

    for (int i = n - 1; i >= 0; i--) {
        double s = 0.0;

        for (int j = i + 1; j < n; j++) {
            s += M(i, j) * x[j];
        }

        x[i] = (b[i] - s) / M(i, i);
    }

    return x;
}

void print_vector(const double *v, int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        if (i < n - 1) {
            printf("%.2f, ", v[i]);
        } else {
            printf("%.2f ", v[i]);
        }
    }
    printf("]\n");
}


double *solve_sup_parallel(const double *A, const double *b, const int n) {
    double *x = (double *) malloc(sizeof(double) * n);

    for (int i = n - 1; i >= 0; i--) {
        double s = 0.0;

        #pragma omp parallel for reduction(+:s) num_threads(2)
        for (int j = i + 1; j < n; j++) {
            s += M(i, j) * x[j];
        }

        x[i] = (b[i] - s) / M(i, i);
    }

    return x;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);

    double *A = (double *) malloc(sizeof(double) * n * n);
    double *b = (double *) malloc(sizeof(double) * n);

    for (int i = 0; i < n; i++) {
        b[i] = 1.0;
        for (int j = 0; j < n; j++) {
            M(i, j) = (i <= j) ? 1.0 : 0.0; // Example: upper triangular matrix
        }
    }

    // Ax = b
    double t0 = omp_get_wtime();
    gauss_elimination(A, b, n);
    double *x = solve_sup(A, b, n);
    double t1 = omp_get_wtime();

    printf("Time (Sequential): %f seconds\n", t1 - t0);
    print_vector(x, n);    
    free(x);

    t0 = omp_get_wtime();
    gauss_elimination_parallel(A, b, n);
    x = solve_sup_parallel(A, b, n);
    t1 = omp_get_wtime();

    printf("Time (Parallel): %f seconds\n", t1 - t0);

    print_vector(x, n);

    free(A);
    free(b);
    free(x);

    return 0;
}