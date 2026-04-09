#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define IDX(i, j) (i) * n + (j)

typedef struct {
    double *L;
    double *U;
} lu_decomp_matrices;

double randr(double a, double b) {
    return ((rand() % 10000000) / 10000000.)*(b-a)+a;
}

double *transpose(const double *M, const int n) {
    double *Mt = malloc(sizeof(double) * n * n);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Mt[IDX(i, j)] = M[IDX(j, i)];
        }
    }

    return Mt;
}

double *solve_sup(const double *A, const double *b, const int n) {
    double *x = (double *) malloc(sizeof(double) * n);

    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;

        #pragma omp parallel for reduction(+:sum)
        for (int j = i + 1; j < n; j++) {
            sum += A[IDX(i, j)] * x[j];
        }

        x[i] = (b[i] - sum) / A[IDX(i, i)];
    }

    return x;
}

double *solve_inf(const double *A, const double *b, const int n) {
    double *x = (double *) malloc(sizeof(double) * n);

    for (int i = 0; i < n; i++) {
        double sum = 0.0;

        #pragma omp parallel for reduction(+:sum)
        for (int j = 0; j < i; j++) {
            sum += A[IDX(i, j)] * x[j];
        }

        x[i] = (b[i] - sum) / A[IDX(i, i)];
    }

    return x;
}

lu_decomp_matrices *lu_decomp(const double *A, int n) {
    double *L  = (double *) calloc(n * n, sizeof(double));
    double *Ut = (double *) calloc(n * n, sizeof(double));

    for (int i = 0; i < n; i++) {
        L[IDX(i, i)] = 1.0;

        for (int j = 0; j < n; j++) {
            double sum = 0.0;

            if (i <= j) {
                #pragma omp parallel for reduction(+: sum)
                for (int k = 0; k < i; k++) {
                    // uso de transposta para melhorar a localidade de cache
                    // acesso são feitos de forma coalescente
                    sum += L[IDX(i, k)] * Ut[IDX(j, k)];
                }

                Ut[IDX(j, i)] = A[IDX(i, j)] - sum;
            } else {
                #pragma omp parallel for reduction(+: sum)
                for (int k = 0; k < j; k++) {
                    sum += L[IDX(i, k)] * Ut[IDX(j, k)];
                }

                L[IDX(i, j)] = (A[IDX(i, j)] - sum) / Ut[IDX(j, j)];
            }
        }
    }

    lu_decomp_matrices *matrices = (lu_decomp_matrices *) malloc(sizeof(lu_decomp_matrices));
    matrices->L = L;
    matrices->U = transpose(Ut, n);
    free(Ut);

    return matrices;
}

double *lu_solver(const double *A, const double *b, int n) {
    lu_decomp_matrices *matrices = lu_decomp(A, n);
    double *L = matrices->L;
    double *U = matrices->U;

    free(matrices);

    double *y = solve_inf(L, b, n);
    double *x = solve_sup(U, y, n);

    free(L);
    free(U);
    free(y);

    return x;
}

int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    double t_start, t_end;

    double *A = (double *) malloc(sizeof(double) * n * n);
    double *b = (double *) malloc(sizeof(double) * n);

    for (int i = 0; i < n; i++) {
        b[i] = randr(0.0, 1.0);

        for (int j = 0; j < n; j++) {
            A[IDX(i, j)] = randr(0.0, 1.0);
        }
    }

    t_start = omp_get_wtime();
    double *x = lu_solver(A, b, n);
    t_end = omp_get_wtime();

    free(A);
    free(b);

    printf("Time: %f\n", t_end - t_start);

    return 0;
}
