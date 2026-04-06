#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


double randr(double a, double b) {
    return ((rand() % 10000000) / 10000000.)*(b-a)+a;
}

double *gen_vector(int n) {
    double *v = (double *) malloc(sizeof(double) * n);

    for (int i = 0; i < n; i++) {
        v[i] = randr(1.0, 10.0);
    }

    return v;
}

void print_vector(double *v, int n) {
    printf("[");

    for (int i = 0; i < n; i++) {
        if (i != n - 1) {
            printf("%2f, ", v[i]);
            continue;
        }

        printf("%2f]\n", v[i]);
    }
}

double dot(double *u, double *v, int n) {
    double s = 0.0;

    for (int i = 0; i < n; i++) {
        s += u[i] * v[i]; 
    }

    return s;
}

double dot_parallel(double *u, double *v, int n) {
    double s = 0.0;

    #pragma omp parallel for reduction(+:s)
    for (int i = 0; i < n; i++) {
        s += u[i] * v[i]; 
    }

    return s;
}

int main(int argc, char **argv) {
    int n = atoi(argv[1]);

    double *u = gen_vector(n);
    double *v = gen_vector(n);

    // print_vector(v, n);
    // print_vector(u, n);

    double t_start, t_end;
    double uv_seq;

    t_start = omp_get_wtime();
    uv_seq = dot(u, v, n);
    t_end = omp_get_wtime();
    printf("Sequential: %.2fs\n", t_end - t_start);

    t_start = omp_get_wtime();
    double uv_par = dot(u, v, n);
    t_end = omp_get_wtime();
    printf("Parallel: %.2fs\n", t_end - t_start);

    printf("Error: %2f\n", uv_par - uv_seq);


    return 0;
}