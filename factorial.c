#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


unsigned int factorial(unsigned int n) {
    if (n == 0) {
        return 1;
    }

    return n * factorial(n - 1);
}

unsigned int factorial_parallel(unsigned int n) {
    if (n == 0) {
        return 1;
    }

    unsigned int f;

    #pragma omp task shared(f)
    f = factorial(n - 1);

    #pragma omp taskwait
    return n * f;
}

int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    double t0, t1;

    t0 = omp_get_wtime();
    int f1 = factorial_parallel(n);
    t1 = omp_get_wtime();

    printf("%d! = %d\n", n, f1);
    printf("Sequential: %fs\n", t1 - t0);

    return 0;
}