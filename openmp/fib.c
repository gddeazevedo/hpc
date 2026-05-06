#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int fib(int n) {
    if (n == 1 || n == 2) {
        return 1;
    }

    return fib(n - 1) + fib(n - 2);
}

int fib_parallel(int n) {
    if (n == 1 || n == 2) {
        return 1;
    }

    int i, j;

    #pragma omp task shared(i) // precisa do shared pois a task está escrevendo numa variável que não é ponteiro
    i = fib_parallel(n - 1);

    #pragma omp task shared(j)
    j = fib_parallel(n - 2);

    #pragma omp taskwait

    return i + j;
}

int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    double t_ini;
    double t_fim;

    t_ini = omp_get_wtime();
    int f = fib(n);
    t_fim = omp_get_wtime();
    printf("Fib Sequential: %.2fs\n", t_fim - t_ini);

    t_ini = omp_get_wtime();
    #pragma omp parallel
    #pragma omp single
    {
        int f1 = fib_parallel(n);
    }
    t_fim = omp_get_wtime();
    printf("Fib Parallel: %.2fs\n", t_fim - t_ini);

    return 0;
}
