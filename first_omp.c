#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>


int main() {

    int total = 0;

    auto t0 = omp_get_wtime();

    //#pragma omp parallel for num_threads(20) reduction(+:total)
    for (int i = 0; i < 6000; i++) {
        total++;
    }

    auto t1 = omp_get_wtime();

    printf("Total: %d\n", total);
    printf("Time (Sequential): %f seconds\n", t1 - t0);

    total = 0;
    t0 = omp_get_wtime();

    #pragma omp parallel for num_threads(20) reduction(+:total)
    for (int i = 0; i < 6000; i++) {
        total++;
    }

    t1 = omp_get_wtime();

    printf("Total: %d\n", total);
    printf("Time (Parallel): %f seconds\n", t1 - t0);

    return 0;
}