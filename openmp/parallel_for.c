#include <stdio.h>
#include <omp.h>

int main() {
    printf("PARALLEL REGION\n");

    #pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();

        for (int i = 0; i < 5; i++) {
            printf("THREAD: %d | ITERATION: %d\n", tid, i);
        }
    }

    printf("\n\nPARALLEL FOR 1\n");

    #pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();

        #pragma omp for
        for (int i = 0; i < 5; i++) {
            printf("THREAD: %d | ITERATION: %d\n", tid, i);
        }
    }

    printf("\n\nPARALLEL FOR 2\n");

    #pragma omp parallel for num_threads(2)
    for (int i = 0; i < 5; i++) {
        int tid = omp_get_thread_num();
        printf("THREAD: %d | ITERATION: %d\n", tid, i);
    }

    return 0;
}