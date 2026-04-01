#include <stdio.h>
#include <omp.h>

int main() {
    #pragma omp parallel num_threads(2)
    {
        #pragma omp for nowait
        for (int i = 0; i < 4; i++) {
            printf("TASK 0: Thread %d is processing iteration %d\n", omp_get_thread_num(), i);
        }

        #pragma omp single
        printf("THERE IS A BARRIER\n");

        #pragma omp for
        for (int j = 0; j < 4; j++) {
            printf("TASK 1: Thread %d is processing iteration %d\n", omp_get_thread_num(), j);
        }
    }

    return 0;
}