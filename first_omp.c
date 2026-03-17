#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>


int main() {

    int N = 10;

    #pragma omp parallel for num_threads(2)
    for (int i = N - 1; i >= 0; i--) {
        int tid = omp_get_thread_num();
        printf("Thread %d: i = %d\n", tid, i);
    }

    return 0;
}