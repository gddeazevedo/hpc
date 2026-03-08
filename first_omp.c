#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>


int main() {
    printf("SEQUENCIAL\n");

    omp_set_num_threads(10);

    printf("Qtd threads: %d\n", omp_get_num_threads());

    int x = 10;
    #pragma omp parallel private(x)
    {
        int tid = omp_get_thread_num();
        int n_threads = omp_get_num_threads();
        
        // if (tid == 0) {
        //     sleep(10);
        // }
        
        printf("PARALLEL %d / %d\n", tid, n_threads - 1);
        printf("TID: %d, X: %d\n", tid, x);
    }

    printf("x = %d\n", x);
    printf("END\n");

    return 0;
}