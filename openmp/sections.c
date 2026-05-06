#include <stdio.h>
#include <omp.h>

void foo(int tid) {
    printf("FOO: %d\n", tid);
}

void bar(int tid) {
    printf("BAR: %d\n", tid);
}

int main() {
    #pragma omp parallel num_threads(3)
    {

        #pragma omp sections
        {
            #pragma omp section
            foo(omp_get_thread_num());
            
            #pragma omp section
            bar(omp_get_thread_num());
            
            #pragma omp section
            for (int i = 0; i < 2; i++) {
                printf("FOR: %d\n", omp_get_thread_num());
            }
            
            // printf("OLA MUNDO FROM %d\n", omp_get_thread_num());
        } // barrier implicito no final de sections
    
        #pragma omp single
        printf("BARRIER\n");

        #pragma omp for
        for (int i = 0; i < 4; i++) {
            printf("FOR: %d is processing iteration %d\n", omp_get_thread_num(), i);
        }
    }

    return 0;
}