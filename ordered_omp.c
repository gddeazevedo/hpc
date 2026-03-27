#include <stdio.h>
#include <omp.h>

void ordered_example(int lb, int ub, int stride) {
    int i;
    // The 'ordered' clause on the 'parallel for' directive
    #pragma omp parallel for ordered num_threads(4)
    for (i = lb; i < ub; i += stride) {
        // This printf might run in any order
        printf("Thread %d working on iteration %d in no-order\n", omp_get_thread_num(), i); 
        
        // The '#pragma omp ordered' directive ensures this block runs sequentially
        #pragma omp ordered 
        printf("Thread %d is executing iteration %d in order\n", omp_get_thread_num(), i); 
    }
}

int main() {
    ordered_example(0, 10, 1);
    return 0;
}
