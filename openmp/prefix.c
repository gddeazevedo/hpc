#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int *gen_array(int n) {
    int *v = (int *) malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        v[i] = 2 * (i + 1);
    }
    return v;
}

void print_array(int *v, int n) {
    printf("[");

    for (int i = 0; i < n; i++) {
        if (i == n - 1) {
            printf("%d", v[i]);
        } else {
            printf("%d, ", v[i]);
        }
    }

    printf("]\n");
}


void prefix_sum(int *v, int n) {
    for (int i = 1; i < n; i++) {
        v[i] = v[i] + v[i - 1];
    }
}

void prefix_sum_parallel(int *v, int n) {
    int n_threads = 4;
    int chunk_size = (n + n_threads - 1) / n_threads;
    int buffer_size = n_threads - 1;
    int buffer[buffer_size];

    #pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        int start = tid * chunk_size + 1;
        int end   = start + chunk_size - 2; // tid * chunk_size + chunk_size - 1

        if (end >= n) {
            end = n - 1;
        }

        for (int i = start; i <= end; i++) {
            v[i] = v[i] + v[i - 1];
        }

        // todas as threads tem que ter terminado de fazer o prefix sum em seu intervalo
        #pragma omp barrier

        #pragma omp single
        {
            buffer[0] = v[chunk_size - 1];
            for (int i = 1; i < buffer_size; i++) {
                buffer[i] = buffer[i - 1] + v[(i + 1) * chunk_size - 1];
            }
        }
        // tem barrier implicita após o bloco single, a menos que nowait seja usado

        // o buffer tem que estar pronto para que as threads possam prosseguir
        // #pragma omp barrier

        if (tid != 0) {
            for (int i = start - 1; i <= end; i++) {
                v[i] += buffer[tid - 1];
            }
        }
    }
}

int main(int argc, char **argv) {
    int n = atoi(argv[1]);

    int *v1 = gen_array(n);

    // print_array(v1, n);

    printf("Sequential prefix sum:\n");

    double t_start = omp_get_wtime();
    prefix_sum(v1, n);
    double t_end = omp_get_wtime();
    printf("\tTotal time: %.6f\n", t_end - t_start);

    // print_array(v1, n);

    free(v1);
    
    printf("Parallel prefix sum:\n");
    int *v2 = gen_array(n);
    t_start = omp_get_wtime();
    prefix_sum_parallel(v2, n);
    t_end = omp_get_wtime();
    printf("\tTotal time: %.6f\n", t_end - t_start);

    // print_array(v2, n);

    free(v2);

    return 0;
}