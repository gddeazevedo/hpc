#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void print_vec(int *v, int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        if (i < n - 1) {
            printf("%d, ", v[i]);
        } else {
            printf("%d]\n", v[i]);
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int n = 10;
    int size;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *v = (int *) malloc(sizeof(int) * n);

    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            v[i] = i;
        }
    }

    MPI_Bcast(v, n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        printf("Rank %d: ", rank);
        print_vec(v, n);
    }

    MPI_Finalize();
    return 0;
}