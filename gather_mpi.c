#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size); // qtd de processos


    char send = 'A' + rank;

    char recv[26];

    MPI_Gather(
        &send, 1, MPI_CHAR,
        recv,  1, MPI_CHAR,
        0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            printf("%c\n", recv[i]);
        }
    }

    MPI_Finalize();
    return 0;
}