#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int size;
    int rank; // qtd de processos

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        printf("RANK: %d\n", rank);
        sleep(2);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    printf("Depois da barreira: %d\n", rank);

    MPI_Finalize();
    return 0;
}