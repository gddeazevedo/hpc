#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char send[size];

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
           send[i] = 'A' + i;
        }
    }

    char recv;

    MPI_Scatter(send, 1, MPI_CHAR,
               &recv, 1, MPI_CHAR, 0,
               MPI_COMM_WORLD);

    printf("[%d] %c\n", rank, recv);           

    MPI_Finalize();
    return 0;
}