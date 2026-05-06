#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int size;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int sendbuf[] = {1, 2};
    int recvbuf[] = {0, 0};

    MPI_Reduce(
        sendbuf,
        recvbuf,
        2,
        MPI_INT,
        MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("revc = [%d, %d]", recvbuf[0], recvbuf[1]);
    }

    MPI_Finalize();
    return 0;
}