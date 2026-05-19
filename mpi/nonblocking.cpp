#include <mpi.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    int size;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Get_processor_name(processor_name, &name_len);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::cout << "[" << rank << "] Hello from processor " << processor_name << ", rank " << rank << " out of " << size << " processors" << std::endl;

    if (rank == 1) {
        MPI_Request request;
        sleep(4); // Simulate some work
        char message[] = "Hello from rank 1!";

        MPI_Isend(
            message,
            sizeof(message),
            MPI_CHAR,
            0, // destination rank
            0, // tag
            MPI_COMM_WORLD,
            &request
        );
    } else if (rank == 0) {
        MPI_Request request;
        char buffer[100];

        MPI_Irecv(
            buffer,
            sizeof(buffer),
            MPI_CHAR,
            1, // source rank
            0, // tag
            MPI_COMM_WORLD,
            &request
        );

        // MPI_Wait(&request, MPI_STATUS_IGNORE);

        int flag = 0;
        double t0 = MPI_Wtime();
        double timeout = 5.0;

        while (!flag) {
            MPI_Test(&request, &flag, MPI_STATUS_IGNORE);

            double t1 = MPI_Wtime();

            if (t1 - t0 > timeout) {
                std::cout << "TIMEOUT: No message received within " << timeout << " seconds." << std::endl;
                return 1;
            }

            std::cout << "[" << rank << "] Waiting for message..." << std::endl;

            usleep(100000); // Sleep for 100ms to avoid busy waiting
        }

        std::cout << "[" << rank << "] Received message: " << buffer << std::endl;
    }

    MPI_Finalize();

    return 0;
}
