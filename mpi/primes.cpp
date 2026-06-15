#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define ROOT_RANK 0
#define CHUNK    1000
#define TAG_REQ  1
#define TAG_RES  2
#define TAG_WORK 3
#define KILL    -1

bool is_prime(long n) {
    for (long i = 2; i <= ceil(sqrt(n)); i++) {
        if (n % i == 0) {
            return false;
        }
    }

    return true;
}

void master(long n, int size) {
    long prox_num = 2;
    int active_workers = size - 1;

    int total_primes = 0;

    while (active_workers > 0) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TAG_REQ) {
            char buffer;
            long to_send = (prox_num > n) ? KILL : prox_num;
            MPI_Request request_recv;

            MPI_Irecv(&buffer, 1, MPI_CHAR, status.MPI_SOURCE, TAG_REQ,
                MPI_COMM_WORLD, &request_recv); // recebe pedido do worker

            if (to_send == KILL) {
                active_workers--;
            } else {
                prox_num += CHUNK;
            }

            MPI_Request request_send;
            MPI_Isend(&to_send, 1, MPI_LONG, status.MPI_SOURCE, TAG_WORK,
                MPI_COMM_WORLD, &request_send); // envia trabalho para o worker
        } else if (status.MPI_TAG == TAG_RES) {
            long primes[CHUNK];
            int count;
            MPI_Get_count(&status, MPI_LONG, &count);
            MPI_Recv(primes, count, MPI_LONG, status.MPI_SOURCE, TAG_RES,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            total_primes += count;

            printf("Worker %d found %d primes.\n", status.MPI_SOURCE, count);
        }
    }

    printf("Total primes: %ld\n", total_primes);
}

void worker(long n) {
    long prox_num = 0;
    char buffer;

    while (true) {
        MPI_Send(&buffer, 1, MPI_CHAR, 0, TAG_REQ, MPI_COMM_WORLD);
        MPI_Recv(&prox_num, 1, MPI_LONG, 0, TAG_WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (prox_num == KILL) {
            break;
        }

        long primes[CHUNK];
        int k = 0;

        for (long p = prox_num; p < prox_num + CHUNK && p <= n; p++) {
            if (is_prime(p)) {
                primes[k] = p;
                k++;
            }
        }

        MPI_Send(primes, k, MPI_LONG, 0, TAG_RES, MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int size;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long n = atoi(argv[1]);

    if (rank == ROOT_RANK) {
        master(n, size);
    } else {
        worker(n);
    }

    MPI_Finalize();
    return 0;
}
