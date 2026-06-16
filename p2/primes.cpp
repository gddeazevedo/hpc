#include <mpi.h>
#include <iostream>
#include <cmath>

#define ROOT  0

#define CHUNK 1000

#define REQ   1
#define WORK  2
#define RES   3
#define KILL -1


bool is_prime(long n) {
    for (long i = 2; i <= ceil(sqrt(n)); i++) {
        if (n % i == 0) {
            return false;
        }
    }

    return true;
}


void orchestrator(const long n, const int size) {
    int workers = size - 1;
    int next_n  = 2; // first prime number

    int total_primes = 0;

    while (workers > 0) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == REQ) {
            char req_buffer;
            MPI_Request reqrecv;
            MPI_Irecv(&req_buffer, 1, MPI_CHAR, status.MPI_SOURCE, REQ, MPI_COMM_WORLD, &reqrecv);

            MPI_Request reqsend;
            long to_send = next_n > n ? KILL : next_n;

            if (to_send == KILL) {
                workers--;
            } else {
                next_n += CHUNK;
            }

            MPI_Isend(&to_send, 1, MPI_LONG, status.MPI_SOURCE, WORK, MPI_COMM_WORLD, &reqsend);
        } else if (status.MPI_TAG == RES) {
            int count;
            int inner_primes[CHUNK];

            MPI_Get_count(&status, MPI_INT, &count);
            MPI_Recv(inner_primes, count, MPI_INT, status.MPI_SOURCE, RES, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            total_primes += count;
        }
    }

    printf("De 2 a %ld foram achados %d números primos\n", n, total_primes);
}

void worker(long n) {
    char buffer;
    long next_n;

    while (true) {
        MPI_Send(&buffer, 1, MPI_CHAR, ROOT, REQ, MPI_COMM_WORLD);
        MPI_Recv(&next_n, 1, MPI_LONG, ROOT, WORK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        if (next_n == KILL) {
            break;
        }

        int counter = 0;
        int primes[CHUNK];

        for (int i = next_n; i < n && i < next_n + CHUNK; i++) {
            if (is_prime(i)) {
                primes[counter] = i;
                counter++;
            }
        }

        MPI_Send(primes, counter, MPI_INT, ROOT, RES, MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = atol(argv[1]);

    if (rank == ROOT) {
        orchestrator(n, size);
    } else {
        worker(n);
    }

    MPI_Finalize();

    return 0;
}
