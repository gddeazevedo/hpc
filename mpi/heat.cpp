#include <iostream>
#include <cmath>
#include "partition.h"
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    constexpr int n = 2;
    Partition p(n, size, rank);

    double **T;
    double **T_prev;
    double **T_next;

    if (rank == 0) {
        T = (double **) malloc(sizeof(double *) * n);

        for (int i = 0; i < n; i++) {
            T[i] = (double *) malloc(sizeof(double) * n);
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || i == n - 1 || j == 0 || j == n - 1) {
                    T[i][j] = 100.0;
                } else {
                    T[i][j] = 0.0;
                }
            }
        }
    }

    T_prev = (double **) malloc(sizeof(double *) * (p.get_chunk_size() + 2)); // 2 linhas ghost, uma em cima e uma embaixo;
    for (int i = 0; i < p.get_chunk_size() + 2; i++) {
        T_prev[i] = (double *) malloc(sizeof(double) * n);
    }

    T_next = (double **) malloc(sizeof(double *) * (p.get_chunk_size() + 2));
    for (int i = 0; i < p.get_chunk_size() + 2; i++) {
        T_next[i] = (double *) malloc(sizeof(double) * n);
    }

    if (rank == 0) {
        int *sendcounts    = p.get_chunks_sizes();
        int *displacements = p.get_chunks_starts();

        for (int i = 1; i < size; i++) {
            int disp = displacements[i];
            for (int j = 0; j < sendcounts[i]; j++) {
                MPI_Send(T[disp + j], n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
        }

        delete[] sendcounts;
        delete[] displacements;
    }

    for (int i = 0; i < p.get_chunk_size() + 2; i++) {
        MPI_Recv(T_prev[i + 1], n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for (int niter = 0; niter < 1; niter++) {
        int up   = rank - 1;
        int down = rank + 1;

        if (up < 1) {
            up = MPI_PROC_NULL;
        }

        if (down > size - 1) {
            down = MPI_PROC_NULL;
        }

        MPI_Sendrecv(
            T_prev[p.get_chunk_size()], n, MPI_DOUBLE, down, 0,
            T_prev[p.get_chunk_size() + 1], n, MPI_DOUBLE, down, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        MPI_Sendrecv(
            T_prev[1], n, MPI_DOUBLE, up, 0,
            T_prev[0], n, MPI_DOUBLE, up, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        int init;

        if (rank == 0) {
            init = 2;
        } else {
            init = 1;
        }

        for (int i = init; i < p.get_chunk_size() + 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                T_next[i][j] = 0.25 * (
                    T_prev[i - 1][j] + T_prev[i + 1][j] + T_prev[i][j - 1] + T_prev[i][j + 1]
                );
            }
        }

        for (int i = init; i < p.get_chunk_size() + 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                T_prev[i][j] = T_next[i][j];
            }
        }

        // if (rank == 0) {
        //     for (int i = 0; i < p.get_chunk_size() + 2; i++) {
        //         for (int j = 0; j < n; j++) {
        //             std::cout << "[" << rank << "] " << T_prev[i + 1][j] << " ";
        //         }
        //         std::cout << std::endl;
        //     }
        // }
    }

    MPI_Finalize();
    return 0;
}
