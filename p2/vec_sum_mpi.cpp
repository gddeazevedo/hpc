#include <mpi.h>
#include "partition.h"
#include <cstdlib>
#include <cstdio>

double *init_vec(int n) {
    double *v = (double *) malloc(sizeof(double) * n);

    for (int i = 0; i < n; i++) {
        v[i] = i;
    }

    return v;
}

void vecsum(double *u, double *v, double *c, const Partition &p) {
    int chunk_size = p.get_chunk_size();

    for (int i = 0; i < chunk_size; i++) {
        c[i] = u[i] + v[i];
    }
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = atoi(argv[1]);

    Partition p(n, rank, size);

    double *u = nullptr;
    double *v = nullptr;
    double *c = nullptr;

    if (rank == 0) {
        u = init_vec(n);
        v = init_vec(n);
        c = (double *) calloc(n, sizeof(double));
    }

    int chunk_size = p.get_chunk_size();
    double *u_local = (double *) malloc(sizeof(double) * chunk_size);
    double *v_local = (double *) malloc(sizeof(double) * chunk_size);
    double *c_local = (double *) malloc(sizeof(double) * chunk_size);

    auto chunks_sizes  = p.get_chunks_sizes();
    auto chunks_starts = p.get_chunks_starts();

    MPI_Scatterv(
        u, chunks_sizes.get(), chunks_starts.get(),
        MPI_DOUBLE, u_local, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD
    );

    MPI_Scatterv(
        v, chunks_sizes.get(), chunks_starts.get(),
        MPI_DOUBLE, v_local, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD
    );

    if (rank == 0) {
        free(u);
        free(v);
    }

    vecsum(u_local, v_local, c_local, p);

    free(u_local);
    free(v_local);

    MPI_Gatherv(
        c_local, chunk_size, MPI_DOUBLE,
        c, chunks_sizes.get(), chunks_starts.get(),
        MPI_DOUBLE, 0, MPI_COMM_WORLD
    );

    free(c_local);

    if (rank == 0) {
        printf("[");
        for (int i = 0; i < n; i++) {
            if (i == n - 1) {
                printf("%.2f", c[i]);
            } else {
                printf("%.2f, ", c[i]);
            }
        }
        printf("]\n");
        free(c);
    }

    MPI_Finalize();
    return 0;
}