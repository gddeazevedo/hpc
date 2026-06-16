#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include "partition.h"

double dot(double *u, double *v, const Partition &p) {
    const int chunk_size = p.get_chunk_size();

    double local_sum = 0.0;
    
    for (int i = 0; i < chunk_size; i++) {
        local_sum += u[i] * v[i];
    }
    
    double sum = 0.0;

    constexpr int count = 1;
    constexpr int dest  = 0;

    MPI_Reduce(&local_sum, &sum, count, MPI_DOUBLE, MPI_SUM, dest, MPI_COMM_WORLD);

    return sum;
}

double *init_vec(int n) {
    double *v = (double *) malloc(sizeof(double) * n);

    for (int i = 0; i < n; i++) {
        v[i] = i;
    }

    return v;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int n = atoi(argv[1]);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Partition p(n, rank, size);

    double *u = nullptr;
    double *v = nullptr;

    double *u_local, *v_local;

    if (rank == 0) {
        u = init_vec(n);
        v = init_vec(n);
    }

    int chunk_size = p.get_chunk_size();
    u_local = (double *) malloc(sizeof(double) * chunk_size);
    v_local = (double *) malloc(sizeof(double) * chunk_size);

    std::unique_ptr<int[]> chunks_sizes  = p.get_chunks_sizes();
    std::unique_ptr<int[]> chunks_starts = p.get_chunks_starts();

    MPI_Scatterv(
        u, chunks_sizes.get(),
        chunks_starts.get(), MPI_DOUBLE, 
        u_local, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(
        v, chunks_sizes.get(),
        chunks_starts.get(), MPI_DOUBLE,
        v_local, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(u);
        free(v);
    }

    double d = dot(u_local, v_local, p);

    free(u_local);
    free(v_local);

    if (rank == 0) {
        printf("DOT PRODUCT: %.2f\n", d);
    }

    MPI_Finalize();
    return 0;
}
