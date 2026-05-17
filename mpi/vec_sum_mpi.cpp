#include <mpi.h>
#include <iostream>
#include "partition.h"


void vec_sum(double *r, const double *v, const double *w, const Partition &p) {
    for (int i = 0; i < p.get_chunk_size(); i++) {
        r[i] = v[i] + w[i];
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 8;

    Partition p(n, size, rank);

    double *v = nullptr;
    double *w = nullptr;
    double *r = nullptr;

    double *v_local = new double[p.get_chunk_size()];
    double *w_local = new double[p.get_chunk_size()];
    double *r_local = new double[p.get_chunk_size()];

    if (rank == 0) {
        v = new double[n];
        w = new double[n];
        r = new double[n];

        for (int i = 0; i < n; i++) {
            v[i] = i;
            w[i] = i;
        }
    }

    std::unique_ptr<int[]> sendcounts    = p.get_chunks_sizes();
    std::unique_ptr<int[]> displacements = p.get_chunks_starts();

    MPI_Scatterv(
        v, sendcounts.get(), displacements.get(), MPI_DOUBLE,
        v_local, p.get_chunk_size(), MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );
    MPI_Scatterv(
        w, sendcounts.get(), displacements.get(), MPI_DOUBLE,
        w_local, p.get_chunk_size(), MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    vec_sum(r_local, v_local, w_local, p);
    
    MPI_Gatherv(
        r_local, p.get_chunk_size(), MPI_DOUBLE,
        r, sendcounts.get(), displacements.get(), MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    delete[] v_local;
    delete[] w_local;
    delete[] r_local;

    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            std::cout << r[i] << std::endl;
        }

        delete[] v;
        delete[] w;
        delete[] r;
    }

    MPI_Finalize();
    return 0;
}