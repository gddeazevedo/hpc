#include <mpi.h>
#include <iostream>
#include <cmath>
#include "partition.h"

double Mean(const double *x_local, const partition::Partition1D &p) {
    double sum = 0.0;

    for (int i = 0; i < p.get_chunk_size(); i++) {
        sum += x_local[i];
    }

    double local_mean = sum / p.get_total_size();
    double mean = 0.0;
    MPI_Allreduce(&local_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // todos precisam ter a media para calcular a variancia e covariancia

    return mean;
}

double Var(double *x_local, double mean, const partition::Partition1D &p) {
    double local_var = 0.0;

    for (int i = 0; i < p.get_chunk_size(); i++) {
        double diff = x_local[i] - mean;
        local_var += diff * diff;
    }

    double var = 0.0;
    MPI_Reduce(&local_var, &var, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return var;
}

double Cov(double *x_local, double mean_x, double *y_local, double mean_y, const partition::Partition1D &p) {
    double local_covar = 0.0;

    for (int i = 0; i < p.get_chunk_size(); i++) {
        double diff_x = x_local[i] - mean_x;
        double diff_y = y_local[i] - mean_y;
        local_covar += diff_x * diff_y;
    }

    double covar = 0.0;
    MPI_Reduce(&local_covar, &covar, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return covar;
}

int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    MPI_Init(&argc, &argv);

    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    partition::Partition1D p(n, size, rank);

    double *x = nullptr;
    double *y = nullptr;

    double *x_local = new double[p.get_chunk_size()];
    double *y_local = new double[p.get_chunk_size()];

    if (rank == 0) {
        x = new double[p.get_total_size()];
        y = new double[p.get_total_size()];

        for (int i = 0; i < p.get_total_size(); i++) {
            x[i] = i;
            y[i] = 2 * i;
        }
    }

    std::unique_ptr<int[]> sendcounts    = p.get_chunks_sizes();
    std::unique_ptr<int[]> displacements = p.get_chunks_starts();

    MPI_Scatterv(
        x, sendcounts.get(), displacements.get(), MPI_DOUBLE,
        x_local, p.get_chunk_size(), MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );
    MPI_Scatterv(
        y, sendcounts.get(), displacements.get(), MPI_DOUBLE,
        y_local, p.get_chunk_size(), MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    if (rank == 0) {
        delete[] x;
        delete[] y;
    }

    double mean_x = Mean(x_local, p);
    double mean_y = Mean(y_local, p);

    double cov_xy = Cov(x_local, mean_x, y_local, mean_y, p);

    double var_x = Var(x_local, mean_x, p);
    double var_y = Var(y_local, mean_y, p);

    delete[] x_local;
    delete[] y_local;

    if (rank == 0) {
        double beta  = cov_xy / var_x;
        double alpha = mean_y - beta * mean_x;
        double rho   = cov_xy / (sqrt(var_x) * sqrt(var_y));

        std::cout << "alpha: " << alpha << std::endl;
        std::cout << "beta:  " << beta  << std::endl;
        std::cout << "rho:   " << rho   << std::endl;
    }

    MPI_Finalize();
    return 0;
}
