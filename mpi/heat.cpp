#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include "partition.h"

#define ROOT_RANK 0
#define HALO_QNT  2
#define NO_TAG    0


bool is_border(const int row, const int col, const int n);
double **generate_matrix(const int n);
double **alloc_submatrix(const partition::Partition1D &p);
void free_matrix(double **M, const partition::Partition1D &p);
void free_matrix(double **M, const int n);
void send_domains(double **T, const partition::Partition1D &p);
void recv_domains(double **T_prev, const partition::Partition1D &p);
void init_root_domain(double **T, double **T_prev, const partition::Partition1D &p);
void calculate_heat_propagation(double **T_prev, double **T_next, const partition::Partition1D &p);
void print_matrix(double **T_prev, const partition::Partition1D &p);


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int n = atoi(argv[1]);

    int size;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    partition::Partition1D p(n, size, rank);

    double **T = nullptr;
    double **T_prev = alloc_submatrix(p);
    double **T_next = alloc_submatrix(p);

    if (rank == ROOT_RANK) {
        T = generate_matrix(n);
        init_root_domain(T, T_prev, p);
        send_domains(T, p);
    } else {
        recv_domains(T_prev, p);
    }

    for (int n_iter = 0; n_iter < 2000; n_iter++) {
        calculate_heat_propagation(T_prev, T_next, p);
    }

    free_matrix(T_next, p);

    if (rank == ROOT_RANK) {
        free_matrix(T, n);
    }

    print_matrix(T_prev, p);

    free_matrix(T_prev, p);

    MPI_Finalize();
    return 0;
}


bool is_border(const int row, const int col, const int n) {
    return row == 0 || row == n - 1 || col == 0 || col == n - 1;
}

double **generate_matrix(const int n) {
    double **M = (double **) malloc(sizeof(double *) * n);

    for (int row = 0; row < n; row++) {
        M[row] = (double *) malloc(sizeof(double) * n);

        for (int col = 0; col < n; col++) {
            if (row == 0 && col == 1) {
                M[row][col] = 100.0;
            } else {
                M[row][col] = 0.0;
            }
        }
    }

    return M;
}

double **alloc_submatrix(const partition::Partition1D &p) {
    const int n_rows     = p.get_chunk_size() + HALO_QNT;
    const int total_size = p.get_total_size();

    // índice das linhas do dominio local vão de 0 até chunk_size + 1
    // no qual as linhas 0 e chunk_size + 1 são as linhas halo/ghost
    double **M = (double **) malloc(sizeof(double *) * n_rows);

    for (int row = 0; row < n_rows; row++) {
        M[row] = (double *) malloc(sizeof(double) * total_size);
    }

    return M;
}

void free_matrix(double **M, const partition::Partition1D &p) {
    if (M == nullptr) {
        return;
    }

    const int n_rows = p.get_chunk_size() + HALO_QNT;

    for (int row = 0; row < n_rows; row++) {
        free(M[row]);
    }

    free(M);
}

void free_matrix(double **M, const int n) {
    if (M == nullptr) {
        return;
    }

    for (int row = 0; row < n; row++) {
        free(M[row]);
    }

    free(M);
}

void send_domains(double **T, const partition::Partition1D &p) {
    std::unique_ptr<int[]> sendcounts    = p.get_chunks_sizes();
    std::unique_ptr<int[]> displacements = p.get_chunks_starts();

    const int n_procs = p.get_n_procs();
    const int n       = p.get_total_size();

    for (int proc = ROOT_RANK + 1; proc < n_procs; proc++) {
        int chunk_size   = sendcounts[proc];
        int displacement = displacements[proc];

        // sending the lines to the process proc
        for (int row = 0; row < chunk_size; row++) {
            MPI_Send(T[row + displacement], n, MPI_DOUBLE, proc, NO_TAG, MPI_COMM_WORLD);
        }
    }
}

void recv_domains(double **T_prev, const partition::Partition1D &p) {
    const int chunk_size = p.get_chunk_size();
    const int n          = p.get_total_size();

    for (int row = 0; row < chunk_size; row++) {
        MPI_Recv(T_prev[row + 1], n, MPI_DOUBLE, ROOT_RANK, NO_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } 
}

void init_root_domain(double **T, double **T_prev, const partition::Partition1D &p) {
    const int chunk_size = p.get_chunk_size();
    const int n          = p.get_total_size();

    for (int row = 0; row < chunk_size; row++) {
        for (int col = 0; col < n; col++) {
            T_prev[row + 1][col] = T[row][col];
        }
    }
}

void calculate_heat_propagation(double **T_prev, double **T_next, const partition::Partition1D &p) {
    const int chunk_size = p.get_chunk_size();
    const int n          = p.get_total_size();
    const int rank       = p.get_rank();
    const int n_procs    = p.get_n_procs();

    int up_proc   = rank - 1;
    int down_proc = rank + 1;

    if (up_proc < 0) {
        up_proc = MPI_PROC_NULL;
    }

    if (down_proc > n_procs - 1) {
        down_proc = MPI_PROC_NULL;
    }

    MPI_Sendrecv(
        T_prev[chunk_size], n, MPI_DOUBLE, down_proc, NO_TAG,
        T_prev[chunk_size + 1], n, MPI_DOUBLE, down_proc, NO_TAG,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    MPI_Sendrecv(
        T_prev[1], n, MPI_DOUBLE, up_proc, NO_TAG,
        T_prev[0], n, MPI_DOUBLE, up_proc, NO_TAG,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    for (int row = 1; row < chunk_size + 1; row++) {
        for (int col = 1; col < n - 1; col++) {
            int global_row = p.get_global_index(row - 1);

            if (is_border(global_row, col, n)) {
                continue;
            }

            T_next[row][col] = 0.25 * (T_prev[row][col + 1] + T_prev[row][col - 1] + T_prev[row + 1][col] + T_prev[row - 1][col]);
        }
    }

    for (int row = 1; row < chunk_size + 1; row++) {
        for (int col = 1; col < n - 1; col++) {
            int global_row = p.get_global_index(row - 1);

            if (is_border(global_row, col, n)) {
                continue;
            }

            T_prev[row][col] = T_next[row][col];
        }
    }
}

void print_matrix(double **T_prev, const partition::Partition1D &p) {
    const int chunk_size = p.get_chunk_size();
    const int n          = p.get_total_size();
    const int rank       = p.get_rank();
    const int n_procs    = p.get_n_procs();

    for (int proc = ROOT_RANK; proc < n_procs; proc++) {
        if (proc == rank) {    
            for (int row = 1; row < chunk_size + 1; row++) {
                for (int col = 0; col < n; col++) {
                    printf("%.2f ", T_prev[row][col]);
                }
                
                printf("\n");
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
}
