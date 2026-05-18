#pragma once

#include <iostream>
#include <memory>


class Partition 
{
    private:
        const int total_size; // total size of the data
        const int n_procs;    // number of processes
        const int rank;       // process id [0, n_procs-1]

    public:
        Partition(int total_size, int n_procs, int rank) : total_size(total_size), n_procs(n_procs), rank(rank) {}

        /**
         * Get the starting index of the partition for the current process
         */
        int get_start() const {
            if (rank < total_size % n_procs) {
                int chunk = total_size / n_procs + 1;
                return rank * chunk;
            }

            int n_procs_unb = total_size % n_procs; // number of processes that get an extra element
            int chunk_unb   = total_size / n_procs + 1;
            int chunk       = total_size / n_procs; // chunk size for processes that do not get an extra element
            return n_procs_unb * chunk_unb + (rank - n_procs_unb) * chunk;
        }

        int get_start(int proc) const {
            if (proc < total_size % n_procs) {
                int chunk = total_size / n_procs + 1;
                return proc * chunk;
            }

            int n_procs_unb = total_size % n_procs; // number of processes that get an extra element
            int chunk_unb   = total_size / n_procs + 1;
            int chunk       = total_size / n_procs; // chunk size for processes that do not get an extra element
            return n_procs_unb * chunk_unb + (proc - n_procs_unb) * chunk;
        }

        /**
         * Get the ending index of the partition for the current process
         */
        int get_end() const {
            int chunk = get_chunk_size();
            return get_start() + chunk - 1;
        }

        int get_end(int proc) const {
            int chunk = get_chunk_size(proc);
            return get_start(proc) + chunk - 1;
        }

        int get_chunk_size() const {
            if (rank < total_size % n_procs) { // processes with rank < total_size % n_procs get an extra element
                return total_size / n_procs + 1;
            }

            return total_size / n_procs;
        }

        int get_chunk_size(int proc) const {
            if (proc < total_size % n_procs) { // processes with rank < total_size % n_procs get an extra element
                return total_size / n_procs + 1;
            }

            return total_size / n_procs;
        }

        int get_total_size() const {
            return total_size;
        }

        /**
         * Get the sizes of the partitions for all processes
         */
        std::unique_ptr<int[]> get_chunks_sizes() const {
            std::unique_ptr<int[]> chunks_sizes = std::make_unique<int[]>(n_procs);

            for (int i = 0; i < n_procs; i++) {
                chunks_sizes[i] = get_chunk_size(i);
            }

            return chunks_sizes;
        }

        /**
         * Get the starting indices of the partitions for all processes
         */
        std::unique_ptr<int[]> get_chunks_starts() const {
            std::unique_ptr<int[]> chunks_starts = std::make_unique<int[]>(n_procs);

            for (int i = 0; i < n_procs; i++) {
                chunks_starts[i] = get_start(i);
            }

            return chunks_starts;
        }
};

class Partition2D
{
    // TODO: implement
};
