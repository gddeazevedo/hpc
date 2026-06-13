#pragma once

#include <iostream>
#include <memory>

namespace partition {

/** 
 * @brief A class to partition a 1D array among multiple processes.
*/
class Partition1D
{
    private:
        const int total_size; // total size of the data
        const int n_procs;    // number of processes
        const int rank;       // process id [0, n_procs-1]

    public:
        Partition1D(const int total_size, const int n_procs, const int rank) : total_size(total_size), n_procs(n_procs), rank(rank) {}

        /**
         * Get the starting index of the partition for the current process
         */
        int get_start() const {
            int n_procs_unb = this->total_size % this->n_procs; // number of processes that get an extra element

            if (this->rank < n_procs_unb) {
                int chunk = this->total_size / this->n_procs + 1;
                return this->rank * chunk;
            }

            int chunk_unb   = this->total_size / this->n_procs + 1;
            int chunk       = this->total_size / this->n_procs; // chunk size for processes that do not get an extra element

            int n_procs_balanced = this->rank - n_procs_unb;
            int unbalanced_offet = n_procs_unb * chunk_unb;

            return unbalanced_offet + n_procs_balanced * chunk;
        }

        int get_start(const int proc) const {
            int n_procs_unb = this->total_size % this->n_procs; // number of processes that get an extra element

            if (proc < n_procs_unb) {
                int chunk = this->total_size / this->n_procs + 1;
                return proc * chunk;
            }

            int chunk_unb   = this->total_size / this->n_procs + 1;
            int chunk       = this->total_size / this->n_procs; // chunk size for processes that do not get an extra element

            int n_procs_balanced = proc - n_procs_unb;
            int unbalanced_offet = n_procs_unb * chunk_unb;

            return unbalanced_offet + n_procs_balanced * chunk;
        }

        /**
         * Get the ending index of the partition for the current process
         */
        int get_end() const {
            int chunk = this->get_chunk_size();
            return this->get_start() + chunk - 1;
        }

        int get_end(const int proc) const {
            int chunk = this->get_chunk_size(proc);
            return this->get_start(proc) + chunk - 1;
        }

        int get_chunk_size() const {
            if (this->rank < this->total_size % this->n_procs) { // processes with rank < total_size % n_procs get an extra element
                return this->total_size / this->n_procs + 1;
            }

            return this->total_size / this->n_procs;
        }

        int get_chunk_size(const int proc) const {
            if (proc < this->total_size % this->n_procs) { // processes with rank < total_size % n_procs get an extra element
                return this->total_size / this->n_procs + 1;
            }

            return this->total_size / this->n_procs;
        }

        int get_total_size() const {
            return this->total_size;
        }

        /**
         * Get the sizes of the partitions for all processes
         */
        std::unique_ptr<int[]> get_chunks_sizes() const {
            std::unique_ptr<int[]> chunks_sizes = std::make_unique<int[]>(this->n_procs);

            for (int i = 0; i < this->n_procs; i++) {
                chunks_sizes[i] = this->get_chunk_size(i);
            }

            return chunks_sizes;
        }

        /**
         * Get the starting indices of the partitions for all processes
         */
        std::unique_ptr<int[]> get_chunks_starts() const {
            std::unique_ptr<int[]> chunks_starts = std::make_unique<int[]>(this->n_procs);

            for (int i = 0; i < this->n_procs; i++) {
                chunks_starts[i] = this->get_start(i);
            }

            return chunks_starts;
        }

        int get_global_index(const int local_index) const {
            return this->get_start() + local_index;
        }
};

class Partition2D
{
    // TODO: implement a class to partition a 2D array among multiple processes
};

} // namespace partition
