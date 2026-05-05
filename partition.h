#pragma once

class Partition 
{
    private:
        const int n;       // total size of the data
        const int n_procs; // number of processes
        const int rank;    // process id [0, n_procs-1]

    public:
        Partition(int n, int n_procs, int rank) : n(n), n_procs(n_procs), rank(rank) {}

        /**
         * Get the starting index of the partition for the current process
         */
        int get_start() const {
            if (rank < n % n_procs) {
                int chunk = n / n_procs + 1;
                return rank * chunk;
            }

            int n_procs_unb = n % n_procs; // number of processes that get an extra element
            int chunk_unb   = n / n_procs + 1;
            int chunk       = n / n_procs; // chunk size for processes that do not get an extra element
            return n_procs_unb * chunk_unb + (rank - n_procs_unb) * chunk;
        }

        int get_start(int proc) const {
            if (proc < n % n_procs) {
                int chunk = n / n_procs + 1;
                return proc * chunk;
            }

            int n_procs_unb = n % n_procs; // number of processes that get an extra element
            int chunk_unb   = n / n_procs + 1;
            int chunk       = n / n_procs; // chunk size for processes that do not get an extra element
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
            if (rank < n % n_procs) { // processes with rank < n % n_procs get an extra element
                return n / n_procs + 1;
            }

            return n / n_procs;
        }

        int get_chunk_size(int proc) const {
            if (proc < n % n_procs) { // processes with rank < n % n_procs get an extra element
                return n / n_procs + 1;
            }

            return n / n_procs;
        }

        int get_n() const {
            return n;
        }

        int *get_chunks_sizes() const {
            int *chunks_sizes = new int[n_procs];

            for (int i = 0; i < n_procs; i++) {
                chunks_sizes[i] = get_chunk_size(i);
            }

            return chunks_sizes;
        }

        int *get_chunks_starts() const {
            int *chunks_starts = new int[n_procs];

            for (int i = 0; i < n_procs; i++) {
                chunks_starts[i] = get_start(i);
            }

            return chunks_starts;
        }
};