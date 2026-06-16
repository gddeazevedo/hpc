#pragma once

#include <memory>


class Partition
{
    private:
        int n;
        int rank;
        int n_procs;

    public:
        Partition(const int n, const int rank, const int n_procs) :
            n(n), rank(rank), n_procs(n_procs) {}

        int get_n() const
        {
            return this->n;
        }

        int get_rank() const
        {
            return this->rank;
        }

        int get_start() const
        {
            int n_unb_procs = this->n % this->n_procs;
            int chunk_unb   = this->n / this->n_procs + 1;

            if (this->rank < n_unb_procs) {
                return chunk_unb * this->rank;
            }

            int chunk  = this->n / this->n_procs;
            int offset = this->rank - n_unb_procs;

            return chunk_unb * n_unb_procs + chunk * offset;
        }

        int get_start(int proc) const
        {
            int n_unb_procs = this->n % this->n_procs;
            int chunk_unb   = this->n / this->n_procs + 1;

            if (proc < n_unb_procs) {
                return chunk_unb * proc;
            }

            int chunk  = this->n / this->n_procs;
            int offset = proc - n_unb_procs;

            return chunk_unb * n_unb_procs + chunk * offset;
        }

        int get_chunk_size() const
        {
            if (this->rank < this->n % this->n_procs) {
                return this->n / this->n_procs + 1;
            }

            return this->n / this->n_procs;
        }

        int get_chunk_size(int proc) const
        {
            if (proc < this->n % this->n_procs) {
                return this->n / this->n_procs + 1;
            }

            return this->n / this->n_procs;
        }

        int get_end() const
        {
            int chunk = this->get_chunk_size();
            return this->get_start() + chunk - 1;
        }

        int get_end(int proc) const
        {
            int chunk = this->get_chunk_size(proc);
            return this->get_start(proc) + chunk - 1;
        }

        std::unique_ptr<int[]> get_chunks_sizes() const
        {
            std::unique_ptr<int[]> chunks_sizes = std::make_unique<int[]>(this->n_procs);

            for (int proc = 0; proc < this->n_procs; proc++) {
                chunks_sizes[proc] = this->get_chunk_size(proc);
            }

            return chunks_sizes;
        }

        std::unique_ptr<int[]> get_chunks_starts() const
        {
            std::unique_ptr<int[]> chunks_starts = std::make_unique<int[]>(this->n_procs);

            for (int proc = 0; proc < this->n_procs; proc++) {
                chunks_starts[proc] = this->get_start(proc);
            }

            return chunks_starts;
        }
};