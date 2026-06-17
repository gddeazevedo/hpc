#include <cstdlib>
#include <cstdio>

__global__ void reduce(const int *v, int *out, const int n) {
    extern __shared__ int block_sum[];

    const int lid       = threadIdx.x;
    const int n_threads = blockDim.x;
    const int gid       = n_threads * blockIdx.x + lid;

    block_sum[lid] = gid < n ? v[gid] : 0;
    __syncthreads();

    for (int i = n_threads / 2; i > 0; i /= 2) {
        if (lid < i) {
            block_sum[lid] += block_sum[lid + i];
        }

        __syncthreads();
    }


    if (lid == 0) {
        out[blockIdx.x] = block_sum[0];
    }
}

__host__ int ceil_div(int n, int m) {
    return (n + m - 1) / m;
}

int main(int argc, char **argv) {
    const int n = atoi(argv[1]);

    const int vec_bytes = sizeof(int) * n;

    int *v = (int *) malloc(vec_bytes);

    for (int i = 0; i < n; i++) {
        v[i] = i + 1;
    }

    dim3 threads_per_block(16);
    dim3 blocks_per_grid(ceil_div(n, threads_per_block.x));

    int *d_v;
    cudaMalloc(&d_v, vec_bytes);
    cudaMemcpy(d_v, v, vec_bytes, cudaMemcpyHostToDevice);
    free(v);

    const int shared_size = threads_per_block.x * sizeof(int);
    const int out_size    = blocks_per_grid.x   * sizeof(int);
    int *d_out;
    cudaMalloc(&d_out, out_size);

    reduce<<<blocks_per_grid, threads_per_block, shared_size>>>(d_v, d_out, n);
    
    cudaFree(d_v);
    
    int *out = (int *) malloc(out_size);
    cudaMemcpy(out, d_out, out_size, cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    int sum = 0.0;

    for (int i = 0; i < blocks_per_grid.x; i++) {
        sum += out[i];
    }

    free(out);

    printf("Sum: %d\n", sum);

    return 0;
}
