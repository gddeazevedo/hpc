#include <cstdlib>
#include <cstdio>
#include <cmath>

__global__ void reduce(
    const double *X,
    const double *Y,
    double *out,
    const double mean_X,
    const double mean_Y,
    const int n
) {
    extern __shared__ double block_sum[];

    const int lid       = threadIdx.x;
    const int n_threads = blockDim.x;
    const int gid       = n_threads * blockIdx.x + lid;

    block_sum[lid] = gid < n ? (X[gid] - mean_X) * (Y[gid] - mean_Y) : 0;
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

__global__ void reduce(const double *X, double *out, const double mean_X, const int n) {
    extern __shared__ double block_sum[];

    const int lid       = threadIdx.x;
    const int n_threads = blockDim.x;
    const int gid       = n_threads * blockIdx.x + lid;

    block_sum[lid] = gid < n ? (X[gid] - mean_X) * (X[gid] - mean_X) : 0;
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

__global__ void reduce(const double *X, double *out, const int n) {
    extern __shared__ double block_sum[];

    const int lid       = threadIdx.x;
    const int n_threads = blockDim.x;
    const int gid       = n_threads * blockIdx.x + lid;

    block_sum[lid] = gid < n ? X[gid] : 0;
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

__host__ double get_mean(const double *d_X, double *d_out, const int n) {
    dim3 threads_per_block(16);
    dim3 blocks_per_grid(ceil_div(n, threads_per_block.x));

    reduce<<<blocks_per_grid, threads_per_block, sizeof(double) * threads_per_block.x>>>(d_X, d_out, n);

    double *out = (double *) malloc(sizeof(double) * blocks_per_grid.x);
    cudaMemcpy(out, d_out, sizeof(double) * blocks_per_grid.x, cudaMemcpyDeviceToHost);

    double sum = 0.0;

    for (int i = 0; i < blocks_per_grid.x; i++) {
        sum += out[i];
    }

    free(out);

    return sum / (double) n;
}

__host__ double get_var(
    const double *d_X,
    double *d_out,
    const double mean_X,
    const int n
) {
    dim3 threads_per_block(16);
    dim3 blocks_per_grid(ceil_div(n, threads_per_block.x));

    reduce<<<blocks_per_grid, threads_per_block, sizeof(double) * threads_per_block.x>>>(d_X, d_out, mean_X, n);

    double *out = (double *) malloc(sizeof(double) * blocks_per_grid.x);
    cudaMemcpy(out, d_out, sizeof(double) * blocks_per_grid.x, cudaMemcpyDeviceToHost);

    double sum = 0.0;

    for (int i = 0; i < blocks_per_grid.x; i++) {
        sum += out[i];
    }

    free(out);

    return sum; 
}

__host__ double get_covar(
    const double *d_X,
    const double *d_Y,
    double *d_out,
    const double mean_X,
    const double mean_Y,
    const int n
) {
    dim3 threads_per_block(16);
    dim3 blocks_per_grid(ceil_div(n, threads_per_block.x));

    reduce<<<blocks_per_grid, threads_per_block, sizeof(double) * threads_per_block.x>>>(
        d_X, d_Y, d_out, mean_X, mean_Y, n);

    double *out = (double *) malloc(sizeof(double) * blocks_per_grid.x);
    cudaMemcpy(out, d_out, sizeof(double) * blocks_per_grid.x, cudaMemcpyDeviceToHost);

    double sum = 0.0;

    for (int i = 0; i < blocks_per_grid.x; i++) {
        sum += out[i];
    }

    free(out);

    return sum; 
}

int main(int argc, char **argv) {
    const int n = atoi(argv[1]);

    double *X = (double *) malloc(sizeof(double) * n);
    double *Y = (double *) malloc(sizeof(double) * n);

    for (int i = 0; i < n; i++) {
        X[i] = i + 1;
        Y[i] = 2 * X[i];
    }

    dim3 threads_per_block(16);
    dim3 blocks_per_grid(ceil_div(n, threads_per_block.x));

    double *d_X, *d_Y, *d_out;

    cudaMalloc(&d_out, sizeof(double) * blocks_per_grid.x);
    cudaMalloc(&d_X, sizeof(double) * n);
    cudaMalloc(&d_Y, sizeof(double) * n);

    cudaMemcpy(d_X, X, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, sizeof(double) * n, cudaMemcpyHostToDevice);

    free(X);
    free(Y);

    double mean_X   = get_mean(d_X, d_out, n);
    double mean_Y   = get_mean(d_Y, d_out, n);
    double var_X    = get_var(d_X, d_out, mean_X, n);
    double var_Y    = get_var(d_Y, d_out, mean_Y, n);
    double covar_XY = get_covar(d_X, d_Y, d_out, mean_X, mean_Y, n);

    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_out);

    double rho   = covar_XY / sqrt(var_X * var_Y);
    double beta  = covar_XY / var_X;
    double alpha = mean_Y - beta * mean_X;

    printf("ρ = %.2f\n", rho);
    printf("β = %.2f\n", beta);
    printf("α = %.2f\n", alpha);

    return 0;
}