#include <iostream>

__global__ void reduce_vector(double *vec, double *output) {
    extern __shared__ double partial_sum[];

    int global_id  = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id   = threadIdx.x;

    partial_sum[local_id] = vec[global_id];
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (local_id < i) {
            partial_sum[local_id] += partial_sum[local_id + i];
        }
        __syncthreads();
    }

    if (local_id == 0) {
        output[blockIdx.x] = partial_sum[0];
    }
}

int main() {
    constexpr int n = 10;
    constexpr int threads_per_block = 2;
    constexpr int blocks = n / threads_per_block;

    double *vec = new double[n];
    for (int i = 0; i < n; i++) {
        vec[i] = i + 1;
    }

    double *d_vec, *d_output;
    cudaMalloc(&d_vec, sizeof(double) * n);
    cudaMemcpy(d_vec, vec, sizeof(double) * n, cudaMemcpyHostToDevice);
    delete[] vec;

    cudaMalloc(&d_output, sizeof(double) * blocks);

    reduce_vector<<<blocks, threads_per_block, sizeof(double) * threads_per_block>>>(d_vec, d_output);

    double *output = new double[blocks];
    cudaMemcpy(output, d_output, sizeof(double) * blocks, cudaMemcpyDeviceToHost);

    for (int i = 0; i < blocks; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    delete[] output;
    cudaFree(d_vec);
    cudaFree(d_output);

    return 0;
}