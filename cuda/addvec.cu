%%cuda

#include <iostream>

__global__ void VecAdd(double *a, double *b, double *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


int main() {
    constexpr int N = 100;
    double *a, *b, *c;
    
    a = new double[N];
    b = new double[N];
    c = new double[N];

    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i;
    }

    double *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(double));
    cudaMalloc(&d_b, N * sizeof(double));
    cudaMalloc(&d_c, N * sizeof(double));

    cudaMemcpy(d_a, a, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, N * sizeof(double), cudaMemcpyHostToDevice);

    delete[] a;
    delete[] b;

    VecAdd<<<1, N>>>(d_a, d_b, d_c);
    cudaMemcpy(c, d_c, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    for (int i = 0; i < N; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;
    delete[] c;
    return 0;
}