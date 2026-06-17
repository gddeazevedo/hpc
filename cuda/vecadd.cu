#include <stdio.h>
#include <stdlib.h>

__global__ void VecAdd(float *A, float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    constexpr int n = 100;
    float *a = (float *) malloc(sizeof(float) * n);
    float *b = (float *) malloc(sizeof(float) * n);
    float *c = (float *) malloc(sizeof(float) * n);

    for (int i = 0; i < n; i++) {
        a[i] = i + 1;
        b[i] = i + 1;
    }

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, sizeof(float) * n);
    cudaMalloc(&d_b, sizeof(float) * n);
    cudaMalloc(&d_c, sizeof(float) * n);

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    free(a);
    free(b);

    dim3 blocksPerGrid(1);
    dim3 threadsPerBlock(n);

    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    for (int i = 0; i < n; i++) {
        printf("%.2f ", c[i]);
    }

    printf("\n");
    free(c);

    return 0;
}
