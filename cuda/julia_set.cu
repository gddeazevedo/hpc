#include "bitmap.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using byte = unsigned char;

/**
 * Class representing a complex number, with real part a and imaginary part b.
 * Re(a + bi) = a
 * Im(a + bi) = b
 */
class Complex {
    private:
        double a;
        double b;

    public:
        __device__ __host__ Complex(double a, double b): a(a), b(b) {}

        __device__ double abs() {
            return sqrt(a*a + b*b);
        }

        __device__ Complex operator*(Complex &other) {
            double real_part      = a * other.a - b * other.b;
            double imaginary_part = a * other.b + other.a * b;
            return Complex(real_part, imaginary_part);
        }

        __device__ Complex operator+(Complex &other) {
            return Complex(a + other.a, b + other.b);
        }
};

__device__ double julia_map(int val, double _min, double _max, int size) {
    return _min + val * (_max - _min) / size;
}

__device__ int get_color(double x, double y, Complex c, int max_iter, double max_abs_z) {
    Complex z(x, y);
    int iter = 0;

    while (z.abs() < max_abs_z && iter < max_iter) {
        z = z * z + c;
        iter++;
    }

    return iter;
}

__global__ void julia_set(
    double xmin, 
    double xmax, 
    double ymin, 
    double ymax,
    int width,
    int height,
    Complex c,
    int max_iter,
    double max_abs_z,
    byte *buf
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < width && j < height) {
        double x = julia_map(i, xmin, xmax, width);
        double y = julia_map(j, ymin, ymax, height);
        int color = get_color(x, y, c, max_iter, max_abs_z);

        buf[3 * (i * height + j) + 0] = (color & 0xF)*16;
        buf[3 * (i * height + j) + 1] = ((color >> 2) & 0xF) * 16;
        buf[3 * (i * height + j) + 2] = ((color >> 3) & 0xF) * 16;
    }
}

int main() {
    int width    = 8000; 
    int height   = 8000;
    double xmin  = -1.5; 
    double ymin  = -1.5; 
    double xmax  = 1.5;
    double ymax  = 1.5; 
    int max_iter = 255;
    double max_abs_z = 64.;
    Complex c(-0.7, 0.27015);

    byte *buf = (byte *) malloc(width * height * 3);

    byte *d_buf;
    cudaMalloc(&d_buf, width * height * 3);

    dim3 threads_per_block(32, 32);
    dim3 blocks_per_grid(
        (width  + threads_per_block.x - 1) / threads_per_block.x,
        (height + threads_per_block.y - 1) / threads_per_block.y,
    );

    julia_set<<<blocks_per_grid, threads_per_block>>>(
        xmin, xmax
        ymin, ymax,
        width, height,
        c, max_iter, max_abs_z, d_buf
    );

    cudaMemcpy(buf, d_buf, width * height * 3, cudaMemcpyDeviceToHost);

    int ret = bmp_generator("julia_fractal.bmp", width, height, buf);

    return 0;
}