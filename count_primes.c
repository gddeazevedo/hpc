// Conta a quantidade de numeros primos existentes menores ou iguais a N

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


int is_prime(int n) {
    if (n == 0 || n == 1) {
        return 0;
    }

    if (n == 2) {
        return 1;
    }

    for (int i = 2; i <= ceil(sqrt(n)); i++) {
        if (n % i == 0) {
            return 0;
        }
    }

    return 1;
}


int main(int argc, char **argv) {
    unsigned int N = atoi(argv[1]);

    unsigned int total_primes = 0;

    #pragma omp parallel for reduction(+:total_primes) schedule(dynamic)
    for (int n = 2; n <= N; n++) {
        if (is_prime(n)) {
            total_primes++;
        }
    }

    printf("Total primes until %d = %d\n", N, total_primes);

    return 0;
}