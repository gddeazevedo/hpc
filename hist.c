#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

#define CHARS 256
#define MAX_SIZE 10000

void hist(uint8_t *txt, uint32_t h[CHARS], uint64_t size) {
    for (uint64_t i = 0; i < CHARS; i++) {
        h[i] = 0;
    }

    for (uint64_t i = 0; i < size; i++) {
        uint32_t idx = (uint32_t) txt[i];
        h[idx] += 1;
    }
}

void hist_parallel(uint8_t *txt, uint32_t h[CHARS], uint64_t size) {
    #pragma omp parallel
    {
        uint8_t local_h[CHARS];

        for (uint64_t i = 0; i < CHARS; i++) {
            local_h[i] = 0; // private para cada thread
        }

        #pragma omp for
        for (uint64_t i = 0; i < size; i++) {
            uint32_t idx = (uint32_t) txt[i];
            local_h[idx] += 1;
        }

        // somente uma thread pode acessar a região crítica por vez
        #pragma omp critical
        for (uint64_t i = 0; i < CHARS; i++) {
            h[i] += local_h[i];
        }
    }
}

uint32_t randint(int32_t min, int32_t max) {
    return rand() % (max - min + 1) + min;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <tamanho>\n", argv[0]);
        return 1;
    }

    srand(time(NULL));

    uint64_t n = atol(argv[1]);

    uint8_t *txt = (uint8_t *) malloc(n);

    for (uint64_t i = 0; i < n; i++) {
        txt[i] = (uint8_t) randint(0, CHARS - 1);
    }

    uint32_t h[CHARS];

    double t0 = omp_get_wtime();
    hist(txt, h, n);
    double t1 = omp_get_wtime();

    printf("Sequential Time: %f\n", t1 - t0);

    t0 = omp_get_wtime();
    hist_parallel(txt, h, n);
    t1 = omp_get_wtime();

    printf("Parallel Time: %f\n", t1 - t0);

    free(txt);

    return 0;
}