#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

#define NUM_CHARS 256
#define MAX_SIZE 10000

void hist(unsigned char *txt, unsigned int h[NUM_CHARS], unsigned long txt_size) {
    for (int i = 0; i < NUM_CHARS; i++) {
        h[i] = 0;
    }

    for (long i = 0; i < txt_size; i++) {
        unsigned int character = (unsigned int) txt[i];
        h[character] += 1;
    }
}

void hist_parallel_v1(unsigned char *txt, unsigned int h[NUM_CHARS], unsigned long txt_size) {
    for (int i = 0; i < NUM_CHARS; i++) {
        h[i] = 0;
    }

    #pragma omp parallel for
    for (long i = 0; i < txt_size; i++) {
        unsigned int character = (unsigned int) txt[i];
        #pragma omp atomic
        h[character] += 1;
    }
}

void hist_parallel_v2(unsigned char *txt, unsigned int h[NUM_CHARS], unsigned long txt_size) {
    for (int i = 0; i < NUM_CHARS; i++) {
        h[i] = 0;
    }

    #pragma omp parallel
    {
        unsigned int local_h[NUM_CHARS];

        for (int i = 0; i < NUM_CHARS; i++) {
            local_h[i] = 0;
        }

        #pragma omp for
        for (long i = 0; i < txt_size; i++) {
            unsigned int character = (unsigned int) txt[i];
            local_h[character] += 1;
        }

        #pragma omp critical
        for (int i = 0; i < NUM_CHARS; i++) {
            h[i] += local_h[i];
        }
    }
}

unsigned int randint(int min, int max) {
    return rand() % (max - min + 1) + min;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <tamanho>\n", argv[0]);
        return 1;
    }

    srand(time(NULL));

    unsigned long txt_size = atol(argv[1]);

    unsigned char *txt = (unsigned char*) malloc(txt_size);

    for (long i = 0; i < txt_size; i++) {
        txt[i] = (unsigned char) randint(0, NUM_CHARS - 1);
    }

    unsigned int h[NUM_CHARS];

    double t0 = omp_get_wtime();
    hist(txt, h, txt_size);
    double t1 = omp_get_wtime();

    printf("Sequential Time: %f\n", t1 - t0);

    t0 = omp_get_wtime();
    hist_parallel_v1(txt, h, txt_size);
    t1 = omp_get_wtime();

    printf("Parallel Time (Atomic): %f\n", t1 - t0);

    t0 = omp_get_wtime();
    hist_parallel_v2(txt, h, txt_size);
    t1 = omp_get_wtime();

    printf("Parallel Time (Scatter & Gather): %f\n", t1 - t0);

    free(txt);

    return 0;
}