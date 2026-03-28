#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#ifdef _WIN32
    #include <windows.h>
#else
    #include <linux/time.h>
#endif

#define INTERVAL_SIZE_LIMIT 100000


void merge(int *arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = (int *)malloc(n1 * sizeof(int));
    int *R = (int *)malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int i = 0; i < n2; i++)
        R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while (i < n1)
        arr[k++] = L[i++];
    while (j < n2)
        arr[k++] = R[j++];

    free(L);
    free(R);
}

void merge_sort(int *arr, int left, int right) {
    if (left < right) {
        int mid = (left + right) / 2;
        merge_sort(arr, left, mid);
        merge_sort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}


void merge_sort_parallel(int *arr, int left, int right) {
    if (left < right) {
        int mid = (left + right) / 2;

        // Por conta da baixa granularidade, não compensa delegar threads, melhor fazer sequencial
        // como serão poucos dados para processar, não terá overhead de gerenciamento de tasks para as threads
        // sendo mais rapido para poucos dados
        if (right - left < INTERVAL_SIZE_LIMIT) {
            merge_sort_parallel(arr, left, mid);
            merge_sort_parallel(arr, mid + 1, right);
        } else {
            #pragma omp task
            merge_sort_parallel(arr, left, mid);
            #pragma omp task
            merge_sort_parallel(arr, mid + 1, right);
        }

        #pragma omp taskwait
        merge(arr, left, mid, right);
    }
}


int partition(int *arr, int low, int high) {
    int pivot = arr[high]; // pivot simples

    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            int tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }

    int tmp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = tmp;

    return i + 1;
}

void quick_sort(int *arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
    }
}

void quick_sort_parallel(int *arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        #pragma omp task
        quick_sort_parallel(arr, low, pi - 1);
        #pragma omp task
        quick_sort_parallel(arr, pi + 1, high);
    }
}


double get_time() {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
#endif
}

void fill_random(int *arr, int n) {
    for (int i = 0; i < n; i++)
        arr[i] = rand();
}


void is_sorted(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            printf("Not is sorted :(\n");
            return;
        }
    }

    printf("Is sorted :)\n");
}


int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Uso: %s <tamanho>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    srand(42);

    int *base = (int *)malloc(n * sizeof(int));
    int *arr1 = (int *)malloc(n * sizeof(int));
    int *arr2 = (int *)malloc(n * sizeof(int));
    int *arr3 = (int *)malloc(n * sizeof(int));
    int *arr4 = (int *)malloc(n * sizeof(int));

    fill_random(base, n);

    memcpy(arr1, base, n * sizeof(int));
    memcpy(arr2, base, n * sizeof(int));
    memcpy(arr3, base, n * sizeof(int));
    memcpy(arr4, base, n * sizeof(int));

    free(base);

    printf("Tamanho: %d\n", n);


    double t_ini;
    double t_fim;

    // -------- Quick Sort --------
    t_ini = get_time();
    quick_sort(arr1, 0, n - 1);
    t_fim = get_time();
    is_sorted(arr1, n);
    printf("Quick Sort: %.6f s\n", t_fim - t_ini);
    free(arr1);

    t_ini = get_time();
    #pragma omp parallel
    #pragma omp single
    quick_sort_parallel(arr2, 0, n - 1);
    t_fim = get_time();
    is_sorted(arr2, n);
    printf("Quick Sort Parallel: %.6f s\n", t_fim - t_ini);
    free(arr2);

    // -------- Merge Sort --------
    t_ini = get_time();
    merge_sort(arr3, 0, n - 1);
    t_fim = get_time();
    is_sorted(arr3, n);
    printf("Merge Sort: %.6f s\n", t_fim - t_ini);
    free(arr3);

    t_ini = get_time();
    #pragma omp parallel
    #pragma omp single
    merge_sort_parallel(arr4, 0, n - 1);
    t_fim = get_time();
    is_sorted(arr4, n);
    printf("Merge Sort Parallel: %.6f s\n", t_fim - t_ini);
    free(arr4);

    return 0;
}
