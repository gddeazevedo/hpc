#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct Node {
    int weight;
    int num_children;
    struct Node** children;
} Node;

Node* build_tree(int depth, int max_depth) {
    Node* node = (Node*)malloc(sizeof(Node));
    node->weight = rand() % 100;
    
    if (depth < max_depth) {
        node->num_children = (rand() % 2) + 2; 
        node->children = (Node**)malloc(node->num_children * sizeof(Node*));
        
        for (int i = 0; i < node->num_children; i++) {
            node->children[i] = build_tree(depth + 1, max_depth);
        }
    } else {
        node->num_children = 0;
        node->children = NULL;
    }
    return node;
}

long sum_tree(Node* node) {
    if (node == NULL) return 0;

    long total_sum = node->weight;

    for (int i = 0; i < node->num_children; i++) {
        
        total_sum += sum_tree(node->children[i]);
    }

    return total_sum;
}

long sum_tree_par(Node *node, int depth) {
    if (node == NULL) return 0;

    if (depth > 4) {
        return sum_tree(node);
    }

    long total_sum = node->weight;

    long *partial_sums = (long *) calloc(node->num_children, sizeof(long));

    for (int i = 0; i < node->num_children; i++) {
        #pragma omp task shared(partial_sums)
        partial_sums[i] = sum_tree_par(node->children[i], depth + 1);
    }

    #pragma omp taskwait

    for (int i = 0; i < node->num_children; i++) {
        total_sum += partial_sums[i];
    }

    free(partial_sums);

    return total_sum;
}

void free_tree(Node* node) {
    if (!node) return;
    for (int i = 0; i < node->num_children; i++) {
        free_tree(node->children[i]);
    }
    if (node->children) free(node->children);
    free(node);
}

int main(int argc, char ** argv ) {
    int depth = atoi( argv[1] ); 
    
    struct timespec start, end;
    
    Node* root = build_tree(0, depth);


    long result = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    #pragma omp parallel
    #pragma omp single
    result = sum_tree_par(root, 0);
        
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("[PAR] Soma Total: %ld\n", result);
    printf("[PAR] Tempo total: %f segundos\n", time_taken);




    clock_gettime(CLOCK_MONOTONIC, &start);

    result = sum_tree(root);

    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("[SEQ] Soma Total: %ld\n", result);
    printf("[SEQ] Tempo total: %f segundos\n", time_taken);




    
    free_tree(root);
    return 0;
}
