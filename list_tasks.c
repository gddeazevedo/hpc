#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

struct list {
    int value;
    struct list *next;
};

typedef struct list list_t;

list_t *gen_list(int n) {
    list_t *head = (list_t *) malloc(sizeof(list_t));
    list_t *ptr = head;

    for (int i = 0; i < n; i++) {
        ptr->value = i;
        if (i == n - 1) {
            ptr->next = NULL;
        } else {
            ptr->next = (list_t *) malloc(sizeof(list_t));
        }
        ptr = ptr->next;
    }

    return head;
}

void free_list(list_t **head) {
    if (*head == NULL) {
        return;
    }

    list_t *ptr = *head;

    *head = NULL;

    while (ptr != NULL) {
        list_t *tmp = ptr->next;
        free(ptr);
        ptr = tmp;
    }
}

void print_list(list_t *head) {
    list_t *ptr = head;

    printf("[");

    while (ptr != NULL) {
        if (ptr->next != NULL) {
            printf("%d, ", ptr->value);
        } else {
            printf("%d", ptr->value);
        }

        ptr = ptr->next;
    }

    printf("]\n");
}

void process(list_t *ptr) {
    printf("Sleeping for %d\n", ptr->value);
    sleep(ptr->value);
    printf("Done %d\n", ptr->value);
}

int main(int argc, char **argv) {
    int n = atoi(argv[1]);

    list_t *head = gen_list(n);


    print_list(head);

    list_t *ptr = head;

    #pragma omp parallel
    #pragma omp single
    while (ptr != NULL) {
        #pragma omp task
        process(ptr);
        ptr = ptr->next;
    }

    free_list(&head);

    return 0;
}
