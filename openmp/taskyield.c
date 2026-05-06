#include <stdlib.h>
#include <stdio.h>


void foo() {
    for (int i = 0; i < 10; i++) {
        printf("%d FOO!\n", i);
    }
}

void bar() {
    printf("BAR\n");
}

void too() {
    printf("TOO\n");
}

void too2() {
    printf("TOO2\n");
}

int main() {

    #pragma omp parallel
    #pragma omp single
    {
        #pragma omp task
        {
            foo();
            #pragma omp taskyield
            bar();
        }

        #pragma omp task
        too();

        #pragma omp task
        too2();
    }


    return 0;
}
