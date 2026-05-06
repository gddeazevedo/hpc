#include <stdlib.h>
#include <stdio.h>


enum operation {
    ADD = '+',
    SUB = '-',
    MUL = '*',
    DIV = '/'
};

struct __ast {
    struct __ast *left;
    struct __ast *right;
    enum operation op;
    float val;
};

typedef struct __ast ast_t;


float exec_operation(float a, float b, enum operation op) {
    switch (op) {
        case ADD:
            return a + b;
        case SUB:
            return a - b;
        case MUL:
            return a * b;
        case DIV:
            return a / b;
        default:
            return 0;
    }
}

float eval(ast_t *ast) {
    if (ast->left == NULL && ast->right == NULL) {
        return ast->val;
    }

    float left_val;
    float right_val;

    #pragma omp task shared(left_val)
    left_val  = eval(ast->left);

    #pragma omp task shared(right_val)
    right_val = eval(ast->right);

    #pragma omp taskwait

    return exec_operation(left_val, right_val, ast->op);
}

ast_t *new_ast(float val) {
    ast_t *ast = (ast_t *)malloc(sizeof(ast_t));
    ast->left  = NULL;
    ast->right = NULL;
    ast->val = val;
    return ast;
}

ast_t *build_ast() {
    ast_t *ast = new_ast(0);
    ast->op = ADD;

    ast->left = new_ast(0);
    ast->left->op = MUL;
    ast->left->left = new_ast(3);
    ast->left->right = new_ast(4);

    ast->right = new_ast(0);
    ast->right->op = SUB;
    ast->right->left = new_ast(10);
    ast->right->right = new_ast(0);
    ast->right->right->op = MUL;
    ast->right->right->left = new_ast(-7.0);
    ast->right->right->right = new_ast(3.14);

    return ast;
}

void free_ast(ast_t *ast) {
    if (ast == NULL) {
        return;
    }
    free_ast(ast->left);
    free_ast(ast->right);
    free(ast);
}

void print_ast(ast_t *ast) {
    if (ast == NULL) {
        return;
    }
    if (ast->left == NULL && ast->right == NULL) {
        printf("%.2f", ast->val);
        return;
    }
    printf("(");
    print_ast(ast->left);
    printf(" %c ", ast->op);
    print_ast(ast->right);
    printf(")");
}

int main() {
    ast_t *ast = build_ast();
    print_ast(ast);
    printf("\n");
    float result;

    #pragma omp parallel
    #pragma omp single
    result = eval(ast);


    printf("Result: %.2f\n", result);
    free_ast(ast);
    return 0;
}
