#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "profile.h"

#define L_SIZE 500
#define M_SIZE 500
#define N_SIZE 500

typedef long long int bigint;

typedef struct {
    int matrix_a[L_SIZE][M_SIZE];
    int matrix_b[M_SIZE][N_SIZE];
} Dataset;

typedef struct {
    bigint prod_matrix[L_SIZE][N_SIZE];
} Result;

static Dataset dataset;
static Result result;

void init_dataset() {
    srand(time(NULL));
    int i, j, k;

#pragma omp parallel for collapse(2) private(i, j)
    for (i = 0; i < L_SIZE; ++i) {
        for (j = 0; j < M_SIZE; ++j)
            dataset.matrix_a[i][j] = rand() % 5;
    }

#pragma omp parallel for collapse(2) private(j, k)
    for (j = 0; j < M_SIZE; ++j) {
        for (k = 0; k < N_SIZE; ++k)
            dataset.matrix_b[j][k] = rand() % 5;
    }
}

void reset_result() {
    int i, k;

#pragma omp parallel for collapse(2) private(i, k)
    for (i = 0; i < L_SIZE; ++i)
        for (k = 0; k < N_SIZE; ++k)
            result.prod_matrix[i][k] = 0;
}

void print_result() {
    printf("Dot Product:\n");
    for (int i = 0; i < L_SIZE; ++i) {
        for (int k = 0; k < N_SIZE; ++k)
            printf("%lld\t", result.prod_matrix[i][k]);
        printf("\n");
    }
    printf("\n");
}

void assert_result_equals(Result* expected_result) {
    for (int i = 0; i < L_SIZE; ++i)
        for (int k = 0; k < N_SIZE; ++k)
            if (expected_result->prod_matrix[i][k] != result.prod_matrix[i][k]) {
                printf("\nAssertion Failed: Invalid Result Matrix\n");
                exit(1);
            }
}

void compute_sequentially() {
    int i, j, k;
    bigint dot_prod;

    for (i = 0; i < L_SIZE; ++i) {
        for (k = 0; k < N_SIZE; ++k) {
            dot_prod = 0;

            for (j = 0; j < M_SIZE; ++j)
                dot_prod += dataset.matrix_a[i][j] * dataset.matrix_b[j][k];
            result.prod_matrix[i][k] = dot_prod;
        }
    }
}

void compute_with_inner_parallel_for_reduction() {
    int i, j, k;
    bigint dot_prod;

    for (i = 0; i < L_SIZE; ++i) {
        for (k = 0; k < N_SIZE; ++k) {
            dot_prod = 0;

#pragma omp parallel for reduction(+: dot_prod) private(j)
            for (j = 0; j < M_SIZE; ++j)
                dot_prod += dataset.matrix_a[i][j] * dataset.matrix_b[j][k];
            result.prod_matrix[i][k] = dot_prod;
        }
    }
}

void compute_with_middle_parallel_for() {
    int i, j, k;
    bigint dot_prod;

    for (i = 0; i < L_SIZE; ++i) {
#pragma omp parallel for private(j, k, dot_prod)
        for (k = 0; k < N_SIZE; ++k) {
            dot_prod = 0;

            for (j = 0; j < M_SIZE; ++j)
                dot_prod += dataset.matrix_a[i][j] * dataset.matrix_b[j][k];
            result.prod_matrix[i][k] = dot_prod;
        }
    }
}

void compute_with_outer_parallel_for() {
    int i, j, k;
    bigint dot_prod;

#pragma omp parallel for private(i, j, k, dot_prod)
    for (i = 0; i < L_SIZE; ++i) {
        for (k = 0; k < N_SIZE; ++k) {
            dot_prod = 0;

            for (j = 0; j < M_SIZE; ++j)
                dot_prod += dataset.matrix_a[i][j] * dataset.matrix_b[j][k];
            result.prod_matrix[i][k] = dot_prod;
        }
    }
}

void compute_with_half_collapsed_parallel_for() {
    int i, j, k;
    bigint dot_prod;

#pragma omp parallel for private(i, j, k, dot_prod) collapse(2)
    for (i = 0; i < L_SIZE; ++i) {
        for (k = 0; k < N_SIZE; ++k) {
            dot_prod = 0;

            for (j = 0; j < M_SIZE; ++j)
                dot_prod += dataset.matrix_a[i][j] * dataset.matrix_b[j][k];
            result.prod_matrix[i][k] = dot_prod;
        }
    }
}

void main() {
    init_dataset();
    reset_result();
    printf("\n\n19BCE0397 - Ritvik Gupta\n\n");

    printf("Sequential Computation\n");
    profile(compute_sequentially);
    // print_result();
    printf("\n");

    Result actual_result = result;
    reset_result();

    printf("Inner Parallel For Reduction Computation\n");
    profile(compute_with_inner_parallel_for_reduction);
    // print_result();
    printf("\n");

    assert_result_equals(&actual_result);
    reset_result();

    printf("Midlle Parallel For Reduction Computation\n");
    profile(compute_with_middle_parallel_for);
    // print_result();
    printf("\n");

    assert_result_equals(&actual_result);
    reset_result();

    printf("Outer Parallel For Reduction Computation\n");
    profile(compute_with_outer_parallel_for);
    // print_result();
    printf("\n");

    assert_result_equals(&actual_result);
    reset_result();

    printf("Half Collapsed Parallel For Reduction Computation\n");
    profile(compute_with_half_collapsed_parallel_for);
    // print_result();
    printf("\n");

    assert_result_equals(&actual_result);
    reset_result();
}
