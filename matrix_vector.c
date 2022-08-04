#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "profile.h"

#define N_SIZE 20000
#define M_SIZE 10000

typedef long long int bigint;

typedef struct {
    int matrix[N_SIZE][M_SIZE];
    int vector[M_SIZE];
} Dataset;

typedef struct {
    bigint prod_vector[N_SIZE];
} Result;

static Dataset dataset;
static Result result;

void init_dataset() {
    srand(time(NULL));

#pragma omp parallel for 
    for (int j = 0; j < M_SIZE; ++j) {
        for (int i = 0; i < N_SIZE; ++i)
            dataset.matrix[i][j] = rand() % 5;
        dataset.vector[j] = rand() % 5;
    }
}

void reset_result() {
#pragma omp parallel for
    for (int i = 0; i < N_SIZE; ++i)
        result.prod_vector[i] = 0;
}

bool result_is_same_to(Result* expected_result) {
    for (int i = 0; i < N_SIZE; ++i) {
        if (expected_result->prod_vector[i] != result.prod_vector[i])
            return false;
    }
    return true;
}

void compute_sequentially() {
    int i, j;
    bigint dot_prod;

    for (i = 0; i < N_SIZE; ++i) {
        dot_prod = 0;

        for (j = 0; j < M_SIZE; ++j)
            dot_prod += dataset.matrix[i][j] * dataset.vector[j];
        result.prod_vector[i] = dot_prod;
    }
}

void compute_with_inner_parallel_for_reduction() {
    int i, j;
    bigint dot_prod;

    for (i = 0; i < N_SIZE; ++i) {
        dot_prod = 0;

#pragma omp parallel for reduction(+: dot_prod) private(j)
        for (j = 0; j < M_SIZE; ++j)
            dot_prod += dataset.matrix[i][j] * dataset.vector[j];
        result.prod_vector[i] = dot_prod;
    }
}


void compute_with_outer_parallel_for() {
    int i, j;
    bigint dot_prod;

#pragma omp parallel for private(i, j, dot_prod)
    for (i = 0; i < N_SIZE; ++i) {
        dot_prod = 0;

        for (j = 0; j < M_SIZE; ++j)
            dot_prod += dataset.matrix[i][j] * dataset.vector[j];
        result.prod_vector[i] = dot_prod;
    }
}

void compute_with_collapsed_parallel_for_reduction() {
    int i, j;
    bigint priv_prod_vector[N_SIZE];

#pragma omp parallel private(i, j, priv_prod_vector)
    {
        for (i = 0; i < N_SIZE; ++i)
            priv_prod_vector[i] = 0;

#pragma omp for collapse(2) 
        for (i = 0; i < N_SIZE; ++i) {
            for (j = 0; j < M_SIZE; ++j)
                priv_prod_vector[i] += dataset.matrix[i][j] * dataset.vector[j];
        }

        for (i = 0; i < N_SIZE; ++i) {
#pragma omp atomic
            result.prod_vector[i] += priv_prod_vector[i];
        }
    }
}

void main() {
    init_dataset();
    reset_result();
    printf("\n\n19BCE0397 - Ritvik Gupta\n\n");

    printf("Sequential Computation\n");
    profile(compute_sequentially);
    // printf("Dot Product:\n");
    // for (int i = 0; i < N_SIZE; ++i)
    //     printf("%lld\t", result.prod_vector[i]);
    printf("\n\n");

    Result actual_result = result;
    reset_result();

    printf("Inner Parallel For Reduction Computation\n");
    profile(compute_with_inner_parallel_for_reduction);
    // printf("Dot Product:\n");
    // for (int i = 0; i < N_SIZE; ++i)
    //     printf("%lld\t", result.prod_vector[i]);
    printf("\n\n");

    if (!result_is_same_to(&actual_result)) {
        printf("\nERROR: INVALID RESULT\n");
        exit(1);
    }
    reset_result();

    printf("Outer Parallel For Reduction Computation\n");
    profile(compute_with_outer_parallel_for);
    // printf("Dot Product:\n");
    // for (int i = 0; i < N_SIZE; ++i)
    //     printf("%lld\t", result.prod_vector[i]);
    printf("\n\n");

    if (!result_is_same_to(&actual_result)) {
        printf("\nERROR: INVALID RESULT\n");
        exit(1);
    }
    reset_result();

    printf("Collapsed Parallel For Reduction Computation\n");
    profile(compute_with_collapsed_parallel_for_reduction);
    // printf("Dot Product:\n");
    // for (int i = 0; i < N_SIZE; ++i)
    //     printf("%lld\t", result.prod_vector[i]);
    printf("\n\n");

    if (!result_is_same_to(&actual_result)) {
        printf("\nERROR: INVALID RESULT\n");
        exit(1);
    }
    reset_result();
}
