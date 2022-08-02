#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "profile.h"

#define SIZE 20000

typedef long int bigint;

typedef struct {
    int matrix[SIZE][SIZE];
    int vector[SIZE];
} Dataset;

typedef struct {
    bigint dot_prod_vector[SIZE];
} Result;

static Dataset dataset;
static Result result;

void init_dataset() {
    srand(time(NULL));

#pragma omp parallel for 
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j)
            dataset.matrix[i][j] = rand() % 10;
        dataset.vector[i] = rand() % 10;
    }

}

void reset_result() {
#pragma omp parallel for
    for (int i = 0; i < SIZE; ++i)
        result.dot_prod_vector[i] = 0;
}

bool result_is_same_to(Result* expected_result) {
    for (int i = 0; i < SIZE; ++i) {
        if (expected_result->dot_prod_vector[i] != result.dot_prod_vector[i])
            return false;
    }
    return true;
}

void compute_sequentially() {
    for (int i = 0; i < SIZE; ++i) {
        int dot_prod = 0;
        for (int j = 0; j < SIZE; ++j)
            dot_prod += dataset.matrix[i][j] * dataset.vector[j];
        result.dot_prod_vector[i] = dot_prod;
    }
}

void compute_with_inner_parallel_for_reduction() {
    for (int i = 0; i < SIZE; ++i) {
        int dot_prod = 0;

#pragma omp parallel for reduction(+: dot_prod)
        for (int j = 0; j < SIZE; ++j)
            dot_prod += dataset.matrix[i][j] * dataset.vector[j];
        result.dot_prod_vector[i] = dot_prod;
    }
}


void compute_with_outer_parallel_for() {
#pragma omp parallel for
    for (int i = 0; i < SIZE; ++i) {
        int dot_prod = 0;

        for (int j = 0; j < SIZE; ++j)
            dot_prod += dataset.matrix[i][j] * dataset.vector[j];
        result.dot_prod_vector[i] = dot_prod;
    }
}

void compute_with_collapsed_parallel_for_reduction() {
    bigint priv_dot_prod_vector[SIZE];

#pragma omp parallel private(priv_dot_prod_vector)
    {
        for (int i = 0; i < SIZE; ++i)
            priv_dot_prod_vector[i] = 0;

#pragma omp for collapse(2) 
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j)
                priv_dot_prod_vector[i] += dataset.matrix[i][j] * dataset.vector[j];
        }

        for (int i = 0; i < SIZE; ++i) {
#pragma omp atomic
            result.dot_prod_vector[i] += priv_dot_prod_vector[i];
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
    // for (int i = 0; i < SIZE; ++i)
    //     printf("%ld\t", result.dot_prod_vector[i]);
    printf("\n\n");

    Result actual_result = result;
    reset_result();

    printf("Inner Parallel For Reduction Computation\n");
    profile(compute_with_inner_parallel_for_reduction);
    // printf("Dot Product:\n");
    // for (int i = 0; i < SIZE; ++i)
    //     printf("%ld\t", result.dot_prod_vector[i]);
    printf("\n\n");

    if (!result_is_same_to(&actual_result)) {
        printf("\nERROR: INVALID RESULT\n");
        exit(1);
    }
    reset_result();

    printf("Outer Parallel For Reduction Computation\n");
    profile(compute_with_outer_parallel_for);
    // printf("Dot Product:\n");
    // for (int i = 0; i < SIZE; ++i)
    //     printf("%ld\t", result.dot_prod_vector[i]);
    printf("\n\n");

    if (!result_is_same_to(&actual_result)) {
        printf("\nERROR: INVALID RESULT\n");
        exit(1);
    }
    reset_result();

    printf("Collapsed Parallel For Reduction Computation\n");
    profile(compute_with_collapsed_parallel_for_reduction);
    // printf("Dot Product:\n");
    // for (int i = 0; i < SIZE; ++i)
    //     printf("%ld\t", result.dot_prod_vector[i]);
    printf("\n\n");

    if (!result_is_same_to(&actual_result)) {
        printf("\nERROR: INVALID RESULT\n");
        exit(1);
    }
    reset_result();
}
