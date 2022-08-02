#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "profile.h"

#define VEC_SIZE 200000000

typedef struct {
    int vector_a[VEC_SIZE], vector_b[VEC_SIZE];
} Vectors;

typedef struct {
    long long int dot_prod;
} DotProdResult;

static Vectors dataset;
static DotProdResult result;

void init_dataset() {
    srand(time(NULL));

#pragma omp parallel for 
    for (int i = 0; i < VEC_SIZE; ++i) {
        dataset.vector_a[i] = rand() % 10;
        dataset.vector_b[i] = rand() % 10;
    }
}

void reset_result() {
    srand(time(NULL));

#pragma omp parallel for 
    for (int i = 0; i < VEC_SIZE; ++i) {
        dataset.vector_a[i] = rand() % 10;
        dataset.vector_b[i] = rand() % 10;
    }
}

void compute_sequentially() {
    long long int dot_prod = 0;

    for (int i = 0; i < VEC_SIZE; ++i)
        dot_prod += dataset.vector_a[i] * dataset.vector_b[i];

    result.dot_prod = dot_prod;
}

void compute_with_parallel_for_reduction() {
    long long int dot_prod = 0;

#pragma omp parallel for reduction(+: dot_prod)
    for (int i = 0; i < VEC_SIZE; ++i)
        dot_prod += dataset.vector_a[i] * dataset.vector_b[i];

    result.dot_prod = dot_prod;
}

void compute_with_simd_reduction() {
    long long int dot_prod = 0;

#pragma omp simd reduction(+: dot_prod)
    for (int i = 0; i < VEC_SIZE; ++i)
        dot_prod += dataset.vector_a[i] * dataset.vector_b[i];

    result.dot_prod = dot_prod;
}

void main() {
    init_dataset();
    reset_result();
    printf("\n\n19BCE0397 - Ritvik Gupta\n\n");

    printf("Sequential Computation\n");
    profile(compute_sequentially);
    printf("Dot Product: %lld\n", result.dot_prod);
    printf("\n");

    printf("Parallel For Reduction Computation\n");
    profile(compute_with_parallel_for_reduction);
    printf("Dot Product: %lld\n", result.dot_prod);
    printf("\n");

    printf("SIMD Reduction Computation\n");
    profile(compute_with_simd_reduction);
    printf("Dot Product: %lld\n", result.dot_prod);
    printf("\n");
}
