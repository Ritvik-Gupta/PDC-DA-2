#ifndef SYMBOL_profile_1659383329
#define SYMBOL_profile_1659383329

#include <stdio.h>
#include <omp.h>

void profile(void (*function_to_profile)()) {
    double start_time, end_time;

    printf("Starting Profile ...\n");

    start_time = omp_get_wtime();
    function_to_profile();
    end_time = omp_get_wtime();

    printf("Profile Ended in %15.15f seconds\n", end_time - start_time);
}

#endif
