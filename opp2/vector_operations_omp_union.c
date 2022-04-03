#include "vector_operations_omp_union.h"

#include <assert.h>

void add_vectors_omp_union(const DOUBLE *first, const DOUBLE *second, DOUBLE *result, size_t N) {
    assert(first);
    assert(second);
    assert(result);
#pragma omp for SCHEDULE
    for (size_t i = 0; i < N; i++) {
        result[i] = first[i] + second[i];
    }
}

void sub_vectors_omp_union(const DOUBLE *first, const DOUBLE *second, DOUBLE *result, size_t N) {
    assert(first);
    assert(second);
    assert(result);
#pragma omp for
    for (size_t i = 0; i < N; i++) {
        result[i] = first[i] - second[i];
    }
}

void scalar_mul_vector_omp_union(const DOUBLE *input, DOUBLE scalar, DOUBLE *result, size_t N) {
    assert(input);
    assert(result);
#pragma omp for SCHEDULE
    for (size_t i = 0; i < N; i++) {
        result[i] = scalar * input[i];
    }
}
