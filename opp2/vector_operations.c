#include "vector_operations.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STR_LEN 40

void copy_vector(const DOUBLE *source, DOUBLE *dest, size_t N) {
    assert(source);
    assert(dest);
    for (size_t i = 0; i < N; i++) {
        dest[i] = source[i];
    }
}

void add_vectors(const DOUBLE *first, const DOUBLE *second, DOUBLE *result, size_t N) {
    assert(first);
    assert(second);
    assert(result);
    for (size_t i = 0; i < N; i++) {
        result[i] = first[i] + second[i];
    }
}

void sub_vectors(const DOUBLE *first, const DOUBLE *second, DOUBLE *result, size_t N) {
    assert(first);
    assert(second);
    assert(result);
    for (size_t i = 0; i < N; i++) {
        result[i] = first[i] - second[i];
    }
}

void scalar_mul_vector(const DOUBLE *input, DOUBLE scalar, DOUBLE *result, size_t N) {
    assert(input);
    assert(result);
    for (size_t i = 0; i < N; i++) {
        result[i] = scalar * input[i];
    }
}

DOUBLE get_scalar_product(const DOUBLE *first, const DOUBLE *second, size_t N) {
    assert(first);
    assert(second);
    DOUBLE result = 0;
    for (size_t i = 0; i < N; i++) {
        result += first[i] * second[i];
    }
    return result;
}

void mul_matrix_and_vector(const DOUBLE *matrix, const DOUBLE *vector, DOUBLE *result, size_t N) {
    assert(matrix);
    assert(vector);
    assert(result);
    for (size_t i = 0; i < N; i++) {
        result[i] = get_scalar_product(&(matrix[i * N]), vector, N);
    }
}

DOUBLE get_vector_modulus(const DOUBLE *vector, size_t N) {
    assert(vector);
    DOUBLE result = sqrt((double ) get_scalar_product(vector, vector, N));
    return result;
}

void print_vector(FILE *fp_output, const DOUBLE *vector, size_t N) {
    assert(fp_output);
    assert(vector);
    for (size_t i = 0; i < N; i++) {
        fprintf(fp_output, "%lf ", vector[i]);
    }
    fprintf(fp_output,"\n");
}
