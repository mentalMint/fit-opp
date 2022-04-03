#ifndef INC_2_VECTOR_OPERATIONS_H
#define INC_2_VECTOR_OPERATIONS_H

#include <stddef.h>
#include <stdio.h>

#include "double.h"

void copy_vector(const DOUBLE *source, DOUBLE *dest, size_t N);

void add_vectors(const DOUBLE *first, const DOUBLE *second, DOUBLE *result, size_t N);

void sub_vectors(const DOUBLE *first, const DOUBLE *second, DOUBLE *result, size_t N);

void scalar_mul_vector(const DOUBLE *input, DOUBLE scalar, DOUBLE *result, size_t N);

DOUBLE get_scalar_product(const DOUBLE *first, const DOUBLE *second, size_t N);

void mul_matrix_and_vector(const DOUBLE *matrix, const DOUBLE *vector, DOUBLE *result, size_t N);

DOUBLE get_vector_modulus(const DOUBLE *vector, size_t N);

void print_vector(FILE *fp_output, const DOUBLE *vector, size_t N);

DOUBLE *get_vector_from_binary_file(FILE *input_file, size_t *size);

DOUBLE *get_vector_from_file(FILE *input_file, size_t *size);

#endif //INC_2_VECTOR_OPERATIONS_H
