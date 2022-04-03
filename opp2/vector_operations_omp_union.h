#ifndef INC_2_VECTOR_OPERATIONS_OMP_UNION_H
#define INC_2_VECTOR_OPERATIONS_OMP_UNION_H

#include <stdio.h>

#include "double.h"

#define SCHEDULE schedule(static)

void add_vectors_omp_union(const DOUBLE *first, const DOUBLE *second, DOUBLE *result, size_t N);

void sub_vectors_omp_union(const DOUBLE *first, const DOUBLE *second, DOUBLE *result, size_t N);

void scalar_mul_vector_omp_union(const DOUBLE *input, DOUBLE scalar, DOUBLE *result, size_t N);

#endif //INC_2_VECTOR_OPERATIONS_OMP_UNION_H
