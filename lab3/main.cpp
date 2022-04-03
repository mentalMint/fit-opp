#include <iostream>
#include <omp.h>

void multiplyMatrix(const double* A, const double* x, double* Ax, int N) {
#pragma omp single
    fillZeroArray(Ax, N);
#pragma omp for SCHEDULE
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Ax[i] += A[i * N + j] * x[j];
        }
    }
}

int main() {
    
    return 0;
}
