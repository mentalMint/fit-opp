#include <iostream>
#include <omp.h>
#include <fstream>
#include <cstdlib>
#include <cstring>

#define SCHEDULE schedule(runtime)

using namespace std;

void fillZeroArray(double* x, int N) {
    memset(x, 0, sizeof(double) * N);
}

void multiplyMatrices(const double* A, const double* B, double* C, int N) {
#pragma omp single
    fillZeroArray(C, N * N);
#pragma omp for SCHEDULE
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Ax[i] += A[i * N + j] * x[j];
        }
    }
}

void readInputData(double* A, double* B) {
    ifstream in("src.txt");
    int N;
    in >> N;
    for (int i = 0; i < N * N; ++i) {
        in >> A[i];
    }
    
    for (int i = 0; i < N * N; ++i) {
        in >> B[i];
    }

    in.close();
}

int main() {
    ifstream in("src.txt");
    int N;
    in >> N;
    in.close();
    double* A = new double[N * N];
    double* B = new double[N * N];
    
    readInputData(A, B);
    
    double* C = new double[N * N];
    multiplyMatrices(A, B, C, N);
    
    return 0;
}
