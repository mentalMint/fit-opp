#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cstring>

double fRand(double fMin, double fMax) {
    double f = (double) rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void fillSymmetricMatrix(double* A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            double num = fRand(-100, 100);
            A[i * N + j] = num;
            A[j * N + i] = num;
            if (i == j) {
                A[i * N + j] += 200;
            }
        }
    }
}

void writeArray(const double* x, int N) {
    std::ofstream out("src.txt", std::ios_base::app);
    for (int i = 0; i < N; ++i) {
        out << x[i] << " ";
    }
    out << std::endl;
    out.close();
}


void writeMatrix(const double* A, int N) {
    for (int i = 0; i < N; ++i) {
        writeArray(&A[i * N], N);
    }
    std:: ofstream out("src.txt", std::ios_base::app);
    out << std::endl;
    out.close();
}

void fillZeroArray(double* x, int N) {
    memset(x, 0, sizeof(double) * N);
}

void multiplyMatrix(const double* A, const double* x, double* Ax, int N) {
    fillZeroArray(Ax, N);
    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Ax[i] += x[j] * A[i * N + j];
        }
    }
}

void fillB(const double* A, double* b, int N) {
    double* x = (double*) malloc(sizeof(double) * N);
    for (int i = 0; i < N; ++i) {
        x[i] = fRand(-100, 100);
    }
    multiplyMatrix(A, x, b, N);
    writeArray(b, N);
    writeArray(x, N);
}

int main() {
    srand(time(NULL));
    int N = 0;
    while (N <= 0) {
        printf("Matrix size N=");
        if (scanf("%d", &N) != 1) {
            printf("Positive integer was expected.\n");
            return EXIT_FAILURE;
        }
        if (N <= 0) {
            printf("N has to be positive. Try again.\n");
        }
    }
    
    std::ofstream out("src.txt", std::ios_base::trunc);
    out << N << std::endl;
    out.close();
    
    double* A = (double*) malloc(sizeof(double) * N * N);
    fillSymmetricMatrix(A, N);
    writeMatrix(A, N);
    double* b = (double*) malloc(sizeof(double) * N);
    fillB(A, b, N);
    free(A);
    free(b);
    return 0;
}
