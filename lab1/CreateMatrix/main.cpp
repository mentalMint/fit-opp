#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cstring>

double fRand(double fMin, double fMax) {
    double f = (double) rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void fillMatrix(double* A, int N, int M) {
    for (int i = 0; i < N * M; i++) {
        double num = fRand(-100, 100);
        A[i] = num;
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

void writeMatrix(const double* A, int N, int M) {
    for (int i = 0; i < N; ++i) {
        writeArray(&A[i * M], M);
    }
    std::ofstream out("src.txt", std::ios_base::app);
    out << std::endl;
    out.close();
}

int main() {
    //srand(time(NULL));
    int N = 0;
    int M = 0;
    int L = 0;
    
    while (N <= 0 || M <= 0 || L <= 0) {
        printf("Matrix size:");
        if (scanf("%d%d%d", &N, &M, &L) != 3) {
            printf("3 positive integers were expected.\n");
            return EXIT_FAILURE;
        }
        if (N <= 0 || M <= 0) {
            printf("L, M and N have to be positive. Try again.\n");
        }
    }
    
    std::ofstream out("src.txt", std::ios_base::trunc);
    out << N << std::endl;
    out << M << std::endl;
    out << L << std::endl;
    
    out.close();
    
    double* A = (double*) malloc(sizeof(double) * N * M);
    double* B = (double*) malloc(sizeof(double) * L * M);
    
    fillMatrix(A, N, M);
    fillMatrix(B, M, L);
    
    writeMatrix(A, N, M);
    writeMatrix(B, M, L);
    
    free(A);
    free(B);
    return 0;
}
