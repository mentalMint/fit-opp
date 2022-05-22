#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>

void fillMatrix(bool* A, int N, int M) {
    A[1] = true;
    A[M + 2] = true;
    A[2 * M] = true;
    A[2 * M + 1] = true;
    A[2 * M + 2] = true;
}

void writeArray(const bool* x, int N) {
    std::ofstream out("src.txt", std::ios_base::app);
    for (int i = 0; i < N; ++i) {
        out << x[i] << " ";
    }
    out << std::endl;
    out.close();
}

void writeMatrix(const bool* A, int N, int M) {
    for (int i = 0; i < N; ++i) {
        writeArray(&A[i * M], M);
    }
    std::ofstream out("src.txt", std::ios_base::app);
    out << std::endl;
    out.close();
}

int main() {
    int N = 0;
    int M = 0;
    
    while (N <= 0 || M <= 0) {
        printf("Matrix size:");
        if (scanf("%d%d", &N, &M) != 2) {
            printf("2 positive integers were expected.\n");
            return EXIT_FAILURE;
        }
        if (N <= 0 || M <= 0) {
            printf("M and N have to be positive. Try again.\n");
        }
    }
    
    std::ofstream out("src.txt", std::ios_base::trunc);
    out << N << std::endl;
    out << M << std::endl;
    
    out.close();
    
    bool* A = (bool*) calloc(N * M, sizeof(bool));
    
    fillMatrix(A, N, M);
    writeMatrix(A, N, M);
    
    free(A);
    return 0;
}
