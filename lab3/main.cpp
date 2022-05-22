#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include "mpi.h"

using namespace std;

void printArray(const double* x, int N) {
    for (int i = 0; i < N; ++i) {
        cout << x[i] << " ";
    }
    cout << std::endl;
}

void printMatrix(const double* A, int N, int M) {
    for (int i = 0; i < N; ++i) {
        printArray(&A[i * N], M);
    }
    cout << std::endl;
}

void readInputData(double* A, double* B) {
    ifstream in("src.txt");
    int AX, AY, BY, BX;
    in >> AY >> AX >> BX;
    BY = AX;
    for (int i = 0; i < AX * AY; ++i) {
        in >> A[i];
    }
    
    for (int i = 0; i < BX * BY; ++i) {
        in >> B[i];
    }
    
    in.close();
}

void multiplyMatrices(const double* A, double* B, double* C, int AX, int AY, int BX) {
    for (int i = 0; i < AY; ++i) {
        double* c = C + i * BX;
        for (int h = 0; h < BX; ++h) {
            c[h] = 0;
        }
        for (int j = 0; j < AX; ++j) {
            double a = A[i * AX + j];
            double* b = B + j * BX;
            for (int k = 0; k < BX; ++k) {
                c[k] += a * b[k];
            }
        }
    }
}

void sendMatrix(double* AFull, double* A, int coord, int sendCount, int recvCount,
                MPI_Comm scatterComm, MPI_Comm bcastComm, MPI_Datatype sendType) {
    if (coord == 0) {
        MPI_Scatter(AFull, sendCount, sendType, A, recvCount, MPI_DOUBLE, 0, scatterComm);
    }
    MPI_Bcast(A, sendCount, MPI_DOUBLE, 0, bcastComm);
}

void createNewType(MPI_Datatype* newTypeResized, int count, int blockLength, int stride) {
    MPI_Datatype newType;
    MPI_Type_vector(count, blockLength, stride, MPI_DOUBLE, &newType);
    MPI_Type_commit(&newType);
    MPI_Type_create_resized(newType, 0, blockLength * sizeof(double), newTypeResized);
    MPI_Type_free(&newType);
    MPI_Type_commit(newTypeResized);
}

void getFullMatrix(int CFullX, int CX, int CY, int p1, int p2, double* C,
                   double* CFull, MPI_Comm comm2d, int rank, int size) {
    MPI_Datatype blockType;
    createNewType(&blockType, CY, CX, CFullX);
    
    int* recvCounts;
    int* displs;
    if (rank == 0) {
        recvCounts = new int[size];
        memset(recvCounts, 1, size);
        displs = new int[size];
        for (int i = 0; i < p1; ++i) {
            for (int j = 0; j < p2; ++j) {
                displs[i * p2 + j] = i * (CY * p2) + j;
            }
        }
    }
    
    MPI_Gatherv(C, CX * CY, MPI_DOUBLE, CFull, recvCounts, displs, blockType,
                  0, comm2d);
    MPI_Type_free(&blockType);
    if (rank == 0) {
        delete[] displs;
        delete[] recvCounts;
    }
}

bool isRightAnswer(const double* C1, const double* C2, int M, int N) {
    for (int i = 0; i < M * N; ++i) {
        if (C1[i] != C2[i]) {
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Two arguments were expected" << endl;
        return EXIT_FAILURE;
    }
    
    char* returnValue1;
    char* returnValue2;
    
    long p1 = strtol(argv[1], &returnValue1, 10);
    long p2 = strtol(argv[2], &returnValue2, 10);
    if (returnValue1 == NULL || returnValue2 == NULL) {
        cerr << "Two arguments were expected" << endl;
        return EXIT_FAILURE;
    }
    
    if (p1 <= 0 || p2 <= 0) {
        cerr << "Two positive int arguments were expected" << endl;
        cerr << p1 << " " << p2 << " were received" << endl;
        return EXIT_FAILURE;
    }
    
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int dims[2] = {p1, p2}, periods[2] = {0, 0}, coords[2], reorder = 0;
    
    MPI_Comm comm2d;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm2d);
    MPI_Comm_rank(comm2d, &rank);
    MPI_Cart_get(comm2d, 2, dims, periods, coords);
    
    MPI_Comm lineComm;
    MPI_Comm columnComm;
    int remainDims[2];
    remainDims[0] = 0;
    remainDims[1] = 1;
    MPI_Cart_sub(comm2d, remainDims, &lineComm);
    remainDims[0] = 1;
    remainDims[1] = 0;
    MPI_Cart_sub(comm2d, remainDims, &columnComm);
    
    int AFullX, AFullY, BFullX, BFullY;
    double* AFull;
    double* BFull;
    double* CFull;
    double* CSeq;
    
    if (rank == 0) {
        ifstream in("src.txt");
        in >> AFullY;
        in >> AFullX;
        BFullY = AFullX;
        in >> BFullX;
        
        in.close();
        AFull = new double[AFullX * AFullY];
        BFull = new double[BFullX * BFullY];
        CFull = new double[AFullY * BFullX];
        readInputData(AFull, BFull);
        CSeq = new double[AFullY * BFullX];
        multiplyMatrices(AFull, BFull, CSeq, AFullX, AFullY, BFullX);
    }
    
    MPI_Bcast(&AFullX, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&AFullY, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&BFullX, 1, MPI_INT, 0, MPI_COMM_WORLD);
    BFullY = AFullX;
    
    int AX = AFullX;
    int AY = AFullY / p1;
    int BX = BFullX / p2;
    int BY = BFullY;
    
    double start, end;
    start = MPI_Wtime();
    
    double* A = new double[AX * AY];
    sendMatrix(AFull, A, coords[1], AX * AY, AX * AY, columnComm, lineComm, MPI_DOUBLE);
    
    double* B = new double[BX * BY];
    MPI_Datatype columnType;
    createNewType(&columnType, BFullY, BX, BFullX);
    sendMatrix(BFull, B, coords[0], BX * BY, BX * BY, lineComm, columnComm, columnType);
    if (rank == 0) {
        delete[] AFull;
        delete[] BFull;
    }
    
    double* C = new double[BX * AY];
    multiplyMatrices(A, B, C, AX, AY, BX);
    getFullMatrix(BFullX, BX, AY, p1, p2, C, CFull, comm2d, rank, size);
    
    end = MPI_Wtime();
    
    if (rank == 0) {
        ofstream res("res.txt", std::ios_base::app);
        if (isRightAnswer(CSeq, CFull, BFullX, AFullY)) {
            cout << "Right answer" << endl;
            res << "Right answer. ";
            
        } else {
            cout << "Wrong answer." << endl;
            res << "Wrong answer. ";
        }

//        printMatrix(CFull, AFullY, BFullX);
//        printMatrix(CSeq, AFullY, BFullX);
        
        cout << "Time taken: " << end - start << endl;
        res << "Time taken: " << end - start << endl;
        
        res.close();
        delete[] CFull;
        delete[] CSeq;
    }
    
    delete[] A;
    delete[] B;
    delete[] C;
    MPI_Finalize();
    return 0;
}