#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <fstream>
#include <cstring>

using namespace std;

#define MOD(a, b) ((((a)%(b))+(b))%(b))

void printArray(const bool* x, int N) {
    for (int i = 0; i < N; ++i) {
        cout << x[i] << " ";
    }
    cout << std::endl;
}

void printMatrix(const bool* A, int N, int M) {
    for (int i = 0; i < N; ++i) {
        printArray(&A[i * M], M);
    }
//    cout << std::endl;
}

void fillZeroArray(double* x, int N) {
    memset(x, 0, sizeof(double) * N);
}

int countNeighboursAlive(bool* board, int N, int i, int j) {
    int counter = 0;
    counter += (int) board[(i - 1) * N + MOD((j - 1), N)] + board[(i - 1) * N + MOD((j + 1), N)] +
               board[(i - 1) * N + MOD(j, N)];
    counter += (int) board[i * N + MOD((j - 1), N)] + board[i * N + MOD((j + 1), N)];
    counter += (int) board[(i + 1) * N + MOD((j - 1), N)] + board[(i + 1) * N + MOD((j + 1), N)] +
               board[(i + 1) * N + MOD(j, N)];
    return counter;
}

void calculateLineCellsState(bool* board, bool* newBoard, int N, int i) {
    for (int j = 0; j < N; j++) {
        int currentIndex = i * N + j;
        int aliveNeighbours = countNeighboursAlive(board, N, i, j);
        if (board[currentIndex]) {
            if (aliveNeighbours < 2 || aliveNeighbours > 3) {
                newBoard[currentIndex] = false;
            } else {
                newBoard[currentIndex] = true;
            }
        } else {
            if (aliveNeighbours == 3) {
                newBoard[currentIndex] = true;
            } else {
                newBoard[currentIndex] = false;
            }
        }
    }
}

void calculateCellsState(bool* board, bool* newBoard, int N, int linesCount) {
    for (int i = 2; i < linesCount; i++) {
        calculateLineCellsState(board, newBoard, N, i);
    }
}

void calculateFirstLineCellsState(bool* board, bool* newBoard, int N) {
    calculateLineCellsState(board, newBoard, N, 1);
}

void calculateLastLineCellsState(bool* board, bool* newBoard, int N, int linesCount) {
    calculateLineCellsState(board, newBoard, N, linesCount);
}

bool areEqual(bool* a, bool* b, int N, int linesCount) {
    for (int i = 0; i < linesCount * N; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

void calculateFlags(bool* flags, bool* board, bool** history, int N, int linesCount, int historySize) {
    for (int i = 0; i < historySize; i++) {
        flags[i] = areEqual(board + N, history[i] + N, N, linesCount);
    }
}

bool compareFlags(bool* allFlags, int size, int j) {
    for (int i = 0; i < j; i++) {
        bool one = true;
        for (int k = 0; k < size; k++) {
            one = one && allFlags[k * j + i];
        }
        if (one) {
            return true;
        }
    }
    return false;
}


void run(bool* board, int Y, int rank, int size, int linesCount) {
    MPI_Request firstLineSendRequest;
    MPI_Request lastLineSendRequest;
    MPI_Request firstLineReceiveRequest;
    MPI_Request lastLineReceiveRequest;
    MPI_Request flagsRequest;
    
    int flagsSize = 8;
    bool* flags = new bool[flagsSize];
    bool* allFlags = new bool[size * flagsSize];
    int maximumIterations = 10000;
    bool** history = new bool* [maximumIterations];
    int iterations;
    for (iterations = 0; iterations < maximumIterations; iterations++) {
        MPI_Isend(board + Y, Y, MPI_C_BOOL, MOD((rank - 1), size),
                  0, MPI_COMM_WORLD, &firstLineSendRequest);
        MPI_Isend(board + Y * linesCount, Y, MPI_C_BOOL, MOD((rank + 1), size),
                  1, MPI_COMM_WORLD, &lastLineSendRequest);
        MPI_Irecv(board, Y, MPI_C_BOOL, MOD((rank - 1), size),
                  1, MPI_COMM_WORLD, &firstLineReceiveRequest);
        MPI_Irecv(board + Y * (linesCount + 1), Y, MPI_C_BOOL, MOD((rank + 1), size),
                  0, MPI_COMM_WORLD, &lastLineReceiveRequest);
    
        if (iterations == flagsSize) {
            flagsSize *= 2;
            bool* newFlags = new bool[flagsSize];
            delete[] flags;
            flags = newFlags;
            bool* newAllFlags = new bool[flagsSize * size];
            delete[] allFlags;
            allFlags = newAllFlags;
        }
        calculateFlags(flags, board, history, Y, linesCount, iterations);
//        MPI_Iallgather(flags, iterations, MPI_C_BOOL,
//                       allFlags, iterations, MPI_C_BOOL, MPI_COMM_WORLD, &flagsRequest);
        MPI_Allgather(flags, iterations, MPI_C_BOOL,
                       allFlags, iterations, MPI_C_BOOL, MPI_COMM_WORLD);
    
        bool* newBoard = new bool[Y * (linesCount + 2)];
        calculateCellsState(board, newBoard, Y, linesCount);
        MPI_Wait(&firstLineSendRequest, MPI_STATUSES_IGNORE);
        MPI_Wait(&firstLineReceiveRequest, MPI_STATUSES_IGNORE);
        calculateFirstLineCellsState(board, newBoard, Y);
        MPI_Wait(&lastLineSendRequest, MPI_STATUSES_IGNORE);
        MPI_Wait(&lastLineReceiveRequest, MPI_STATUSES_IGNORE);
        calculateLastLineCellsState(board, newBoard, Y, linesCount);
        
        
        history[iterations] = board;
        board = newBoard;

//        if (rank == 0) {
//            for (int k = 0; k < iterations + 1; k++) {
//                printMatrix(history[k], linesCount + 2, Y);
//            }
////            cout << endl;
//        }

//        MPI_Wait(&flagsRequest, MPI_STATUSES_IGNORE);
//        for (int i = 0; i < size; i++) {
//            MPI_Barrier(MPI_COMM_WORLD);
//            if (i == rank) {
////                printArray(allFlags, size * iterations);
////                printMatrix(allFlags, size, iterations);
//                cout << "flags:" << endl;
//                printArray(flags, iterations);
//                cout << "board:" << endl;
//                printMatrix(board + Y, linesCount, Y);
//                if (rank == size - 1) {
//                    cout << endl << endl;
//                }
//            }
//        }

//        if (rank == 0) {
//            printMatrix(allFlags, size, iterations);
//            cout << endl;
//        }
        
        if (compareFlags(allFlags, size, iterations)) {
            if (rank == 0) {
                cout << "Cycle. Iterations: " << iterations << endl;
            }
            break;
        }
    }
    if (rank == 0) {
        if (iterations == maximumIterations) {
            cout << "Iterations limit exceeded.";
        }
    }
    
    delete[] flags;
    delete[] allFlags;
    delete[] history;
}

void readData(bool* AFull) {
    ifstream in("src.txt");
    int X, Y;
    in >> X >> Y;
    for (int i = 0; i < X * Y; ++i) {
        in >> AFull[i];
    }
    in.close();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    ifstream in("src.txt");
    int X, Y;
    in >> X >> Y;
    in.close();
    
    int linesCount = X / size;
    int remainder = X % size;
    for (int i = 0; i < remainder; i++) {
        if (rank == i) {
            linesCount += 1;
        }
    }
    
    int* linesCountPerProcess = new int[size];
    linesCountPerProcess[rank] = linesCount;
    MPI_Allgather(&linesCount, 1, MPI_INT, linesCountPerProcess, 1, MPI_INT, MPI_COMM_WORLD);
    
    int shift = 0;
    for (int i = 0; i < size; i++) {
        if (rank == i) {
            shift = i * linesCount * Y;
            if (rank >= remainder) {
                shift += Y * remainder;
            }
        }
    }
    
    int* shifts = new int[size];
    MPI_Allgather(&shift, 1, MPI_INT, shifts, 1, MPI_INT, MPI_COMM_WORLD);

    bool* AFull;
    if (rank == 0) {
        AFull = new bool[X * Y];
        readData(AFull);
    }
    
    int* toSend = new int[size];
    for (int i = 0; i < size; i++) {
        toSend[i] = Y * linesCountPerProcess[i];
    }
    
    bool* A = new bool[Y * (linesCount + 2)];
    MPI_Scatterv(AFull, toSend, shifts, MPI_C_BOOL, A + Y, Y * linesCount, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    delete[] shifts;
    delete[] toSend;
    if (rank == 0) {
        delete[] AFull;
    }
    
    double start, end;
    start = MPI_Wtime();
    
    run(A, Y, rank, size, linesCount);
    
    end = MPI_Wtime();
    
    if (rank == 0) {
        ofstream res("res.txt", std::ios_base::app);
        cout << "Time taken: " << end - start << endl;
        res << "Time taken: " << end - start << endl;
        res.close();
    }
    
    delete[] linesCountPerProcess;
    delete[] A;
    MPI_Finalize();
    return 0;
}
