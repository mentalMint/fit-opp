#include <iostream>
#include <cstdlib>
#include <cmath>
#include <mpi.h>
#include <fstream>
#include <cstring>

using namespace std;

void writeArray(const double* x, int N) {
    std::ofstream out("out.txt", std::ios_base::app);
    for (int i = 0; i < N; ++i) {
        out << x[i] << " ";
    }
    out << std::endl;
    out.close();
}

void printArray(const double* x, int N) {
    for (int i = 0; i < N; ++i) {
        cout << x[i] << " ";
    }
    cout << std::endl;
}

void printMatrix(const double* A, int N) {
    for (int i = 0; i < N; ++i) {
        printArray(&A[i * N], N);
    }
    cout << std::endl;
}

void readInputData(double* AFull, double* b, double* result) {
    ifstream in("src.txt");
    int N;
    in >> N;
    for (int i = 0; i < N * N; ++i) {
        in >> AFull[i];
    }
    
    for (int i = 0; i < N; ++i) {
        in >> b[i];
    }
    
    for (int i = 0; i < N; ++i) {
        in >> result[i];
    }
    in.close();
}

void fillZeroArray(double* x, int N) {
    memset(x, 0, sizeof(double) * N);
}

void multiplyMatrix(const double* A, const double* x, double* Ax, int N, int M) {
    fillZeroArray(Ax, M);
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            Ax[i] += x[j] * A[i * N + j];
        }
    }
}

double calcInnerProduct(const double* x, const double* y, int N) {
    double res = 0;
    for (int i = 0; i < N; ++i) {
        res += x[i] * y[i];
    }
    return res;
}

void calcFullInnerProduct(const double* x, const double* y, double* result, int linesCount) {
    double innerProduct = calcInnerProduct(x, y, linesCount);
    MPI_Allreduce(&innerProduct, result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

void calcNextX(const double* x, double alpha, const double* z, double* nextX, int linesCount) {
    for (int i = 0; i < linesCount; ++i) {
        nextX[i] = x[i] + alpha * z[i];
    }
}

void calcNextR(const double* r, double alpha, const double* Az, double* nextR, int linesCount) {
    for (int i = 0; i < linesCount; ++i) {
        nextR[i] = r[i] - alpha * Az[i];
    }
}

void calcNextZ(const double* nextR, double beta, const double* z, double* nextZ, int linesCount) {
    for (int i = 0; i < linesCount; ++i) {
        nextZ[i] = nextR[i] + beta * z[i];
    }
}


void assignVector(double* x, const double* y, int N) {
    memcpy(x, y, sizeof(double) * N);
}

void getZFull(double* z, double* zFull, int N, const int* linesCountPerProcess, const int* vectorShifts, int linesCount) {
    MPI_Allgatherv(z, linesCount, MPI_DOUBLE, zFull, linesCountPerProcess, vectorShifts, MPI_DOUBLE, MPI_COMM_WORLD);
}

double checkAnswer(double* x, double* solution, int N) {
    double measurementError = 0;
    for (int i = 0; i < N; ++i) {
        double difference = abs(solution[i] - x[i]);
        if (difference > measurementError) {
            measurementError = difference;
        }
    }
    return measurementError;
}

void solveLinearSystem(double* A, double* b, double* xFull,
                       int N, int size, int rank, int linesCount,
                       const int* linesCountPerProcess, const int* vectorShifts) {
    double bScalar;
    calcFullInnerProduct(b, b, &bScalar, linesCount);
    
    if (bScalar == 0) {
        fillZeroArray(xFull, N);
        cerr << "Too good linear system" << endl;
        return;
    }
    
    double* Az = new double[linesCount];
    multiplyMatrix(A, xFull, Az, N, linesCount);
    
    double* r = new double[linesCount];
    calcNextR(b, 1, Az, r, linesCount);
    
    double* z = new double[linesCount];
    assignVector(z, r, linesCount);
    
    double* zFull = new double[N];
    getZFull(z, zFull, N, linesCountPerProcess, vectorShifts, linesCount);
    
    double* x = new double[linesCount];
//    MPI_Scatterv(xFull, linesCountPerProcess, vectorShifts, MPI_DOUBLE, x, linesCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    fillZeroArray(x, linesCount);

    int iteration = 0;
    const double eps = 0.00001 * 0.00001 * bScalar;
    int exitConditionCounter = 0;
    double* nextR = new double[linesCount];
    double rScalar;
    calcFullInnerProduct(r, r, &rScalar, linesCount);
    
    while (exitConditionCounter < 3 && iteration < 50000) {
        ++iteration;
        
        if (rScalar == 0) {
            break;
        }
        
        multiplyMatrix(A, zFull, Az, N, linesCount);
        
        double zAz;
        calcFullInnerProduct(Az, z, &zAz, linesCount);
        if (zAz == 0) {
            break;
        }
        
        if (rScalar < eps) {
            ++exitConditionCounter;
        } else {
            exitConditionCounter = 0;
        }
        
        double alpha = rScalar / zAz;
        calcNextX(x, alpha, z, x, linesCount);

        calcNextR(r, alpha, Az, nextR, linesCount);
        
        double nextRScalar;
        calcFullInnerProduct(nextR, nextR, &nextRScalar, linesCount);
        
        double beta = nextRScalar / rScalar;
        rScalar = nextRScalar;
    
        assignVector(r, nextR, linesCount);
    
        calcNextZ(r, beta, z, z, linesCount);
        
        getZFull(z, zFull, N, linesCountPerProcess, vectorShifts, linesCount);
    }
    
    if (iteration >= 50000 && rank == 0) {
        cerr << "Too much iterations" << endl;
    }
    
    if (iteration < 3 && rank == 0) {
        cerr << "Too few iterations" << endl;
    }
    
    MPI_Gatherv(x, linesCount, MPI_DOUBLE, xFull, linesCountPerProcess, vectorShifts, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    delete[] Az;
    delete[] nextR;
    delete[] zFull;
    delete[] x;
    delete[] r;
    delete[] z;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    ifstream in("src.txt");
    int N;
    in >> N;
    in.close();
    
    int linesCount = N / size;
    int remainder = N % size;
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
            shift = i * linesCount * N;
            if (rank >= remainder) {
                shift += N * remainder;
            }
        }
    }
    
    int* shifts = new int[size];
    MPI_Allgather(&shift, 1, MPI_INT, shifts, 1, MPI_INT, MPI_COMM_WORLD);
    
    int vectorShift = shift / N;
    int* vectorShifts = new int[size];
    MPI_Allgather(&vectorShift, 1, MPI_INT, vectorShifts, 1, MPI_INT, MPI_COMM_WORLD);
    
    double* AFull;
    double* bFull;
    double* solution;
    
    if (rank == 0) {
        AFull = new double[N * N];
        bFull = new double[N];
        solution = new double[N];
        readInputData(AFull, bFull, solution);
    }
    
    int* toSend = new int[size];
    for (int i = 0; i < size; i++) {
        toSend[i] = N * linesCountPerProcess[i];
    }
    
    double* A = new double[N * linesCount];
    MPI_Scatterv(AFull, toSend, shifts, MPI_DOUBLE, A, N * linesCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    delete[] shifts;
    delete[] toSend;
    
    double* b = new double[linesCount];
    MPI_Scatterv(bFull, linesCountPerProcess, vectorShifts, MPI_DOUBLE, b, linesCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        delete[] bFull;
        delete[] AFull;
    }
    
    double start = MPI_Wtime();
    
    double* xFull = new double[N];
    fillZeroArray(xFull, N);
    
    solveLinearSystem(A, b, xFull, N, size, rank, linesCount, linesCountPerProcess, vectorShifts);
    
    double end = MPI_Wtime();
    
    if (rank == 0) {
        double measurementError = checkAnswer(xFull, solution, N);
        delete[] solution;
        
        cout << "MPI_1. Time taken: " << end - start << " sec" << endl;
        cout << "Measurement error: " << measurementError << endl;
        
        ofstream out("out.txt", std::ios_base::app);
        out << "MPI_1. Time taken: " << end - start << " sec" << endl;
        out << "Measurement error: " << measurementError << endl;
        out << endl;
        out.close();
    
        ofstream res("res.txt", std::ios_base::app);
        res << end - start << " ; ";
        res.close();
    }
    
    if (rank == 0) {
//        printArray(xFull, N);
//        writeArray(xFull, N);
        cout << endl;
    }
    
    delete[] vectorShifts;
    delete[] linesCountPerProcess;
    delete[] A;
    delete[] xFull;
    delete[] b;
    
    MPI_Finalize();
    return 0;
}
