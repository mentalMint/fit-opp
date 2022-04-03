#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <sys/times.h>
#include <unistd.h>
#include <cstring>

using namespace std;

void printArray(const double* x, int N) {
    for (int i = 0; i < N; ++i) {
        std::cout << x[i] << " ";
    }
    std::cout << std::endl;
}

void writeArray(const double* x, int N) {
    std::ofstream out("out.txt", std::ios_base::app);
    for (int i = 0; i < N; ++i) {
        out << x[i] << " ";
    }
    out << std::endl;
    out.close();
}

void printMatrix(const double* A, int N) {
    for (int i = 0; i < N; ++i) {
        printArray(&A[i * N], N);
    }
    std::cout << std::endl;
}

void writeMatrix(const double* A, int N) {
    for (int i = 0; i < N; ++i) {
        writeArray(&A[i * N], N);
    }
    std::ofstream out("out.txt", std::ios_base::app);
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

double calcInnerProduct(const double* x, const double* y, int N) {
    double res = 0;
    for (int i = 0; i < N; ++i) {
        res += x[i] * y[i];
    }
    return res;
}

double calcBeta(double rScalar, const double* nextR, int N) {
    if (rScalar == 0) {
        std::cerr << "null denominator" << std::endl;
        abort();
    }
    double tmp = calcInnerProduct(nextR, nextR, N);
    double beta = tmp / rScalar;
    return beta;
}

void calcNextX(double* x, double alpha, const double* z, double* nextX, int N) {
    for (int i = 0; i < N; ++i) {
        nextX[i] = x[i] + alpha * z[i];
    }
}

void calcNextR(const double* r, double alpha, const double* Az, double* nextR, int N) {
    for (int i = 0; i < N; ++i) {
        nextR[i] = r[i] - alpha * Az[i];
    }
}

void calcNextZ(const double* nextR, double beta, const double* z, double* nextZ, int N) {
    for (int i = 0; i < N; ++i) {
        nextZ[i] = nextR[i] + beta * z[i];
    }
}

double calcVectorNorm(double* x, int N) {
    return sqrt(calcInnerProduct(x, x, N));
}

void assignVector(double* x, const double* y, int N) {
    memcpy(x, y, sizeof(double) * N);
}

bool isZeroVector(const double* x, int N) {
    bool zeroVector = true;
    for (int i = 0; i < N; ++i) {
        if (x[i] != 0) {
            zeroVector = false;
        }
    }
    return zeroVector;
}

void solveLinearSystem(double* A, double* b, double* x, int N) {
    double bScalar = calcInnerProduct(b, b, N);
    if (bScalar == 0) {
        fillZeroArray(x, N);
        cerr << "Too good linear system" << endl;
        return;
    }
    
    double* Az = new double[N];
    multiplyMatrix(A, x, Az, N);
    
    double* r = new double[N];
    calcNextR(b, 1, Az, r, N);
    
    double* z = new double[N];
    assignVector(z, r, N);
    
    const double eps = 0.00001 * 0.00001 * bScalar;
    int exitConditionCounter = 0;
    int iteration = 0;
    double* nextR = new double[N];
    double rScalar = calcInnerProduct(r, r, N);
    
    while (exitConditionCounter < 3 && iteration < 50000) {
        ++iteration;
    
        if (rScalar == 0) {
            break;
        }
    
        multiplyMatrix(A, z, Az, N);
        
        double zAz = calcInnerProduct(Az, z, N);
        if (zAz == 0) {
            break;
        }
        
//        cerr << toCompare << endl;
    
        if (rScalar < eps) {
            ++exitConditionCounter;
        } else {
            exitConditionCounter = 0;
        }
        
        double alpha = rScalar / zAz;
        calcNextX(x, alpha, z, x, N);
        
        calcNextR(r, alpha, Az, nextR, N);
        
        double nextRScalar = calcInnerProduct(nextR, nextR, N);
    
        double beta = nextRScalar / rScalar;
        rScalar = nextRScalar;
        
        calcNextZ(nextR, beta, z, z, N);

        assignVector(r, nextR, N);
    }
    
    if (iteration >= 50000) {
        cerr << "Too much iterations" << endl;
    }
    
    if (iteration < 3) {
        cerr << "Too few iterations" << endl;
    }
    
    delete[] Az;
    delete[] nextR;
    delete[] r;
    delete[] z;
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

int main() {
    std::ifstream in("src.txt");
    int N = 0;
    in >> N;
    in.close();
    
    double* A = new double[N * N];
    double* b = new double[N];
    double* solution = new double[N];
    readInputData(A, b, solution);
    
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    
    double* x = new double[N];
    fillZeroArray(x, N);
    
    solveLinearSystem(A, b, x, N);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    
    double measurementError = checkAnswer(x, solution, N);
    
    double time = end.tv_sec - start.tv_sec
                  + 0.000000001 * (end.tv_nsec - start.tv_nsec);

    std::ofstream out("out.txt", std::ios_base::trunc);
    out.close();

    cout << "Sequential. Time taken: " << time << " sec" << endl;
    cout << "Measurement error: " << measurementError << endl;
//    printArray(x, N);
    cout << endl;
    
    out.open("out.txt",  std::ios_base::app);
    out << "Sequential. Time taken: " << time << " sec" << endl;
    out << "Measurement error: " << measurementError << endl;
    
//    writeArray(x, N);
    out << endl;
    out.close();
    
    ofstream res("res.txt", std::ios_base::app);
    res << time << " ; ";
    res.close();
    
    delete[] solution;
    delete[] x;
    delete[] b;
    delete[] A;
    return 0;
}
