#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <omp.h>

#define SCHEDULE schedule(static)

using namespace std;

void fillZeroArray(double* x, int N) {
    memset(x, 0, sizeof(double) * N);
}

double calcInnerProduct(const double* x, const double* y, int N) {
    double res = 0;
    for (int i = 0; i < N; ++i) {
        res += x[i] * y[i];
    }
    return res;
}

void calcNextX(const double* x, double alpha, const double* z, double* nextX, int N) {
#pragma omp for
    for (int i = 0; i < N; ++i) {
        nextX[i] = x[i] + alpha * z[i];
    }
}

void calcNextR(const double* r, double alpha, const double* Az, double* nextR, int N) {
#pragma omp for
    for (int i = 0; i < N; ++i) {
        nextR[i] = r[i] - alpha * Az[i];
    }
}

void calcNextZ(const double* nextR, double beta, const double* z, double* nextZ, int N) {
#pragma omp for
    for (int i = 0; i < N; ++i) {
        nextZ[i] = nextR[i] + beta * z[i];
    }
}

void assignVector(double* x, const double* y, int N) {
    memcpy(x, y, sizeof(double) * N);
}

double calcScalar(double* accumulator, const double* x, const double* y, int N) {
#pragma omp single
    *accumulator = 0;
    
    double res = 0;
#pragma omp for SCHEDULE
    for (int i = 0; i < N; ++i) {
        res += x[i] * y[i];
    }

#pragma omp critical
    {
        *accumulator += res;
    }
#pragma omp barrier
    
    double rScalar = *accumulator;
#pragma omp barrier
    
    return rScalar;
}

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


void solveLinearSystem(const double* A, const double* b, double* x, int N) {
    double bScalar = calcInnerProduct(b, b, N);

    if (bScalar == 0) {
        fillZeroArray(x, N);
        cerr << "Too good linear system" << endl;
        return;
    }
    const double eps = 0.00001 * 0.00001 * bScalar;
    double accumulator = 0;
    double* z = new double[N];
    double* r = new double[N];
    double* nextR = new double[N];
    double* Az = new double[N];

#pragma omp parallel num_threads(4)
    {
        multiplyMatrix(A, x, Az, N);
        
        calcNextR(b, 1, Az, r, N);
        
        assignVector(z, r, N);
        
        int exitConditionCounter = 0;
        int iteration = 0;
        double rScalar = calcScalar(&accumulator, r, r, N);
        
        while (exitConditionCounter < 3 && iteration < 50000) {
            ++iteration;
            if (rScalar == 0) {
                break;
            }
    
            multiplyMatrix(A, z, Az, N);
            double zAz = calcScalar(&accumulator, z, Az, N);
            
            if (zAz == 0) {
                break;
            }

            if (rScalar < eps) {
                ++exitConditionCounter;
            } else {
                exitConditionCounter = 0;
            }
            
            double alpha = rScalar / zAz;
            calcNextX(x, alpha, z, x, N);
            
            calcNextR(r, alpha, Az, nextR, N);
            
            double nextRScalar = calcScalar(&accumulator, nextR, nextR, N);
            
            double beta = nextRScalar / rScalar;
            rScalar = nextRScalar;
            
            calcNextZ(nextR, beta, z, z, N);
            
            assignVector(r, nextR, N);
        }
        
        if (iteration >= 50000) {
#pragma omp single
            cerr << "Too much iterations" << endl;
        }
        
        if (iteration < 3) {
#pragma omp single
            cerr << "Too few iterations" << endl;
        }
#pragma omp single
        cout << "Iterations: " << iteration << endl;
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
    
    double start = omp_get_wtime();
    
    double* x = new double[N];
    fillZeroArray(x, N);
    
    ofstream log("log.txt", std::ios_base::trunc);
    log.close();
    
    solveLinearSystem(A, b, x, N);
    
    double end = omp_get_wtime();
    double time = end - start;
    double measurementError = checkAnswer(x, solution, N);
    
    cout << "OpenMP. Time taken: " << time << " sec" << endl;
    cout << "Measurement error: " << measurementError << endl;
    cout << endl;
    
    ofstream out("out.txt", std::ios_base::trunc);
    out << "OpenMP. Time taken: " << time << " sec" << endl;
    out << "Measurement error: " << measurementError << endl;
    out << endl;
    out.close();
    
    ofstream res("res.txt", std::ios_base::app);
    res << time << endl;
    res.close();
    
    delete[] x;
    delete[] solution;
    delete[] b;
    delete[] A;
    return 0;
}
