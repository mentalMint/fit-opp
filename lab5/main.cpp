#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <fstream>
#include <cstring>
#include <pthread.h>
#include <valarray>
#include "Task.h"

#define TASKS_NUMBER 1000

using namespace std;

void printArray(const int* x, int N) {
    for (int i = 0; i < N; ++i) {
        cerr << x[i] << " ";
    }
    cerr << std::endl;
}

void fillZeroArray(double* x, int N) {
    memset(x, 0, sizeof(double) * N);
}

void createTaskList(Task* taskList, int iterCounter, int rank, int size) {
    for (int i = 0; i < TASKS_NUMBER; i++) {
        taskList[i].setRepeatNum(abs(50 - i % 100) * abs(rank - (iterCounter % size)) * 1);
    }
}

typedef struct Data {
    Task* taskList = new Task[TASKS_NUMBER];
    MPI_Request* requests;
    pthread_mutex_t mutex;
    int rank;
    int size;
    int iterNumber = 1;
    int left = TASKS_NUMBER;
} Data;

pthread_mutex_t reduceMutex;
int counter = 1;
MPI_Request* allgatherRequests;
pthread_cond_t cond;

void* threadWork(void* argument) {
    Data* data = (struct Data*) argument;
    double start, end;
    int rank = data->rank;
    int size = data->size;
    double globalSum = 0;
    double result = 0;
    
    for (int iterCounter = 0; iterCounter < data->iterNumber; iterCounter++) {
        data->left = TASKS_NUMBER;
        createTaskList(data->taskList, iterCounter, rank, size);
        start = MPI_Wtime();
        double sum = 0;
        for (int i = 0; i < TASKS_NUMBER; i++) {
            sum += sin(data->taskList[i].perform());
            pthread_mutex_lock(&data->mutex);
            data->left--;
            pthread_mutex_unlock(&data->mutex);
        }
        MPI_Ibcast(NULL, 0, MPI_C_BOOL, rank, MPI_COMM_WORLD, &data->requests[rank]);
        MPI_Wait(&data->requests[rank], MPI_STATUS_IGNORE);
        int lefts[size];
    
//        pthread_mutex_lock(&data->mutex);
        MPI_Iallgather(&data->left, 1, MPI_INT, lefts, 1, MPI_INT, MPI_COMM_WORLD, &allgatherRequests[rank]);
//        pthread_mutex_unlock(&data->mutex);
        MPI_Wait(&allgatherRequests[rank], MPI_STATUS_IGNORE);
//        cerr << rank << " ";
//        printArray(lefts, size);
        if (0 != pthread_mutex_lock(&reduceMutex)) {
            perror("");
            abort();
        }
        if (counter != 0) {
            int stat = pthread_cond_wait(&cond,&reduceMutex);
            if (0 != stat) {
                cerr << stat << endl;
                abort();
            }
        }
        pthread_mutex_unlock(&reduceMutex);
        if (rank == 1) {
            cerr << "ass " << rank << endl;
        }
        end = MPI_Wtime();
//        cerr << "Rank " << rank << ", iteration " << iterCounter << ", time taken " << end - start << endl;
        MPI_Reduce(&sum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            result += globalSum;
        }
    }
}

void* receive(void* argument) {
    Data* data = (struct Data*) argument;
    int rank = data->rank;
    int size = data->size;
    int lefts[size];
    MPI_Request requests[size - 1];

    for (int i = 0; i < size; i++) {
        if (rank != i) {
            MPI_Ibcast(NULL, 0, MPI_C_BOOL, i, MPI_COMM_WORLD, &data->requests[i]);
        }
    }
    
    for (int i = 0; i < size - 1; i++) {
        if (i < rank) {
            requests[i] = data->requests[i];
        } else {
            requests[i] = data->requests[i + 1];
        }
    }
    
    for (int i = 0; i < size - 1; i++) {
        int index;
        MPI_Status status;
        MPI_Waitany(size - 1, requests, &index, &status);
        
//        pthread_mutex_lock(&data->mutex);
        MPI_Iallgather(&data->left, 1, MPI_INT, lefts, 1, MPI_INT, MPI_COMM_WORLD, &allgatherRequests[status.MPI_SOURCE]);
//        pthread_mutex_unlock(&data->mutex);
        
        MPI_Wait(&allgatherRequests[status.MPI_SOURCE], MPI_STATUS_IGNORE);

//        if (rank == 0) {
//            cerr << rank << " ";
//            printArray(lefts, size);
//        }
    }
    
//    pthread_mutex_lock(&reduceMutex);
//    counter = 0;
//    pthread_cond_signal(&cond);
//    pthread_mutex_unlock(&reduceMutex);
}

void run(int rank, int size) {
    pthread_t worker;
    pthread_t receiver;
    pthread_t* threads[2];
    pthread_attr_t attrs;
    Data* data = new Data;
    data->taskList = new Task[TASKS_NUMBER];
    data->rank = rank;
    data->size = size;
    data->requests = new MPI_Request[size];
    allgatherRequests = new MPI_Request[size];
    pthread_mutexattr_t mutexattr;
    if (0 != pthread_attr_init(&attrs)) {
        perror("Cannot initialize attributes");
        abort();
    }
    if (0 != pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE)) {
        perror("Error in setting attributes");
        abort();
    }
    
    if (0 != pthread_mutexattr_init(&mutexattr)) {
        perror("Cannot initialize mutex attributes");
        abort();
    }
    if (0 != pthread_mutex_init(&reduceMutex, &mutexattr)) {
        perror("Cannot initialize mutex");
        abort();
    }
    if (0 != pthread_mutex_init(&data->mutex, &mutexattr)) {
        perror("Cannot initialize mutex");
        abort();
    }
    
    if (0 != pthread_create(&worker, &attrs, threadWork, data)) {
        perror("Cannot create a thread");
        abort();
    }
    if (0 != pthread_create(&receiver, &attrs, receive, data)) {
        perror("Cannot create a thread");
        abort();
    }
    
    pthread_attr_destroy(&attrs);
    threads[0] = &worker;
    threads[1] = &receiver;
    for (auto &thread: threads) {
        if (0 != pthread_join(*thread, NULL)) {
            perror("Cannot join a thread");
            abort();
        }
    }
//    delete[] data->taskList;
    delete data;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    run(rank, size);
    
    MPI_Finalize();
    return 0;
}
