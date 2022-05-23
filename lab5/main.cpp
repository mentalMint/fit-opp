#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <fstream>
#include <cstring>
#include <pthread.h>
#include <valarray>
#include "Task.h"

#define TASKS_NUMBER size * 1000

using namespace std;

void printArray(const int* x, int N) {
    for (int i = 0; i < N; ++i) {
        cerr << x[i] << " ";
    }
    cerr << std::endl;
}

void fillZeroArray(int* x, int N) {
    memset(x, 0, sizeof(int) * N);
}

void createTaskList(Task* taskList, int iterCounter, int rank, int size) {
    for (int i = 0; i < TASKS_NUMBER; i++) {
        taskList[i].setRepeatNum(abs(50 - i % 100) * abs(rank - (iterCounter % size)) * 1000);
    }
}

typedef struct Data {
    Task* taskList;
    pthread_mutex_t mutex;
    pthread_mutex_t reduceMutex;
    pthread_cond_t cond;
    int mutexCounter;
    int rank;
    int size;
    int iterNumber;
    int currentTask;
} Data;

void threadWork(void* argument) {
    Data* data = (struct Data*) argument;
    double start, end;
    int rank = data->rank;
    int size = data->size;
    double globalSum = 0;
    double result = 0;
    int tasksToSendNumber = (TASKS_NUMBER / size) / 4;
    int* newTasks = new int[TASKS_NUMBER / 4];
    int* nullTasks = new int[tasksToSendNumber];
    
    for (int iterCounter = 0; iterCounter < data->iterNumber; iterCounter++) {
        createTaskList(data->taskList, iterCounter, rank, size);
        data->currentTask = 0;
        double sum = 0;
        start = MPI_Wtime();
        
        for (int i = 0; i < TASKS_NUMBER;) {
            pthread_mutex_lock(&data->mutex);
            data->currentTask++;
            i = data->currentTask;
            sum += sin(data->taskList[i].perform());
            pthread_mutex_unlock(&data->mutex);
        }
        
        MPI_Request sendReqs[size - 1];
        for (int i = 0; i < size; i++) {
            if (i < rank) {
                MPI_Isend(NULL, 0, MPI_C_BOOL, i, rank, MPI_COMM_WORLD, &sendReqs[i]);
            } else if (i > rank) {
                MPI_Isend(NULL, 0, MPI_C_BOOL, i, rank, MPI_COMM_WORLD, &sendReqs[i - 1]);
            }
        }
        MPI_Waitall(size - 1, sendReqs, MPI_STATUSES_IGNORE);
        
        fillZeroArray(nullTasks, tasksToSendNumber);
        MPI_Gather(nullTasks, tasksToSendNumber, MPI_INT, newTasks, tasksToSendNumber, MPI_INT, rank, MPI_COMM_WORLD);
        
        for (int i = 0; i < TASKS_NUMBER / 4; i++) {
            pthread_mutex_lock(&data->mutex);
            data->taskList[i].setRepeatNum(newTasks[i]);
            sum += sin(data->taskList[i].perform());
            pthread_mutex_unlock(&data->mutex);
        }
        end = MPI_Wtime();
    
        if (0 != pthread_mutex_lock(&data->reduceMutex)) {
            perror("");
            MPI_Finalize();
            abort();
        }
        if (data->mutexCounter < iterCounter) {
            int ret = pthread_cond_wait(&data->cond, &data->reduceMutex);
            if (0 != ret) {
                cerr << ret << endl;
                MPI_Finalize();
                abort();
            }
        }
        if (0 != pthread_mutex_unlock(&data->reduceMutex)) {
            perror("");
            MPI_Finalize();
            abort();
        }
        
        MPI_Reduce(&sum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            result += globalSum;
//            cerr << result << endl;
        }
        for (int i = 0; i < size; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (i == rank) {

                ofstream res("res.txt", std::ios_base::app);
                if (i == 0) {
                    cout << "Iteration " << iterCounter << endl;
                    res << "Iteration " << iterCounter << endl;
                }
                cout << "Rank " << rank << ". Time taken: " << end - start << endl;
                res << "Rank " << rank << ". Time taken: " << end - start << endl;
                res.close();
            }
        }
    }
    delete[] newTasks;
    delete[] nullTasks;
}

void* receive(void* argument) {
    Data* data = (struct Data*) argument;
    int rank = data->rank;
    int size = data->size;
    MPI_Request requests[size - 1];
    int tasksToSendNumber = (TASKS_NUMBER / size) / 4;
    int* tasksToSend = new int[tasksToSendNumber];
    
    for (int j = 0; j < data->iterNumber; j++) {
        for (int i = 0; i < size; i++) {
            if (i < rank) {
                MPI_Irecv(NULL, 0, MPI_C_BOOL, i, i, MPI_COMM_WORLD, &requests[i]);
            } else if (i > rank) {
                MPI_Irecv(NULL, 0, MPI_C_BOOL, i, i, MPI_COMM_WORLD, &requests[i - 1]);
            }
        }
        
        for (int i = 0; i < size - 1; i++) {
            int index;
            MPI_Status status;
            MPI_Waitany(size - 1, requests, &index, &status);
            
            fillZeroArray(tasksToSend, tasksToSendNumber);
            pthread_mutex_lock(&data->mutex);
            int limit = (TASKS_NUMBER - data->currentTask) / 3;
            for (int j = 0; j < tasksToSendNumber && j < limit; j++) {
                data->currentTask++;
                tasksToSend[j] = data->taskList[j + data->currentTask].getRepeatNum();
            }
            pthread_mutex_unlock(&data->mutex);
            
            MPI_Gather(tasksToSend, tasksToSendNumber, MPI_INT, NULL, tasksToSendNumber, MPI_INT, status.MPI_TAG,
                       MPI_COMM_WORLD);
        }
        pthread_mutex_lock(&data->reduceMutex);
        data->mutexCounter++;
        pthread_cond_signal(&data->cond);
        pthread_mutex_unlock(&data->reduceMutex);
    }
    delete[] tasksToSend;
}

void run(int rank, int size) {
//    pthread_t worker;
    pthread_t receiver;
//    pthread_t* threads[2];
    pthread_attr_t attrs;
    Data* data = new Data;
    data->taskList = new Task[TASKS_NUMBER];
    data->rank = rank;
    data->size = size;
    data->mutexCounter = 0;
    data->currentTask = -1;
    data->iterNumber = 4;
    pthread_mutexattr_t mutexattr;
    if (0 != pthread_attr_init(&attrs)) {
        perror("Cannot initialize attributes");
        MPI_Finalize();
        abort();
    }
    if (0 != pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE)) {
        perror("Error in setting attributes");
        MPI_Finalize();
        abort();
    }
    
    if (0 != pthread_mutexattr_init(&mutexattr)) {
        perror("Cannot initialize mutex attributes");
        MPI_Finalize();
        abort();
    }
    if (0 != pthread_mutex_init(&data->reduceMutex, &mutexattr)) {
        perror("Cannot initialize mutex");
        MPI_Finalize();
        abort();
    }
    if (0 != pthread_mutex_init(&data->mutex, &mutexattr)) {
        perror("Cannot initialize mutex");
        MPI_Finalize();
        abort();
    }
    
    pthread_condattr_t condattr;
    if (0 != pthread_condattr_init(&condattr)) {
        perror("Cannot initialize mutex");
        MPI_Finalize();
        abort();
    }
    if (0 != pthread_cond_init(&data->cond, &condattr)) {
        perror("Cannot initialize mutex");
        MPI_Finalize();
        abort();
    }
    
//    if (0 != pthread_create(&worker, &attrs, threadWork, data)) {
//        perror("Cannot create a thread");
//        MPI_Finalize();
//        abort();
//    }
    if (0 != pthread_create(&receiver, &attrs, receive, data)) {
        perror("Cannot create a thread");
        MPI_Finalize();
        abort();
    }
    
    threadWork(data);
    
    pthread_mutexattr_destroy(&mutexattr);
    pthread_cond_destroy(&data->cond);
    pthread_attr_destroy(&attrs);
    pthread_mutex_destroy(&data->mutex);
    pthread_mutex_destroy(&data->reduceMutex);
    
//    threads[0] = &worker;
//    threads[1] = &receiver;
//    for (int i = 1; i < 2; i++) {
//        if (0 != pthread_join(*threads[i], NULL)) {
//            perror("Cannot join a thread");
//            MPI_Finalize();
//            abort();
//        }
//    }
    if (0 != pthread_join(receiver, NULL)) {
        perror("Cannot join a thread");
        MPI_Finalize();
        abort();
    }
    
    delete[] data->taskList;
    delete data;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    run(rank, size);
    
    MPI_Finalize();
    return 0;
}
