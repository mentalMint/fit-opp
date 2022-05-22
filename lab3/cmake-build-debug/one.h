//
// Created by Roman on 11.04.2022.
//

#ifndef LAB3_ONE_H
#define LAB3_ONE_H


#include <iostream>
#include <cstdlib>
#include <iomanip>
#include "mpi.h"

void setArgs(int C_x, int C_y, int C_part_x, int C_part_y, int* displs, int* block_size)
{
    int id = 0;
    for (int i = 0; i < CFullY / CY; ++i)
    {
        for (int j = 0; j < CFullX / CX; ++j)
        {
            blockSize[id] = 1;
            displs[id] = j + i * (CFullX / CX * CY);
            id++;
        }
    }
}
void FillMatrix(int size_x, int size_y, double* A)
{
    for (int i = 0; i < size_y; ++i)
    {
        double* a = A + i * size_x;
        for (int j = 0; j < size_x; ++j)
        {
            //Generate random double number with 2 decimal places from [-100; 100]
            double rand = double(std::rand() % 10000) / 100;
            int sign = std::rand() % 2;
            if (sign == 1)
            {
                rand *= -1;
            }
            a[j] = rand;
        }
    }
}
void InitMatrices(int A_x, int A_y, double* A, int B_x, int B_y, double* B)
{
    FillMatrix(A_x, A_y, A);
    FillMatrix(B_x, B_y, B);
}
void SendPartMatrixA(int* coords, int send_count, int recv_count, double* A, double*
A_part, MPI_Comm column_comm, MPI_Comm row_comm)
{
    if (coords[1] == 0) //For processes from 1st column 2D lattice
    {
        MPI_Scatter(A, send_count, MPI_DOUBLE, A_part, recv_count, MPI_DOUBLE, 0,
                    column_comm); //Each column_comm has its own "root" == 0 process
    }
    MPI_Bcast(A_part, send_count, MPI_DOUBLE, 0, row_comm);//Each row_comm has its own "root" == 0 process
}
void SendPartMatrixB(int* coords, int B_x, int B_y, int B_part_x, int B_part_y, double* B,
                     double* B_part, MPI_Comm column_comm, MPI_Comm row_comm)
{
    //Creating column type
    MPI_Datatype column, column_type;
    MPI_Type_vector(B_y, B_part_x, B_x, MPI_DOUBLE, &column);
    MPI_Type_commit(&column);
    MPI_Type_create_resized(column, 0, B_part_x * sizeof(double), &column_type);
    MPI_Type_commit(&column_type);
    if (coords[0] == 0) //For processes from 1st row 2D lattice
    {
        MPI_Scatter(B, 1, column_type, B_part, B_part_x * B_part_y, MPI_DOUBLE, 0,
                    row_comm); //Each row_comm has its own "root" == 0 process
    }
    MPI_Bcast(B_part, B_part_x * B_part_y, MPI_DOUBLE, 0, column_comm); //Each column_comm has its own "root" == 0 process
    MPI_Type_free(&column);
    MPI_Type_free(&column_type);
}
void MultMatrix(int a_x, int a_y, int b_x, double* A, double* B, double* C)
{
    for (int i = 0; i < a_y; ++i)
    {
        double* c = C + i * b_x;
        for (int l = 0; l < b_x; ++l)
        {
            c[l] = 0;
        }
        for (int j = 0; j < a_x; ++j)
        {
            double a = A[i * a_x + j];
            double* b = B + j * b_x;
            for (int k = 0; k < b_x; ++k)
            {
                c[k] += a * b[k];
            }
        }
    }
}
void getFullMatrix(int size, int CFullX, int CFullY, int CX, int CY, double* C,
                   double* CFull, MPI_Comm comm2d)
{
    //Creating block type
    MPI_Datatype block, block_type;
    MPI_Type_vector(CY, CX, CFullX, MPI_DOUBLE, &block);
    MPI_Type_commit(&block);
    MPI_Type_create_resized(block, 0, CX * sizeof(double), &block_type);
    MPI_Type_commit(&block_type);
    //Set auxiliary arrays
    auto displs = new int[size];
    auto block_size = new int[size];
    setArgs(CFullX, CFullY, CX, CY, size, displs, block_size);
    //Matrix assembly
    MPI_Gatherv(C, CX * CY, MPI_DOUBLE, CFull, block_size, displs, block_type,
                0, comm2d);
    MPI_Type_free(&block_type);
    delete[] displs;
    delete[] block_size;
}
int CheckAnsw(int A_x, int A_y, int B_x, double* A, double* B, double* C)
{
    auto Right_answ = new double[A_y * B_x];
    int mistakes = 0;
    MultMatrix(A_x, A_y, B_x, A, B, Right_answ);
    for (int i = 0; i < A_y; ++i)
    {
        double* c = C + i * B_x;
        double* answ = Right_answ + i * B_x;
        for (int j = 0; j < B_x; ++j)
        {
            if (answ[j] - c[j] != 0)
            {
                mistakes++;
            }
        }
    }
    delete[]Right_answ;
    return mistakes;
}
int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "Not enough arguments!" << std::endl;
        return 0;
    }
    int dims[2] = {0, 0}, periods[2] = {0, 0};
    int coords[2], varying_coords[2];
    int reorder = 0;
    int size, rank;
    
    int size_y;
    int size_x;
    int A_x, A_y, B_x, B_y;
    double start, end;
    MPI_Comm comm2d;
    MPI_Comm row_comm;
    MPI_Comm column_comm;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //Set the dims of the lattice
    size_y = atoi(argv[1]);
    size_x = atoi(argv[2]);
    dims[0] = size_y;
    dims[1] = size_x;
    //Set the dims of the main matrices
    A_x = 5800;
    A_y = size_y * 600;
    B_y = A_x;
    B_x = size_x * 3900;
    //Set the dims of the sub matrices
    int A_part_x = A_x;
    int A_part_y = A_y / size_y;
    int B_part_x = B_x / size_x;
    int B_part_y = B_y;
    //Memory allocation for matrices
    auto A = new double[A_x * A_y];
    auto B = new double[B_x * B_y];
    auto C = new double[A_y * B_x];
    auto A_part = new double[A_part_x * A_part_y];
    auto B_part = new double[B_part_x * B_part_y];
    auto C_part = new double[A_part_y * B_part_x];
    //Creating a communicator
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm2d);
    MPI_Comm_rank(comm2d, &rank);
    MPI_Cart_coords(comm2d, rank, 2, coords);
    //Creating a subcommunicator for rows
    varying_coords[0] = 0;
    varying_coords[1] = 1;
    MPI_Cart_sub(comm2d, varying_coords, &row_comm);
    //Creating a subcommunicator for columns
    varying_coords[0] = 1;
    varying_coords[1] = 0;
    MPI_Cart_sub(comm2d, varying_coords, &column_comm);
    //Matrix initialization
    if (rank == 0)
    {
        InitMatrices(A_x, A_y, A, B_x, B_y, B);
    }
    start = MPI_Wtime();
    //Sepparating matrices
    SendPartMatrixA(coords, A_part_x * A_part_y, A_x * A_y, A, A_part, column_comm,
                    row_comm);
    SendPartMatrixB(coords, B_x, B_y, B_part_x, B_part_y, B, B_part, column_comm,
                    row_comm);
    //Multiply matrices
    MultMatrix(A_part_x, A_part_y, B_part_x, A_part, B_part, C_part);
    //C matrix assembly
    getFullMatrix(size, B_x, A_y, B_part_x, A_part_y, C_part, C, comm2d);
    end = MPI_Wtime();
    if (rank == 0)
    {
        std::cout << "+===================+" << std::endl;
        std::cout << "| Mistakes: " << CheckAnsw(A_x, A_y, B_x, A, B, C) << " |" << std::endl;
        std::cout << "+===================+" << std::endl;
        std::cout << "| Times: " << std::fixed << std::setprecision(3) << end - start << "sec.|" << std::endl;
        std::cout << "+===================+" << std::endl;
    }
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] A_part;
    delete[] B_part;
    delete[] C_part;
    MPI_Finalize();
    return 0;
}


#endif //LAB3_ONE_H
