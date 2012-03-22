#pragma once
#include<cmath>
#include<iostream>
#include<cstdlib>
#include"cuda.h"
#include"cuComplex.h"
#include<fstream>
#include "sort/util/cucpp.h" // MGPU utility classes
#include "sort/inc/mgpusort.hpp"
#include "lattice.h"
//#include"thrust/sort.h"
//#include"thrust/device_ptr.h"
//#include"thrust/count.h"
//#include"thrust/device_vector.h"
//#include"thrust/host_vector.h"
//#include"thrust/reduce.h"

#define WARP_SIZE 32
#define NUM_THREADS 1024
#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
#define LOG_NUM_THREADS 10
#define LOG_NUM_WARPS (LOG_NUM_THREADS - 5)
 
#define SCAN_STRIDE (WARP_SIZE + WARP_SIZE / 2 + 1)

using namespace std;

__device__ uint bfi(uint x, uint y, uint bit, uint numBits);

__host__ __device__ int idx(int i, int j, int lda);

//__device__ int d_num_Elem = 65536; //all the diagonal elements

struct d_hamiltonian
{

    int* rows;
    int* cols;
    double* vals;
    int fulldim;
    int sectordim;
};


struct f_hamiltonian
{

    int* rows;
    int* cols;
    float* vals;
    int* set;
    int fulldim;
    int sectordim;
};

/*struct ham_sort_function{

	__host__ __device__ bool operator()(hamstruct a, hamstruct b){
		if (a.rowindex == -1 || a.colindex == -1) return false;
		if (b.rowindex == -1 || b.colindex == -1) return true;

		//return ( (a.colindex == -1 || a.rowindex == -1) ? true : ( (a.colindex + a.rowindex*a.dim) < (b.colindex + b.rowindex*b.dim) ) );
		return (a.colindex + a.rowindex*a.dim) < (b.colindex + b.rowindex*b.dim);
	}

};*/


__host__ int GetBasis(int dim, int lattice_Size, int Sz, int basis_Position[], int basis[]);

__device__ float HOffBondX(const int si, const int bra, const float JJ);

__device__ float HOffBondY(const int si, const int bra, const float JJ);

__device__ float HDiagPart(const int bra, int lattice_Size, int2* d_Bond, const float JJ);

__host__ void ConstructSparseMatrix(const int how_many, int* model_Type, int* lattice_Size, int** Bond, d_hamiltonian*& hamil_lancz, float* JJ, float* h, int* Sz, int*& count_array, int device);

__global__ void FillDiagonals(int* d_basis, int dim, int* H_rows, int* H_cols, float* H_vals, int* H_set, int* d_Bond, int lattice_Size, float JJ);

__global__ void FillSparse(int* d_basis_Position, int* d_basis, int dim, int* H_rows, int* H_cols, float* H_vals, int* H_set, int* d_Bond, const int lattice_Size, const float JJ, const float h, int* num_Elem, int index);

__global__ void ScanBlocks(int* count, unsigned int* counter, int index);

__global__ void ScanBlocksFinal(int* count, unsigned int* counter, int index, int bound);

__global__ void MultiscanFirstPass(const int* values, int* inclusive, int cutoff);

__global__ void MultiscanFinal(const int* values, int* inclusive, int index);

__global__ void FullToCOO(int num_Elem, float* H_vals, double* hamil_Values, int dim);

