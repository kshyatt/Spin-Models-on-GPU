#include<cmath>
#include<iostream>
#include<cstdlib>
#include"cuda.h"
#include<limits.h>
#include"cuComplex.h"
#include<fstream>
#include"thrust/sort.h"
#include"thrust/device_ptr.h"
#include"thrust/device_vector.h"
#include"thrust/reduce.h"

using namespace thrust;

__host__ __device__ int idx(int i, int j, int lda);

__device__ int d_num_Elem = 0; //all the diagonal elements

struct hamstruct{

	int rowindex;
        int colindex;
	cuDoubleComplex value;
        int dim;
};

struct ham_sort_function{

        __host__ __device__ bool operator()(hamstruct a, hamstruct b){

                return ( (a.colindex + a.rowindex*a.dim) < (b.colindex + b.rowindex*b.dim) );
        
        }

};

__host__ int GetBasis(int dim, int lattice_Size, int Sz, int basis_Position[], int basis[]);

__device__ cuDoubleComplex HOffBondX(const int si, const int bra, const double JJ);

__device__ cuDoubleComplex HOffBondY(const int si, const int bra, const double JJ);

__device__ cuDoubleComplex HDiagPart(const int bra, int lattice_Size, int3* d_Bond, const double JJ);

__host__ int ConstructSparseMatrix(int model_Type, int lattice_Size, int* Bond, cuDoubleComplex* hamil_Values, int* hamil_PosRow, int* hamil_PosCol, int* vdim, double JJ, int Sz);

__global__ void FillSparse(int* d_basis_Position, int* d_basis, int dim, cuDoubleComplex* H_vals, int2* H_pos, int* d_Bond, int lattice_Size, const double JJ);

__global__ void CompressSparse(const cuDoubleComplex* H_vals, const int2* H_pos, hamstruct* H_sort, int d_dim, const int lattice_Size, const int num_Elem);

__global__ void FullToCOO(int num_Elem, hamstruct* H_sort, cuDoubleComplex* hamil_Values, int* hamil_PosRow, int* hamil_PosCol, int dim);

__global__ void Copy(int* thrust_ptr, int2* H_pos, int lattice_Size, int vdim);
