#include<cmath>
#include <iostream>
#include <cstdlib>
#include "cuda.h"
#include "cuComplex.h"
#include <fstream>
#include "sort/util/cucpp.h" // MGPU utility classes
#include "sort/inc/mgpusort.hpp"
#include "lattice.h"
#include "thrust/device_ptr.h"
#include "thrust/reduce.h"

using namespace std;

//__host__ __device__ int idx(int i, int j, int lda);

struct parameters
{
    int model_type;
    int Sz;
    int nsite;
    float J1;
    float J2;
};

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

__host__ void ConstructSparseMatrix(const int how_many, int** Bond, d_hamiltonian*& hamil_lancz, parameters* data, int*& count_array, int device);

__global__ void FullToCOO(int num_Elem, float* H_vals, double* hamil_Values, int dim);

//--------Declarations of Hamiltonian functions for Heisenberg Model--------------

__device__ float HOffBondXHeisenberg(const int si, const int bra, const float JJ);

__device__ float HOffBondYHeisenberg(const int si, const int bra, const float JJ);

__device__ float HDiagPartHeisenberg(const int bra, int lattice_Size, int3* d_Bond, const float JJ);

__global__ void FillDiagonalsHeisenberg(int* d_basis, f_hamiltonian H, int* d_Bond, parameters data);

__global__ void FillSparseHeisenberg(int* d_basis_Position, int* d_basis, f_hamiltonian H, int* d_Bond, parameters data, int offset);

//------Declarations of Hamiltonian functions for XY Model -------------------

__device__ float HOffBondXXY(const int si, const int bra, const float JJ);

__device__ float HOffBondYXY(const int si, const int bra, const float JJ);

__device__ float HDiagPartXY(const int bra, int lattice_Size, int3* d_Bond, const float JJ);

__global__ void FillDiagonalsXY(int* d_basis, f_hamiltonian H, int* d_Bond, parameters data);

__global__ void FillSparseXY(int* d_basis_Position, int* d_basis, f_hamiltonian H, int* d_Bond, parameters data, int offset);

//--------Declarations of Hamiltonian functions for transverse field Ising Model-------

__device__ float HOffBondXTFI(const int si, const int bra, const float JJ);

__device__ float HOffBondYTFI(const int si, const int bra, const float JJ);

__device__ float HDiagPartTFI(const int bra, int lattice_Size, int2* d_Bond, const float JJ);

__global__ void FillDiagonalsTFI(int* d_basis, f_hamiltonian H, int* d_Bond, parameters data);

__global__ void FillSparseTFI(int* d_basis_Position, int* d_basis, f_hamiltonian H, int* d_Bond, parameters data, int offset);

