#include<cmath>
#include<iostream>
#include<cstdlib>
#include"cuda.h"
#include"cuComplex.h"
#include<fstream>
#include "sort/util/cucpp.h" // MGPU utility classes
#include "sort/inc/mgpusort.hpp"
#include "lattice.h"

using namespace std;

__host__ __device__ int idx(int i, int j, int lda);

//__device__ int d_num_Elem = 12870; //all the diagonal elements

struct d_hamiltonian
{

    int* rows;
    int* cols;
    cuDoubleComplex* vals;
    int fulldim;
    int sectordim;
};


struct f_hamiltonian
{

    int* rows;
    int* cols;
    float* vals;
    int fulldim;
    int sectordim;
};

/*struct hamstruct
{

    int rowindex;
    int colindex;
    cuDoubleComplex value;
    int dim;
};

struct ham_sort_function
{

    __host__ __device__ bool operator()(hamstruct a, hamstruct b)
    {

        if (a.rowindex == -1) return false;
        if (b.rowindex == -1) return true;
        if (a.colindex == -1) return false;
        if (b.colindex == -1) return true;
        else return ( (a.colindex + a.rowindex*a.dim) < (b.colindex + b.rowindex*b.dim) );

    }

};*/

__host__ int GetBasis(int dim, int lattice_Size, int Sz, int basis_Position[], int basis[]);

__device__ float HOffBondX(const int si, const int bra, const float JJ);

__device__ float HOffBondY(const int si, const int bra, const float JJ);

__device__ float HDiagPart(const int bra, int lattice_Size, int3* d_Bond, const float JJ);

__host__ int* ConstructSparseMatrix(const int how_many, int* model_Type, int* lattice_Size, int** Bond, d_hamiltonian*& hamil_lancz, float* JJ, int* Sz );

__global__ void FillDiagonals(int* d_basis, int dim, int* H_rows, int* H_cols, float* H_vals, int* d_Bond, int lattice_Size, float JJ);

__global__ void FillSparse(int* d_basis_Position, int* d_basis, int dim, int* H_rows, int* H_cols, float* H_vals, int* d_Bond, const int lattice_Size, const float JJ, int* num_Elem, int index);

__global__ void FullToCOO(int num_Elem, float* H_vals, cuDoubleComplex* hamil_Values, int dim);



