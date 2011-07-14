#include<cmath>
#include<iostream>
#include<cstdlib>
#include<vector>
#include<stdio.h>
#include"cuda.h"
#include"cuComplex.h"
#include<fstream>
#include"thrust/sort.h"
#include"thrust/device_ptr.h"

using namespace thrust;

__host__ __device__ long idx(long i, long j, long lda){

	return ( j + (i*lda));

}

__device__ long d_num_Elem;

struct hamstruct{

	long rowindex;
        long colindex;
	cuDoubleComplex value;
        long dim;

	/*bool operator>(const hamstruct& rhs)const{return ( (rowindex + colindex*dim) > (rhs.rowindex + rhs.colindex*rhs.dim) ) ;};
	bool operator<(const hamstruct& rhs)const{return ( (rowindex + colindex*dim) < (rhs.rowindex + rhs.colindex*rhs.dim) ) ;};
	bool operator>=(const hamstruct& rhs)const{return ( (rowindex + colindex*dim) >= (rhs.rowindex + rhs.colindex*rhs.dim) ) ;};
	bool operator<=(const hamstruct& rhs)const{return ( (rowindex + colindex*dim) <= (rhs.rowindex + rhs.colindex*rhs.dim) ) ;};*/

};

struct ham_sort_function{

        __host__ __device__ bool operator()(hamstruct a, hamstruct b){

                return ( (a.colindex + a.rowindex*a.dim) < (b.colindex + b.rowindex*b.dim) );
        
        }

};

__device__ long atomicAdd(long* address, long val){
	unsigned long long int* address_as_ull = (unsigned long long int *)address; 
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, (long long)(val + (long)(assumed)));
	} while (assumed != old);

	return (long)(old);
}

__global__ void FillSparse(long* d_basis_Position, long* d_basis, int dim, cuDoubleComplex* H_vals, long2* H_pos, long* d_Bond, int lattice_Size, const double JJ);

__global__ void CompressSparse(cuDoubleComplex* H_vals, long2* H_pos, long d_dim, const int lattice_Size);

__global__ void UpperHalfToFull(long2* H_pos, cuDoubleComplex* H_vals, hamstruct* H_sort, long num_Elem, long dim, int lattice_Size);

__global__ void FullToCOO(long num_Elem, hamstruct* H_sort, cuDoubleComplex* hamil_Values, long* hamil_PosRow, long* hamil_PosCol, long dim);

__global__ void GetNumElem(long2* H_pos, int lattice_Size);


