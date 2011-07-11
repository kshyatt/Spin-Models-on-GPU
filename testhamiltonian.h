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

#define likely(x)   __builtin_expect((x),1)
#define unlikely(x) __builtin_expect((x),0)
//#define idx(i,j,lda) = ( j + (i*lda))

__host__ __device__ long idx(long i, long j, long lda){

	return ( j + (i*lda));

}

__device__ long d_num_Elem;

struct hamstruct{

	long row;
        long col;
	cuDoubleComplex value;
        long dim;

	bool operator>(const hamstruct& rhs)const{return ( (row + col*dim) > (rhs.row + rhs.col*rhs.dim) ) ;};
	bool operator<(const hamstruct& rhs)const{return ( (row + col*dim) < (rhs.row + rhs.col*rhs.dim) ) ;};
	bool operator>=(const hamstruct& rhs)const{return ( (row + col*dim) >= (rhs.row + rhs.col*rhs.dim) ) ;};
	bool operator<=(const hamstruct& rhs)const{return ( (row + col*dim) <= (rhs.row + rhs.col*rhs.dim) ) ;};
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

__global__ void CompressSparse(cuDoubleComplex* H_vals, long2* H_pos, hamstruct* H_sort, long d_dim, const int lattice_Size);

__host__ void UpperHalfToFull(long2* H_pos, cuDoubleComplex* H_vals, long2* buffer_pos, cuDoubleComplex* buffer_val, long num_Elem, long dim, int lattice_Size);

__global__ void FullToCOO(long num_Elem, cuDoubleComplex* H_vals, long2* H_pos, cuDoubleComplex* hamil_Values, long* hamil_PosRow, long* hamil_PosCol, long dim);

__global__ void GetNumElem(long2* H_pos, int lattice_Size);

__global__ void CopyBack(hamstruct* H_sort, long2* H_pos, cuDoubleComplex* H_vals, long dim, int lattice_Size, long num_Elem);
