#include<cmath>
#include<iostream>
#include<cstdlib>
#include<algorithm>
#include<vector>
#include<stdio.h>
#include"cuda.h"
#include"cuComplex.h"


using namespace std;

#define likely(x)   __builtin_expect((x),1)
#define unlikely(x) __builtin_expect((x),0)
//#define idx(i,j,lda) = ( j + (i*lda))

__host__ __device__ long idx(long i, long j, long lda){

	return ( j + (i*lda));

}

struct hamstruct{

	long position;
	cuDoubleComplex value;

	bool operator>(const hamstruct& rhs)const{return position > rhs.position;};
	bool operator<(const hamstruct& rhs)const{return position < rhs.position;};
	bool operator>=(const hamstruct& rhs)const{return position >= rhs.position;};
	bool operator<=(const hamstruct& rhs)const{return position <= rhs.position;};
};
/*
__global__ void CDCarraysalloc(cuDoubleComplex** a, long dim, long n, long m){
  long i = blockDim.x*blockIdx.x + threadIdx.x + m;
  
  if (i < dim){
    a[i] = (cuDoubleComplex*)malloc(n*sizeof(cuDoubleComplex));
  }

  if  (a[i] == (cuDoubleComplex*)NULL) {
    printf("The %d th positions array failed to allocate! \n", i);
  }

}

__global__ void longarraysalloc(long** a, long dim, long n, long m){
  long i = blockDim.x*blockIdx.x + threadIdx.x + m;
  if (i < dim){
    a[i] = (long*)malloc(n*sizeof(long));
  }

  if (a[i] == (long*)NULL) {
    printf("The %d th values array failed to allocate! \n", i);
  }

}
*/
__global__ void FillSparse(long* d_basis_Position, long* d_basis, int dim, cuDoubleComplex* H_vals, long2* H_pos, long* d_Bond, int lattice_Size, const double JJ);



__global__ void CompressSparse(cuDoubleComplex* H_vals, long2* H_pos, int d_dim, const int lattice_Size);

__host__ void UpperHalfToFull(long2* H_pos, cuDoubleComplex* H_vals, long2* buffer_pos, cuDoubleComplex* buffer_val, long num_Elem, long dim, int lattice_Size);

__global__ void FullToCOO(long num_Elem, cuDoubleComplex* H_vals, long2* H_pos, cuDoubleComplex* hamil_Values, long* hamil_PosRow, long* hamil_PosCol, long dim);

__global__ void SortHamiltonian(long2* H_pos, cuDoubleComplex* H_vals, long dim, int lattice_Size);
