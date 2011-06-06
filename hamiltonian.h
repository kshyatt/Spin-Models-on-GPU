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

struct hamstruct{

	long position;
	cuDoubleComplex value;

	bool operator>(const hamstruct& rhs)const{return position > rhs.position;};
	bool operator<(const hamstruct& rhs)const{return position < rhs.position;};
	bool operator>=(const hamstruct& rhs)const{return position >= rhs.position;};
	bool operator<=(const hamstruct& rhs)const{return position <= rhs.position;};
};

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

__global__ void FillSparse(long* d_basis_Position, long* d_basis, int d_dim, cuDoubleComplex** H_vals, long** H_pos, long* d_Bond, int d_lattice_Size, const double JJ);



__global__ void CompressSparse(cuDoubleComplex** H_vals, long** H_pos, int d_dim, int lattice_Size);

__host__ void UpperHalfToFull(cuDoubleComplex** H_vals, long** H_pos, long dim, int lattice_Size);

__global__ void FullToCOO(long num_Elem, cuDoubleComplex** H_vals, long** H_pos, cuDoubleComplex* hamil_Values, long* hamil_PosRow, long* hamil_PosCol);
