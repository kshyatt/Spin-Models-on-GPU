#include<cmath>
#include<iostream>
#include<cstdlib>
#include<algorithm>
#include<cuda.h>
#include"cuComplex.h"

using namespace std;

/*
typedef struct Hamiltonian {
	long n_elements;
	cuDoubleComplex* values;
	long* row_indices;
	long* col_indices;
}//setting up a struct that will be easy to use as an encoding for a COO sparse matrix

struct Hamiltonian create(long n, cuDoubleComplex* val, long* row, long* col){

	struct Hamiltonian temp;

	temp.n_elements = n;
	temp.values = val;
	temp.row_indices = row;
	temp.col_indices = col;

	return temp;
}

struct Hamiltonian addpoint(Hamiltonian H, double val, long i, long j){
*/
//keeping this stuff commented out for now 

__global__ void CDCarraysalloc(cuDoubleComplex** a, long dim, long n, long m){
  int i = 256*blockIdx.x + threadIdx.x + m;
  if (i < dim){
    a[i] = (cuDoubleComplex*)malloc(n*sizeof(cuDoubleComplex));
  }
}

__global__ void longarraysalloc(long** a, long dim, long n, long m){
  int i = 256*blockIdx.x + threadIdx.x + m;
  if (i < dim){
    a[i] = (long*)malloc(n*sizeof(long));
  }
}

__global__ void FillSparse(long* d_basis_Position, long* d_basis, int d_dim, cuDoubleComplex** H_vals, long** H_pos, long* d_Bond, int d_lattice_Size, const double JJ);



__global__ void CompressSparse(cuDoubleComplex** H_vals, long** H_pos, int d_dim, int lattice_Size);
