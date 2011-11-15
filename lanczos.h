#include<iostream>
#include<cstdlib>
#include<cmath>
#include<fstream>
#include<iomanip>
#include"cuda.h"
#include"cuda_runtime.h"
#include"cublas_v2.h"
#include"cusparse_v2.h"
#include"cuComplex.h"
#include"thrust/device_vector.h"
#include"thrust/host_vector.h"
#include"thrust/copy.h"
#include"thrust/fill.h"
#include"thrust/sort.h"
#include"thrust/device_ptr.h"
#include"thrust/system_error.h"

#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

using namespace thrust;

__host__ void lanczos(const int num_Elem, cuDoubleComplex*& d_H_vals, int*& d_H_rows, int*& d_H_cols, const int dim, int max_Iter, const int num_Eig, const double conv_req);

__global__ void normalize(cuDoubleComplex* v, const int size, double norm);

__host__ int tqli(double* d, double* e, int n, int max_Iter, double* z);
 
__host__ double pythag(double a, double b);
