#include<iostream>
#include<cstdlib>
#include<cmath>
#include<exception>
#include"cuda.h"
#include"cuda_runtime.h"
#include"cublas_v2.h"
#include"cusparse.h"
#include"cuComplex.h"
#include"tqli.h"
#include"thrust/device_vector.h"
#include"thrust/host_vector.h"
#include"thrust/copy.h"
#include"thrust/fill.h"
#include"thrust/sort.h"
#include"thrust/device_ptr.h"
#include"thrust/system_error.h"

using namespace thrust;

__host__ void lanczos(const int num_Elem, cuDoubleComplex*& d_H_vals, int*& d_H_rows, int*& d_H_cols, const int dim, int max_Iter, const int num_Eig, const double conv_req);
