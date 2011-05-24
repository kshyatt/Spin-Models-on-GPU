
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<thrust/sort.h>
#include<thrust/device_ptr.h>
#include<stdio.h>
#include<stdlib.h>
#include"cublas.h"
#include"cusparse.h"
#include"cuComplex.h"
#include"tqli.h"
#include"hamiltonian.h"

using namespace thrust;

void lanczos(const int h_num_nonzeroelem, const cuDoubleComplex* h_values, const int* h_rowstart, const int* h_colindex, const cusparseMatDescr_t* h_descrH, const int dim, int max_Iter, const int num_Eig, const double conv_req);

