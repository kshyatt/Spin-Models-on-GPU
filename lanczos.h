#include<iostream>
#include<cstdlib>
#include<cmath>
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

using namespace thrust;

//#include"hamiltonian.h"

void lanczos(const cuDoubleComplex* h_H, const int dim, int max_Iter, const int num_Eig, const double conv_req);
