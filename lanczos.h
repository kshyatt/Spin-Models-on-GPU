#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include"cublas.h"
#include"cusparse.h"
#include"cuComplex.h"
#include"tqli.h"
#include"hamiltonian.h"

void lanczos(const cuDoubleComplex* h_H, const int dim, int max_Iter, const int num_Eig, const double conv_req);
