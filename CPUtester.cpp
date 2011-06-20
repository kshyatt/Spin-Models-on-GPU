#define DO_LANCZOS //Lanczos or LAPACK full ED

#include <fstream> 
#include <vector> 
#include <math.h>
using namespace std;

#include <iostream>
#include <limits.h>
#include <stdio.h>
#include <time.h>
#include <iomanip>
#include <blitz/array.h>

BZ_USING_NAMESPACE(blitz)

#include "GenHam.h"
#include "lapack.h"
#include "simparam.h"

#include "cuComplex.h"
#include "converter.h"
#include "lanczos.h"
#include "cusparse.h"

int main(){

//  cout<<sizeof(long double)<<endl;
//  cout<<sizeof(float)<<endl;

  PARAMS prm;
  double J;
  double J2;
  double Q;
  int Sz;
  
  J=prm.JJ_;
  J2=prm.J2_;
  Q=-prm.QQ_;// SIGN HAS TO BE FLIPPED: NOW SIGN AT INPUT IS SAME AS PAPER 
  Sz=prm.Sz_;

  GENHAM HV(16,J,J2,Q,Sz); 
  HV.Bonds_16B(); 

  HV.SparseHamJQ();

  long num_Elem = 0;
  long* row_index;
  long* col_index;
  cuDoubleComplex* values;

  converter(HV.PosHam, HV.ValHam, &(num_Elem), row_index, col_index, values);

  cusparseMatDescr_t description = (CUSPARSE_MATRIX_TYPE_HERMITIAN, 

  lanczos(&(num_Elem), values, row_index, col_index, description, 65536, 200, 1e-12);

  return 0;

}
