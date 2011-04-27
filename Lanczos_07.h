//Lanczos_07.h
//c++ class for performing a Lanczos diagonalization
//Roger Melko, November 2007

#ifndef LANCZOS_07
#define LANCZOS_07


#include <iostream>
#include <limits.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <iomanip>
#include <vector>
using namespace std;

#include <blitz/array.h>
BZ_USING_NAMESPACE(blitz)

#include "GenHam.h"

typedef long double l_double;  //precision for lanczos

class LANCZOS{

  public:
    //Data
    int Dim; //dimension
    Array<l_double,1> Psi;  //eigenvector

   //Methods
   LANCZOS(const int);
   void Diag(const GENHAM&, const int, const int);
   void tred3(Array<double,2>& , Array<double,1>& , Array<double,1>& e, const int );


  private:
   int STARTIT;
   long double CONV_PREC; //convergence precision

   Array<l_double,1> V0;  
   Array<l_double,1> V1;    //Ground state vector
   Array<l_double,1> V2;

   void apply(Array<l_double,1>&, const GENHAM&, const Array<l_double,1>&);  //apply H to |V>
   void Normalize(Array<l_double,1>& );
   int tqli2(Array<l_double,1>& , Array<l_double,1>& , int , Array<l_double,2>& , const int );

}; //LANCZOS class


#endif
