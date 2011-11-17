#include<iostream>
#include"testhamiltonian.h"
#include"lanczos.h"
#include"cuda.h"
#include"cuComplex.h"
#include"lattice.h"

using namespace std;

int main(){

  int* Bond;
  Bond = (int*)malloc(16*3*sizeof(int));
	
  //Fill_Bonds_16B(Bond);
  Bond[0] = 0;    Bond[1] = 1;    Bond[2] = 2;    Bond[3] = 3;    Bond[4] = 4;
  Bond[5] = 5;    Bond[6] = 6;    Bond[7] = 7;    Bond[8] = 8;    Bond[9] = 9;
  Bond[10] = 10;  Bond[11] = 11;  Bond[12] = 12;  Bond[13] = 13;  Bond[14] = 14;
  Bond[15] = 15;  Bond[16] = 1;   Bond[17] = 2;   Bond[18] = 3;   Bond[19] = 0;
  Bond[20] = 5;   Bond[21] = 6;   Bond[22] = 7;   Bond[23] = 4;   Bond[24] = 9;
  Bond[25] = 10;  Bond[26] = 11;  Bond[27] = 8;   Bond[28] = 13;  Bond[29] = 14;
  Bond[30] = 15;  Bond[31] = 12;  Bond[32] = 4;   Bond[33] = 5;   Bond[34] = 6;
  Bond[35] = 7;   Bond[36] = 8;   Bond[37] = 9;   Bond[38] = 10;  Bond[39] = 11;
  Bond[40] = 12;  Bond[41] = 13;  Bond[42] = 14;  Bond[43] = 15;  Bond[44] = 0;
  Bond[45] = 1;   Bond[46] = 2;   Bond[47] = 3;

  cuDoubleComplex* hamil_Values;
  int* hamil_PosRow;
  int* hamil_PosCol;

  int nsite = 16;

  int dim;

  int Sz = 0;
  float JJ = 1.;

  const int num_Elem = ConstructSparseMatrix( 0, nsite, Bond, hamil_Values, hamil_PosRow, hamil_PosCol, &dim, JJ, Sz );
  //cout<<"ConstructSparseMatrix finished"<<endl;
  int rtn = num_Elem;
  if (rtn == 1){
	return 1;
  }
  rtn = 0;
  free(Bond);

  lanczos(num_Elem, hamil_Values, hamil_PosRow, hamil_PosCol, dim, 200, 3, 1e-3);

  cudaFree(hamil_Values);
  cudaFree(hamil_PosRow);
  cudaFree(hamil_PosCol);
  return rtn;

}


