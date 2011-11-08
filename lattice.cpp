#include"lattice.h"

//lattice::lattice(int lattice_Size){
  
  //Bond = (int*)malloc(lattice_Size*3*sizeof(int));

//}

void Fill_Bonds_12A(int* Bond){

  Bond = (int*)malloc(12*3*sizeof(int));

  for(int i = 0; i < 12; i++){
    Bond[i] = i;
  }

  Bond[12] = 5; Bond[13] = 2; Bond[14] = 3; Bond[15] = 4;
  Bond[16] = 10; Bond[17] = 6; Bond[18] = 7; Bond[19] = 8;
  Bond[20] = 9; Bond[21] = 0; Bond[22] = 11; Bond[23] = 1;
  Bond[24] = 3; Bond[25] = 6; Bond[26] = 7; Bond[27] = 8;
  Bond[28] = 9; Bond[29] = 4; Bond[30] = 10; Bond[31] = 11;
  Bond[32] = 1; Bond[33] = 2; Bond[34] = 0; Bond[35] = 5;
}

void Fill_Bonds_16B(int* Bond){

  Bond = (int*)malloc(16*3*sizeof(int));

  Bond[0] = 0; Bond[1] = 1; Bond[2] = 2; Bond[3] = 3; Bond[4] = 4;
  Bond[5] = 5; Bond[6] = 6; Bond[7] = 7; Bond[8] = 8; Bond[9] = 9;
  Bond[10] = 10; Bond[11] = 11; Bond[12] = 12; Bond[13] = 13; Bond[14] = 14;
  Bond[15] = 15; Bond[16] = 1; Bond[17] = 2; Bond[18] = 3; Bond[19] = 0;
  Bond[20] = 5; Bond[21] = 6; Bond[22] = 7; Bond[23] = 4; Bond[24] = 9;
  Bond[25] = 10; Bond[26] = 11; Bond[27] = 8; Bond[28] = 13; Bond[29] = 14;
  Bond[30] = 15; Bond[31] = 12; Bond[32] = 4; Bond[33] = 5; Bond[34] = 6;
  Bond[35] = 7; Bond[36] = 8; Bond[37] = 9; Bond[38] = 10; Bond[39] = 11;
  Bond[40] = 12; Bond[41] = 13; Bond[42] = 14; Bond[43] = 15; Bond[44] = 0;
  Bond[45] = 1; Bond[46] = 2; Bond[47] = 3;

}

/*void Fill_Bonds_18A(int* Bond){

  Bond = (int*)malloc(3*18*sizeof(int));

  for(int i = 0; i < 18; i++){
    Bond[i] = i;
  }

  Bond[18] = 3; Bond[19] = 4; Bond[20] = 5;
  Bond[21] = 

}*/

void Fill_Bonds_22A(int* Bond){

  Bond = (int*)malloc(3*22*sizeof(int));

  for(int i = 0; i < 22; i++){
    Bond[i] = i;
    Bond[i + 22] = i + 1;
  }
  Bond[43] = 0;
  
  Bond[44] = 5; Bond[45] = 6; Bond[46] = 7; Bond[47] = 8; Bond[48] = 9; 
  Bond[49] = 10; Bond[50] = 11; Bond[51] = 12; Bond[52] = 13; Bond[53] = 14;
  Bond[54] = 15; Bond[55] = 16; Bond[56] = 17; Bond[57] = 18; Bond[58] = 19;
  Bond[59] = 20; Bond[60] = 21; Bond[61] = 0; Bond[62] = 1; Bond[63] = 2;
  Bond[64] = 3; Bond[65] = 4; 
}

void Fill_Bonds_22B(int* Bond){

  Bond = (int*)malloc(3*22*sizeof(int));

  for(int i = 0; i < 22; i++){
    Bond[i] = i;
  }

  Bond[22] = 15; Bond[23] = 2; Bond[24] = 3; Bond[25] = 20;
  Bond[26] = 5; Bond[27] = 6; Bond[28] = 7; Bond[29] = 8;
  Bond[30] = 9; Bond[31] = 0; Bond[32] = 11; Bond[33] = 12;
  Bond[34] = 13; Bond[35] = 14; Bond[36] = 15; Bond[37] = 1; 
  Bond[38] = 17; Bond[39] = 18; Bond[40] = 19; Bond[41] = 4;
  Bond[42] = 21; Bond[43] = 9;

  Bond[44] = 1; Bond[45] = 5; Bond[46] = 6; Bond[47] = 7;
  Bond[48] = 10; Bond[49] = 11; Bond[50] = 12; Bond[51] = 13;
  Bond[52] = 14; Bond[53] = 15; Bond[54] = 16; Bond[55] = 17;
  Bond[56] = 18; Bond[57] = 19; Bond[58] = 4; Bond[59] = 2; 
  Bond[60] = 3; Bond[61] = 20; Bond[62] = 21; Bond[63] = 9;
  Bond[64] = 8; Bond[65] = 0;
}
