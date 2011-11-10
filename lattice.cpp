#include"lattice.h"

void Fill_Bonds_16B(int* Bond){

  Bond = (int*)malloc(3*16*sizeof(int));

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

void Fill_Bonds_18A(int* Bond){

  Bond = (int*)malloc(3*18*sizeof(int));

  for(int i = 0; i < 18; i++){
    Bond[i] = i;
  }

  Bond[18] = 3; Bond[19] = 4; Bond[20] = 5;
  Bond[21] = 

}

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
  Bond[26] = 5; Bond[27] = [
