#include<hamiltonian.h>

void ConstructSparseMatrix(enum model_Type, int lattice_Size, long* Bond){
	
	int num_Elem; // the total number of elements in the matrix, will get this (or an estimate) from the input types


	switch (model_Type){
		case 0: num_Elem = 0;
		case 1: num_Elem = 10; //guesses
	}

	cuDoubleComplex* hamil_Values;
	cudaMalloc(&hamil_Values, num_Elem*sizeof(cuDoubleComplex));

	long* hamil_PosRow;
	cudaMalloc(&hamil_PosRow, (1<<lattice_Size)*sizeof(long));

	long* hamil_PosCol;
	cudaMalloc(&hamil_PosCol, num_Elem*sizeof(long));

	long* d_Bond;
	cudaMalloc(&d_Bond, Bond.size()*sizeof(long));

	cudaMemcpy(d_Bond, Bond, Bond.size()*sizeof(long), cudaMemcpyHostToDevice);
		

	
	
