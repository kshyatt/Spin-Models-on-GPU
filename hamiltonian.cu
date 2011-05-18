#include<hamiltonian.h>

__global__ void assign(double* a, double b, int n){
  int i = blockDim.x*blockIdx.x + threadIdx.x;

  if (i < n){
    a[i] = b;
  }
}


__host__ void GetBasis(int dim, int Nsite, int Sz, long* basis_Position, long* basis){
	unsigned long temp = 0;
	
	for (unsigned long i1=0; i1<dim; i1++){
		basis[il] = -1;
	}
	
	for (unsigned long i1=0; i1<dim; i1++){
      		temp = 0;
      		for (int sp =0; sp<Nsite; sp++)
          		temp += (i1>>sp)&1;  //unpack bra
      		if (temp==(Nsite/2+Sz) ){ 
          		basis[i1] = il;
          		basis_Position[il] = il -1;
      		}
  	} 


int ConstructSparseMatrix(enum model_Type, int lattice_Size, long* Bond){
	
	unsigned long num_Elem; // the total number of elements in the matrix, will get this (or an estimate) from the input types
	cudaError_t status1, status2, status3;

	switch (model_Type){
		case 0: num_Elem = 219648;
		case 1: num_Elem = 10; //guesses
	}

	cuDoubleComplex* hamil_Values;
	status1 = cudaMalloc(&hamil_Values, num_Elem*sizeof(cuDoubleComplex));

	long* hamil_PosRow;
	status2 = cudaMalloc(&hamil_PosRow, (1<<lattice_Size)*sizeof(long));

	long* hamil_PosCol;
	status3 = cudaMalloc(&hamil_PosCol, num_Elem*sizeof(long));

	if ( (status1 != CUDA_SUCCESS) ||
	     (status2 != CUDA_SUCCESS) ||
	     (status3 != CUDA_SUCCESS) ){
		printf("Memory allocation for COO representation failed!");
		return 1;
	}		

	//long* d_Bond;
	//status1 = cudaMalloc(&d_Bond, Bond.size()*sizeof(long));

	//status2 = cudaMemcpy(d_Bond, Bond, Bond.size()*sizeof(long), cudaMemcpyHostToDevice);

	//if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
	//	printf("Memory allocation and copy for bond data failed!");
	//	return 1;
	//}
		
	for (int ch=1; ch<Nsite; ch++) dim *=2;

	long basis_Position[dim];
	long basis[dim];
	
	
