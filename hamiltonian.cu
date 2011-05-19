#include<hamiltonian.h>



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

}

__device__ double HOffBondX(const int si, const long bra, const double JJ){

	double valH;
  	int S0, S1;
  	int T0, T1;

  	valH = JJ*0.5; //contribution from the J part of the Hamiltonian

  	return valH;


} 

__device__ double HOffBondY(const int si, const long bra, const double JJ){

	double valH;
  	int S0, S1;
  	int T0, T1;

  	valH = JJ*0.5; //contribution from the J part of the Hamiltonian

  	return valH;


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

	long* d_Bond;
	status1 = cudaMalloc(&d_Bond, sizeof(Bond)*sizeof(long));

	status2 = cudaMemcpy(d_Bond, Bond, sizeof(Bond)*sizeof(long), cudaMemcpyHostToDevice);

	if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
		printf("Memory allocation and copy for bond data failed!");
		return 1;
	}
		
	for (int ch=1; ch<Nsite; ch++) dim *=2;

	long basis_Position[dim];
	long basis[dim];
	
	GetBasis(dim, lattice_Size, Sz, &basis_Position, &basis);

	long* d_basis_Position;
	long* d_basis;

	status1 = cudaMalloc(&d_basis_Position, dim*sizeof(long));
	status2 = cudaMalloc(&d_basis, dim*sizeof(long));

	if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
		printf("Memory allocation for basis arrays failed!");
		return 1;
	}

	status1 = cudaMemcpy(d_basis_Position, basis_Position, dim*sizeof(long), cudaMemcpyHostToDevice);
	status2 = cudaMemcpy(d_basis, basis, dim*sizeof(long), cudaMemcopyHostToDevice);

	if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
		printf("Memory copy for basis arrays failed!");
		return 1;
	}

	int bpg;
	int tpb; //these are going to need to depend on dim and Nsize

	FillSparse<<<bpg, tpb>>>(d_basis_Position, 
