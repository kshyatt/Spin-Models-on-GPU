#include<hamiltonian.h>



__host__ void GetBasis(int dim, int Nsite, int Sz, long basis_Position[], long basis[]){
	unsigned long temp = 0;

	for (unsigned long i1=0; i1<dim; i1++){
      		temp = 0;
		basis_Position[i1] = -1;
      		for (int sp =0; sp<Nsite; sp++){
          		temp += (i1>>sp)&1;
		}  //unpack bra
      		if (temp==(Nsite/2+Sz) ){ 
          		basis[i1] = i1;
          		basis_Position[i1] = i1 -1;
      		}
  	} 

}

__device__ cuDoubleComplex HOffBondX(const int si, const long bra, const double JJ){

	cuDoubleComplex valH;
  	int S0, S1;
  	int T0, T1;

  	valH = make_cuDoubleComplex( JJ*0.5 , 0.); //contribution from the J part of the Hamiltonian

  	return valH;


} 

__device__ cuDoubleComplex HOffBondY(const int si, const long bra, const double JJ){

	cuDoubleComplex valH;
  	int S0, S1;
  	int T0, T1;

  	valH = make_cuDoubleComplex( JJ*0.5 , 0. ); //contribution from the J part of the Hamiltonian

  	return valH;


}

__device__ cuDoubleComplex HDiagPart(const long bra, int lattice_Size, long* d_Bond, const double JJ){

  int S0b,S1b ;  //spins (bra 
  int T0,T1;  //site
  int P0, P1, P2, P3; //sites for plaquette (Q)
  int s0p, s1p, s2p, s3p;
  cuDoubleComplex valH = make_cuDoubleComplex( 0. , 0.);

  for (int Ti=0; Ti<lattice_Size; Ti++){
    //***HEISENBERG PART

    T0 = d_Bond[Ti]; //lower left spin
    S0b = (bra>>T0)&1;  
    //if (T0 != Ti) cout<<"Square error 3\n";
    T1 = d_Bond[Ti + lattice_Size]; //first bond
    S1b = (bra>>T1)&1;  //unpack bra
    valH.x += JJ*(S0b-0.5)*(S1b-0.5);
    T1 = d_Bond[Ti + 2*lattice_Size]; //second bond
    S1b = (bra>>T1)&1;  //unpack bra
    valH.x += JJ*(S0b-0.5)*(S1b-0.5);

  }//T0

  //cout<<bra<<" "<<valH<<endl;

  return valH;

}//HdiagPart 

int ConstructSparseMatrix(int model_Type, int lattice_Size, long* Bond){
	
	unsigned long num_Elem; // the total number of elements in the matrix, will get this (or an estimate) from the input types
	cudaError_t status1, status2, status3;

	switch (model_Type){
		case 0: num_Elem = 219648;
		case 1: num_Elem = 10; //guesses
	}

	int dim = 2;

	cuDoubleComplex* hamil_Values;
	status1 = cudaMalloc(&hamil_Values, num_Elem*sizeof(cuDoubleComplex));

	long* hamil_PosRow;
	status2 = cudaMalloc(&hamil_PosRow, (1<<lattice_Size)*sizeof(long));

	long* hamil_PosCol;
	status3 = cudaMalloc(&hamil_PosCol, num_Elem*sizeof(long));

	if ( (status1 != CUDA_SUCCESS) ||
	     (status2 != CUDA_SUCCESS) ||
	     (status3 != CUDA_SUCCESS) ){
		cout<<"Memory allocation for COO representation failed!";
		return 1;
	}		

	long* d_Bond;
	status1 = cudaMalloc(&d_Bond, sizeof(Bond)*sizeof(long));

	status2 = cudaMemcpy(d_Bond, Bond, sizeof(Bond)*sizeof(long), cudaMemcpyHostToDevice);

	if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
		cout<<"Memory allocation and copy for bond data failed!";
		return 1;
	}
		
	for (int ch=1; ch<lattice_Size; ch++) dim *=2;

	long basis_Position[dim];
	long basis[dim];

	int Sz = 0;
	
	GetBasis(dim, lattice_Size, Sz, basis_Position, basis);

	long* d_basis_Position;
	long* d_basis;

	status1 = cudaMalloc(&d_basis_Position, dim*sizeof(long));
	status2 = cudaMalloc(&d_basis, dim*sizeof(long));

	if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
		cout<<"Memory allocation for basis arrays failed!";
		return 1;
	}

	status1 = cudaMemcpy(d_basis_Position, basis_Position, dim*sizeof(long), cudaMemcpyHostToDevice);
	status2 = cudaMemcpy(d_basis, basis, dim*sizeof(long), cudaMemcpyHostToDevice);

	if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
		cout<<"Memory copy for basis arrays failed!";
		return 1;
	}

	dim3 bpg = (ceil(dim/256.), lattice_Size);
	int tpb = 256; //these are going to need to depend on dim and Nsize

	long** H_pos;
	cuDoubleComplex** H_vals; 

	status1 = cudaMalloc(&H_pos, dim*sizeof(long*));
	status2 = cudaMalloc(&H_vals, dim*sizeof(cuDoubleComplex*));

	if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
		cout<<"Memory allocation for upper half arrays failed!";
		return 1;
	}

	int* d_dim;
	status1 = cudaMalloc(&d_dim, sizeof(int));
	status2 = cudaMemcpy(d_dim, &dim, sizeof(int), cudaMemcpyHostToDevice);

	if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
		cout<<"Memory allocation and copy for dimension failed!";
		return 1;
	}

	int* d_lattice_Size;
	status1 = cudaMalloc(&d_lattice_Size, sizeof(int));
	status2 = cudaMemcpy(d_lattice_Size, &lattice_Size, sizeof(int), cudaMemcpyHostToDevice);

	if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
		cout<<"Memory allocation and copy for lattice size failed!";
		return 1;
	}


	
	CDCarraysalloc<<<bpg, tpb>>>(H_vals, *d_dim, 2*(*d_lattice_Size) + 1, 0);
	
	longarraysalloc<<<bpg, tpb>>>(H_pos, *d_dim, 2*(*d_lattice_Size) + 2, 0); //going to stick a number in front to count the number of nonzero elements

	for(int jj = 0; jj < dim; jj++){
		cudaMemset(&H_pos[jj][0], 1, sizeof(int));
	} //counting the diagonal element

	double JJ = 1.;

	FillSparse<<<bpg, tpb>>>(d_basis_Position, d_basis, *d_dim, H_vals, H_pos, d_Bond, *d_lattice_Size, JJ);

	cudaThreadSynchronize(); //need to make sure all elements are initialized before I start compression
	
	CompressSparse<<<bpg, tpb>>>(H_vals, H_pos, *d_dim, *d_lattice_Size);

	long** h_H_pos;
	cuDoubleComplex** h_H_vals;

	cudaMallocHost(&h_H_pos, dim*sizeof(long*));
	cudaMallocHost(&h_H_vals, dim*sizeof(cuDoubleComplex*));

	cudaMemcpy(h_H_pos, H_pos, dim*sizeof(long*), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_H_vals, H_vals, dim*sizeof(cuDoubleComplex*), cudaMemcpyDeviceToHost);

	for(int ii = 0; ii < dim; ii++){

		long stemp;
    		bool noswap = false;
    		while (noswap == false){
      			noswap = true; 
			cuDoubleComplex tempD;
      			for (int i2=1; i2<h_H_pos[ii][0]-1; i2++){ //ignore 0 element
				
        			if (h_H_pos[ii][i2] > h_H_pos[ii][i2+1] ) {
          				stemp = h_H_pos[ii][i2];
          				h_H_pos[ii][i2] = h_H_pos[ii][i2 +1];
          				h_H_pos[ii][i2 +1] = stemp;
          				tempD = h_H_vals[ii][i2];
          				h_H_vals[ii][i2] = h_H_vals[ii][i2+1];
          				h_H_vals[ii][i2+1] = tempD;
          				noswap = false;
        			}
      			}//i2
    		}//while


		for(int jj = 0; jj < (h_H_pos[ii][0] + 1); jj++){
			cout<<"At position: ("<<ii<<", "<<h_H_pos[ii][jj+1]<<": ";
			cout<<h_H_vals[ii][jj].x<<endl;
		}
	}

}


__global__ void FillSparse(long* d_basis_Position, long* d_basis, int d_dim, cuDoubleComplex** H_vals, long** H_pos, long* d_Bond, int d_lattice_Size, const double JJ){

	int T0 = blockIdx.y; //my indices!
	int ii = threadIdx.x + 256*blockIdx.x;

	int si, sj,sk,sl; //spin operators
	unsigned long tempi, tempj, tempod;
	cuDoubleComplex tempD;

	if( ii < d_dim ){
		//Diagonal part----------------
		tempi = d_basis[ii];
		H_pos[ii][1] = d_basis_Position[tempi];

		H_vals[ii][0] = HDiagPart(tempi, d_lattice_Size, d_Bond, JJ);

		//-------------------------------
		//Horizontal bond ---------------
		si = d_Bond[T0];
		tempod = tempi;
		sj = d_Bond[T0 + d_lattice_Size];
	
		tempod ^= (1<<si);   //toggle bit 
		tempod ^= (1<<sj);   //toggle bit 

		if (d_basis_Position[tempod] > ii){ //build only upper half of matrix
        		H_pos[ii][(2*T0)+2] = d_basis_Position[tempod];

        		H_vals[ii][(2*T0)+1] = HOffBondX(T0,tempi, JJ);
			H_pos[ii][0]++; 
      		}

		else {
			H_pos[ii][(2*T0)+2] = -1;
			H_vals[ii][(2*T0)+1] = make_cuDoubleComplex(0., 0.);
		}

		//Vertical bond -----------------
		tempod = tempi;
      		sj = d_Bond[T0 + 2*d_lattice_Size];
      		tempod ^= (1<<si);   //toggle bit 
     		tempod ^= (1<<sj);   //toggle bit 
      		if (d_basis_Position[tempod] > ii){ 
        		H_pos[ii][(2*T0)+3] = d_basis_Position[tempod];
   
        		H_vals[ii][(2*T0)+2] = HOffBondY(T0,tempi, JJ);
			H_pos[ii][0]++;
      		}

		else {
			H_pos[ii][(2*T0)+3] = -1;
			H_vals[ii][(2*T0)+2] = make_cuDoubleComplex(0., 0.);
		}
	}
}


__global__ void CompressSparse(cuDoubleComplex** H_vals, long** H_pos, int d_dim, int lattice_Size){

	int i = 256*blockIdx.x + threadIdx.x;

	int iter = 0;

	if (i < d_dim){
	
		long* temp_pos = (long*)malloc(H_pos[i][0]*sizeof(long));
		cuDoubleComplex* temp_val = (cuDoubleComplex*)malloc((H_pos[i][0]+ 1)*sizeof(long));	

		temp_pos[0] = H_pos[i][0];

		for(int j = 0; j < 2*lattice_Size + 1; j++){
			if( (H_pos[i][j+1] != -1) ){
				temp_pos[iter+1] = H_pos[i][j+1];
				temp_val[iter] = H_vals[i][j];
				iter++;
			}
		 
		}

		free(H_pos[i]); //switching out the old ones for the new
		free(H_vals[i]);
		
		H_pos[i] = temp_pos;
		H_vals[i] = temp_val;

		
	}
}
