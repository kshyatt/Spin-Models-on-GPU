#include"hamiltonian.h"



__host__ void GetBasis(int dim, int lattice_Size, int Sz, long basis_Position[], long basis[]){
	unsigned long temp = 0;

	for (unsigned long i1=0; i1<dim; i1++){
      		temp = 0;
		basis_Position[i1] = -1;
      		for (int sp =0; sp<lattice_Size; sp++){
          		temp += (i1>>sp)&1;
		}  //unpack bra
      		if (temp==(lattice_Size/2+Sz) ){ 
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
	//status1 = cudaMalloc(&hamil_Values, num_Elem*sizeof(cuDoubleComplex));

	long* hamil_PosRow;
	//status2 = cudaMalloc(&hamil_PosRow, (1<<lattice_Size)*sizeof(long));

	long* hamil_PosCol;
	//status3 = cudaMalloc(&hamil_PosCol, num_Elem*sizeof(long));

	/*if ( (status1 != CUDA_SUCCESS) ||
	     (status2 != CUDA_SUCCESS) ||
	     (status3 != CUDA_SUCCESS) ){
		cout<<"Memory allocation for COO representation failed!";
		return 1;
	}*/		

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

	dim3 bpg = (dim/256, lattice_Size);
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


	for(long i = 0; i<dim; i++){
		cudaMalloc(&H_vals[i], (2*(lattice_Size)+1)*sizeof(cuDoubleComplex));
		cudaMalloc(&H_pos[i], (2*(lattice_Size)+2)*sizeof(long));
	}

	
	//CDCarraysalloc<<<bpg, tpb>>>(H_vals, *d_dim, 2*(*d_lattice_Size) + 1, 0);
	
	//longarraysalloc<<<bpg, tpb>>>(H_pos, *d_dim, 2*(*d_lattice_Size) + 2, 0); //going to stick a number in front to count the number of nonzero elements

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

		cudaMemcpy(h_H_pos[ii], H_pos[ii], (2*lattice_Size + 2)*sizeof(long), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_H_vals[ii], H_vals[ii], (2*lattice_Size + 1)*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

		cudaFree(&H_vals[ii]);
		cudaFree(&H_pos[ii]);

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
		//bubble sorting sucks and I should see if I can find something in std that lets you sort key-value pairs

		for(int jj = 0; jj < (h_H_pos[ii][0] + 1); jj++){
			cout<<"At position: ("<<ii<<", "<<h_H_pos[ii][jj+1]<<": ";
			cout<<h_H_vals[ii][jj].x<<endl;
		}
	}

	UpperHalfToFull(h_H_vals, h_H_pos, dim, lattice_Size);

	dim3 tpb2 = ( tpb, tpb );	
	dim3 bpg2 = ( dim/tpb, dim/tpb );

	num_Elem = 0;

	for(long mm = 0; mm < dim; mm++){
		num_Elem += h_H_pos[mm][0];
		cudaMalloc(&H_pos[mm], h_H_pos[mm][0]*sizeof(long));
		cudaMalloc(&H_vals[mm], (h_H_pos[mm][0] + 1)*sizeof(cuDoubleComplex));
		cudaMemcpy(H_vals[mm], h_H_vals[mm], h_H_pos[mm][0]*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
		cudaMemcpy(H_pos[mm], h_H_pos[mm], (h_H_pos[mm][0] + 1)*sizeof(long), cudaMemcpyHostToDevice);

	}

	FullToCOO<<<bpg2, tpb2>>>(num_Elem, H_vals, H_pos, );

}

int main(){



	long* Bond;
	Bond = (long*)malloc(16*3*sizeof(long));
	
	Bond[0] = 0; 	Bond[1] = 1; 	Bond[2] = 2;	Bond[3] = 3;	Bond[4] = 4;	Bond[5] = 5;
	Bond[6] = 6; 	Bond[7] = 7;	Bond[8] = 8;	Bond[9] = 9;	Bond[10] = 10;	Bond[11] = 11;
	Bond[12] = 12;	Bond[13] = 13;	Bond[14] = 14;	Bond[15] = 15;	Bond[16] = 1; 	Bond[17] = 2;
	Bond[18] = 3; 	Bond[19] = 0;	Bond[20] = 5; 	Bond[21] = 6;	Bond[22] = 7;	Bond[23] = 4;
	Bond[24] = 9;	Bond[25] = 10; 	Bond[26] = 11;	Bond[27] = 8;	Bond[28] = 13;	Bond[29] = 14;
	Bond[30] = 15; 	Bond[31] = 12; 	Bond[32] = 4;	Bond[33] = 5;	Bond[34] = 6;	Bond[35] = 7;
	Bond[36] = 8;	Bond[37] = 9;	Bond[38] = 10;	Bond[39] = 11;	Bond[40] = 12;	Bond[41] = 13;
	Bond[42] = 14;	Bond[43] = 15;	Bond[44] = 0;	Bond[45] = 1;	Bond[46] = 2;	Bond[48] = 3;

	int rtn = ConstructSparseMatrix(0, 16, Bond);

	free(Bond);

	return rtn;
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

//this function takes the upper half form I had from FillSparse and CompressSparse and fills out the bottom half of the matrix - since there are so many comparisons it's probably faster to just do this on CPU
__host__ void UpperHalfToFull(cuDoubleComplex** H_vals, long** H_pos, long dim, int lattice_Size) {

	for(int ii = 0; ii<dim; ii++){

		long size = dim - H_pos[ii][0];
		cuDoubleComplex* temp;
		long* temp_col;

		temp = (cuDoubleComplex*)malloc(size*sizeof(cuDoubleComplex));
		temp_col = (long*)malloc(size*sizeof(long));

		long iter = 0;

		for(int jj = 0; jj<ii; jj++){
			for(int kk = 1; kk <= H_pos[jj][0]; kk++){

				if(H_pos[jj][kk] = ii){
					
					temp[iter] = H_vals[jj][kk-1];
					temp_col[iter] = jj;
					iter++;
				}

			}
		}


		cuDoubleComplex temp_vals[H_pos[ii][0] + iter];
		long temp_pos[H_pos[ii][0] + iter + 1];
		temp_pos[0] = iter + H_pos[ii][0]; //we'll need this number later!
		
		for(int ll = 0; ll < H_pos[ii][0] + iter; ll++){
			if (ll < iter){
				temp_vals[ll] = temp[ll];
				temp_pos[ll+1] = temp_col[ll];

			}
			else{
				temp_vals[ll] = H_vals[ii][ll-iter];
				temp_pos[ll+1] = H_pos[ii][ll-iter + 1];
			}
		}
		
		free(&H_vals[ii]);
		free(&H_pos[ii]);
		
		H_vals[ii] = temp_vals;
		H_pos[ii] = temp_pos;
	}

} 



