#include"hamiltonian.h"


/* Function GetBasis - fills two arrays with information about the basis
Inputs: dim - the initial dimension of the Hamiltonian
	lattice_Size - the number of sites
	Sz - the value of the Sz operator
	basis_Position[] - an empty array that records the positions of the basis
	basis - an empty array that records the basis
Outputs:	basis_Position - a full array now
		basis[] - a full array now

*/
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

/* Function HOffBondX
Inputs: si - the spin operator in the x direction
        bra - the state
        JJ - the coupling constant
Outputs:  valH - the value of the Hamiltonian 

*/

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

/* Function: SetFirst - sets the first row elements of a matrix to some value

Inputs: H_pos - an array on the device whose first row elements will be changed
	dim - the number of rows in H_pos
	value - the value we want to set the first elements to
*/

__global__ void SetFirst(long** H_pos, long dim, long value){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i < dim){
      
      H_pos[i][0] = value;
    }
}
/* Function: ConstructSparseMatrix:

Inputs: model_Type - tells this function how many elements there could be, what generating functions to use, etc. Presently only supports Heisenberg
	lattice_Size - the number of lattice sites
	Bond - the bond values ??
	hamil_Values - an empty pointer for a device array containing the values 
	hamil_PosRow - an empty pointer for a device array containing the locations of each value in a row
	hamil_PosCol - an empty pointer to a device array containing the locations of each values in a column

Outputs:  hamil_Values - a pointer to a device array containing the values 
	hamil_PosRow - a pointer to a device array containing the locations of each value in a row
	hamil_PosCol - a pointer to a device array containing the locations of each values in a column

*/


int ConstructSparseMatrix(int model_Type, int lattice_Size, long* Bond, cuDoubleComplex* hamil_Values, long* hamil_PosRow, long* hamil_PosCol){
	
	unsigned long num_Elem = 0; // the total number of elements in the matrix, will get this (or an estimate) from the input types
	cudaError_t status1, status2, status3;

	int dim;
	
	switch (model_Type){
		case 0: dim = 2;
		case 1: dim = 10; //guesses
	}

	
	long* d_Bond;
	status1 = cudaMalloc(&d_Bond, 3*lattice_Size*sizeof(long));

	status2 = cudaMemcpy(d_Bond, Bond, 3*lattice_Size*sizeof(long), cudaMemcpyHostToDevice);

	status1 = cudaPeekAtLastError();

	/*if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
		cout<<"Memory allocation and copy for bond data failed!";
		return 1;
	}*/

	if (status1 != CUDA_SUCCESS){
		cout<<"Bond data allocation error: "<<cudaGetErrorString( status1 )<<endl;
		return 1;
	}

	if (status2 != CUDA_SUCCESS){
		cout<<"Bond data copy error: "<<cudaGetErrorString( status2 )<<endl;
		return 1;
	}


	for (int ch=1; ch<lattice_Size; ch++) dim *=2;

        
	long basis_Position[dim];
	long basis[dim];

	int Sz = 0;

	//----------------Construct basis and copy it to the GPU --------------------//
	GetBasis(dim, lattice_Size, Sz, basis_Position, basis);

	long* d_basis_Position;
	long* d_basis;

	status1 = cudaMalloc(&d_basis_Position, dim*sizeof(long));
	status2 = cudaMalloc(&d_basis, dim*sizeof(long));

	if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
		cout<<"Memory allocation for basis arrays failed! Error: ";
                cout<<cudaPeekAtLastError()<<endl;
		return 1;
	}

	status1 = cudaMemcpy(d_basis_Position, basis_Position, dim*sizeof(long), cudaMemcpyHostToDevice);
	status2 = cudaMemcpy(d_basis, basis, dim*sizeof(long), cudaMemcpyHostToDevice);

	if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
		cout<<"Memory copy for basis arrays failed! Error: ";
                cout<<cudaGetErrorString( cudaPeekAtLastError() )<<endl;
		return 1;
	}

	dim3 bpg = (dim/256, 16, 1);
	dim3 tpb = (256, 1, 1); //these are going to need to depend on dim and Nsize

	//--------------Declare the Hamiltonian arrays on the device, and copy the pointers to them to the device -----------//

	long** h_H_pos;
	cuDoubleComplex** h_H_vals; 

	//status1 = cudaMalloc(&H_pos, dim*sizeof(long*));
	//status2 = cudaMalloc(&H_vals, dim*sizeof(cuDoubleComplex*));

        h_H_pos = (long**)malloc(dim*sizeof(long*));
        h_H_vals = (cuDoubleComplex**)malloc(dim*sizeof(cuDoubleComplex*));

	if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
		cout<<"Memory allocation for upper half arrays failed! Error: ";
                cout<<cudaGetErrorString( cudaPeekAtLastError() )<<endl;
		return 1;
	}

	
        long** d_H_pos;
        cuDoubleComplex** d_H_vals;

        status1 = cudaMalloc(&d_H_pos, dim*sizeof(long*));
        status2 = cudaMalloc(&d_H_vals, dim*sizeof(cuDoubleComplex*));

        if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
              cout<<"Memory allocation for device Hamiltonian failed! Error: "<<cudaGetErrorString( cudaPeekAtLastError() )<<endl;
              return 1;
        }

	/*for(long i = 0; i<dim; i++){
		status1 = cudaMalloc(&(h_H_vals[i]), (2*(lattice_Size)+1)*sizeof(cuDoubleComplex));
		status2 = cudaMalloc(&(h_H_pos[i]), (2*(lattice_Size)+2)*sizeof(long));
                if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
                      cout<<"Memory allocation for "<<i<<"th Hamiltonian arrays failed! Error: "<<cudaPeekAtLastError()<<endl;
                      return 1;
                }

      }



      status1 = cudaMemcpy(d_H_pos, h_H_pos, dim*sizeof(long*), cudaMemcpyHostToDevice);
      status2 = cudaMemcpy(d_H_vals, h_H_vals, dim*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);

      if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
          cout<<"Copy of Hamiltonian array pointers from host to device failed! Error: "<<cudaPeekAtLastError()<<endl;
          return 1;
      }

	This code is for the case where we have a device of compute capability <2.0. It's super slow T_T

      */

      CDCarraysalloc<<<(dim/256), 256 >>>(d_H_vals, dim, 2*(lattice_Size) + 1, 0);
      cudaThreadSynchronize();
      longarraysalloc<<<(dim/256), 256>>>(d_H_pos, dim, 2*(lattice_Size) + 2, 0);
      cudaThreadSynchronize();
      
      status1 = cudaMemcpy(h_H_vals, d_H_vals, dim*sizeof(cuDoubleComplex*), cudaMemcpyDeviceToHost);
      status2 = cudaMemcpy(h_H_pos, d_H_pos, dim*sizeof(long*), cudaMemcpyDeviceToHost);    
      
      if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
          cout<<"Copy of Hamiltonian array pointers from host to device failed! Error: "<<cudaGetErrorString( cudaPeekAtLastError() )<<endl;
          return 1;
      }
     
      SetFirst<<<(dim/256), tpb>>>(d_H_pos, dim, 1); 
      /*	
      for(int jj = 0; jj < dim; jj++){


		status3 = cudaMemset((h_H_pos[jj]), 1l, sizeof(long));
                if (status3 != CUDA_SUCCESS){
                    cout<<"Counting the diagonal element failed! Error: "<<cudaPeekAtLastError()<<endl;
                    return 1;
                }
      } //counting the diagonal element
      */

	// --------------------- Fill up the sparse matrix and compress it to remove extraneous elements ------//
      	double JJ = 1.;

        cudaThreadSynchronize();
	

	FillSparse<<<bpg, tpb>>>(d_basis_Position, d_basis, dim, d_H_vals, d_H_pos, d_Bond, lattice_Size, JJ);

	cudaThreadSynchronize(); //need to make sure all elements are initialized before I start compression
	
	bpg = (dim/256, (((2*lattice_Size + 2)/32) + 1) , 1);
	tpb = (256, 32, 1);
	
	CompressSparse<<<bpg, tpb>>>(d_H_vals, d_H_pos, dim, lattice_Size);
        cudaThreadSynchronize();
	
	long** buffer_H_pos; //created these arrays to hold pointers for me - may be able to remove them later
	cuDoubleComplex** buffer_H_vals;

	status1 = cudaMallocHost(&buffer_H_pos, dim*sizeof(long*));
	status2 = cudaMallocHost(&buffer_H_vals, dim*sizeof(cuDoubleComplex*));

        if ( (status1 != CUDA_SUCCESS ) || (status2 != CUDA_SUCCESS ) ){
            cout<<"Memory allocation for Hamiltonian arrays on host failed!"<< cudaGetErrorString(cudaPeekAtLastError())<<endl;
            return 1;
        }

	//----Copy over the Hamiltonian arrays and sort them-------------//

	for(int ii = 0; ii < dim; ii++){
                long* temp;
                temp = (long*)malloc(sizeof(long));

                status3 = cudaMemcpy(temp, &d_H_pos[ii][0], sizeof(long), cudaMemcpyDeviceToHost);

                status1 = cudaMallocHost(h_H_pos + ii, (*temp+1)*sizeof(long));
                status2 = cudaMallocHost(h_H_vals + ii, (*temp)*sizeof(cuDoubleComplex));

                if ( (status1 != CUDA_SUCCESS) ||
                     (status2 != CUDA_SUCCESS) ||
                     (status3 != CUDA_SUCCESS) ){
                    
                    cout<<"Memory allocation for "<<ii<<"th host Hamiltonian arrays failed! Error: "<<cudaGetErrorString( cudaPeekAtLastError() )<<endl;
                    return 1;
                }

		status1 = cudaMemcpy(h_H_pos[ii], d_H_pos[ii], (*temp + 1)*sizeof(long), cudaMemcpyDeviceToHost);
		status2 = cudaMemcpy(h_H_vals[ii], d_H_vals[ii], (*temp)*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

                if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
                    cout<<"Copying "<<ii<<" th Hamiltonian arrays from device to host failed! Error: "<<cudaGetErrorString( cudaPeekAtLastError() )<<endl;
                    return 1;
                }

		cudaFree(&d_H_vals[ii]);
		cudaFree(&d_H_pos[ii]);

                hamstruct temphamstruct;

                vector<hamstruct> sortcontainer;

                for (uint jj = 0; jj << h_H_pos[ii][0]; jj++){
                  	temphamstruct.position = h_H_pos[ii][jj+1];
                  	temphamstruct.value = h_H_vals[ii][jj];

                  	sortcontainer.push_back(temphamstruct);

                }

                sort(sortcontainer.begin(), sortcontainer.end());

		for(uint kk = 0; kk < sortcontainer.size(); kk++){
			
			h_H_pos[ii][kk + 1] = (sortcontainer.at(kk)).position;
			h_H_vals[ii][kk] = (sortcontainer.at(kk)).value;
		}


		for(uint ll = 0; ll < (h_H_pos[ii][0] + 1); ll++){
			cout<<"At position: ("<<ii<<", "<<h_H_pos[ii][ll+1]<<": ";
			cout<<h_H_vals[ii][ll].x<<endl;
		}

                free(&temp);
	}

	UpperHalfToFull(h_H_vals, h_H_pos, dim, lattice_Size);

	dim3 tpb2 = ( tpb.x, tpb.x );	
	dim3 bpg2 = ( dim/tpb.x, dim/tpb.x );


	for(long mm = 0; mm < dim; mm++){ //this for loop allocates just enough memory in the device arrays to hold the full row, copies the data, and also finds the true number of nonzero elements
		num_Elem += h_H_pos[mm][0];

                //long* temppos;
                //cuDoubleComplex* tempval;
		
                status1 = cudaMalloc(&buffer_H_pos[mm], (h_H_pos[mm][0]+ 1)*sizeof(long));
		status2 = cudaMalloc(&buffer_H_vals[mm], (h_H_pos[mm][0])*sizeof(cuDoubleComplex));
		
                if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
                    cout<<"Memory allocation for "<<mm<<"th arrays on device failed! Error: "<<cudaGetErrorString( cudaPeekAtLastError() )<<endl;
                    return 1;
                }

                status1 = cudaMemcpy(buffer_H_pos[mm], h_H_pos[mm], (h_H_pos[mm][0] + 1)*sizeof(long), cudaMemcpyHostToHost);
                status2 = cudaMemcpy(buffer_H_vals[mm], h_H_vals[mm], (h_H_pos[mm][0])*sizeof(cuDoubleComplex), cudaMemcpyHostToHost);
                

                if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
                    cout<<"Array copy from host to buffer failed! Error: "<<cudaGetErrorString( cudaPeekAtLastError() )<<endl;
                    return 1;
                }

                
	}

        status1 = cudaMemcpy(d_H_vals, buffer_H_vals, dim*sizeof(cuDoubleComplex*), cudaMemcpyHostToDevice);
        status2 = cudaMemcpy(d_H_pos, buffer_H_pos, dim*sizeof(long*), cudaMemcpyHostToDevice);
                                                                                        if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
            cout<<"Pointer memory copy from host buffer to device failed!"<< cudaGetErrorString( cudaPeekAtLastError() )<<endl;     
            return 1;                                                                                       
        }
        
        
	status1 = cudaMalloc(&hamil_Values, num_Elem*sizeof(cuDoubleComplex));
	status2 = cudaMalloc(&hamil_PosRow, num_Elem*sizeof(long));
	status3 = cudaMalloc(&hamil_PosCol, num_Elem*sizeof(long));

	if ( (status1 != CUDA_SUCCESS) ||
	     (status2 != CUDA_SUCCESS) ||
	     (status3 != CUDA_SUCCESS) ){
		cout<<"Memory allocation for COO representation failed! Error: "<<cudaGetErrorString( cudaPeekAtLastError() )<<endl;
		return 1;
	}
	
        FullToCOO<<<bpg, tpb>>>(num_Elem, d_H_vals, d_H_pos, hamil_Values, hamil_PosRow, hamil_PosCol, dim); // csr and description initializations happen somewhere else

	for(int nn = 0; nn < dim; nn++){
		cudaFree(&d_H_vals[nn]);
		cudaFree(&d_H_pos[nn]);
                cudaFreeHost(&h_H_vals[nn]);
                cudaFreeHost(&h_H_pos[nn]);
                cudaFreeHost(&buffer_H_vals[nn]);
                cudaFreeHost(&buffer_H_pos[nn]);
	}

	cudaFree(&d_H_vals); //cleanup
	cudaFree(&d_H_pos);

	cudaFreeHost(&h_H_vals);
	cudaFreeHost(&h_H_pos);
        cudaFreeHost(&buffer_H_vals);
        cudaFreeHost(&buffer_H_pos);
        cudaFree(&d_basis);
        cudaFree(&d_basis_Position);
        cudaFree(&d_Bond);

	return 0;
}

int main(){
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024); //have to set a heap size or malloc()s on the device will fail
        
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

	cuDoubleComplex* hamil_Values;

	long* hamil_PosRow;

	long* hamil_PosCol;


	int rtn = ConstructSparseMatrix(0, 16, Bond, hamil_Values, hamil_PosRow, hamil_PosCol);

	free(Bond);

	return rtn;
}

__global__ void FillSparse(long* d_basis_Position, long* d_basis, int d_dim, cuDoubleComplex** H_vals, long** H_pos, long* d_Bond, int d_lattice_Size, const double JJ){

	int T0 = blockIdx.y; //my indices!
	int ii = threadIdx.x + blockDim.x*blockIdx.x;

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

/* Function: CompressSparse - this function takes the sparse matrix with lots of "buffer" memory sitting on the end of each array, and compresses it down to get rid of the extra memory
Inputs:	H_vals - an array of arrays of Hamiltonian values
	H_pos - an array of arrays of value positions in columns
	d_dim - the dimension of the Hamiltonian, stored on the device
	lattice_Size - the number of lattice sites
Outputs: H_vals - an array of smaller arrays than before
	 H_pos - see above

*/
__global__ void CompressSparse(cuDoubleComplex** H_vals, long** H_pos, int d_dim, const int lattice_Size){

	int row = blockDim.x*blockIdx.x + threadIdx.x;
	int col = blockDim.y*blockIdx.y + threadIdx.y;

	int iter = 0;

	if (row < d_dim){

		// the basic idea here is to have each x thread go to the ith row, and each y thread go to the jth element of that row. then using a set of __shared__ temp arrays, we read in the Hamiltonian values and do our comparisons n stuff on them
		__shared__ int count;
		count = H_pos[row][0];

		const int size1 = 2*lattice_Size + 2;
		const int size2 = 2*lattice_Size + 1;
	
		__shared__ long s_H_pos[34]; //hardcoded for now because c++ sucks
		__shared__ cuDoubleComplex s_H_vals[33];
		

		if (col < size2){
			s_H_pos[col] = H_pos[row][col];
			s_H_vals[col] = H_vals[row][col];
		}

		if (col == size2){
			s_H_vals[col] = H_vals[row][col];
		} //loading the Hamiltonian information into shared memory

		__syncthreads(); // have to make sure all loading is done before we start anything else

		long* temp_pos = (long*)malloc((count + 1)*sizeof(long));
		cuDoubleComplex* temp_val = (cuDoubleComplex*)malloc((count)*sizeof(cuDoubleComplex));
                /*
                if ( temp_pos == (long*)NULL){
                  printf("Allocation of temp_pos in iteration %d failed! \n", i);
                }	

                if ( temp_val == (cuDoubleComplex*)NULL){
                  printf("Allocation of temp_val in iteration %d failed! \n", i);
                }
		*/
		temp_pos[0] = count;

		for(int j = 0; j < size2; j++){
			if( (s_H_pos[j+1] != -1) ){
				temp_pos[iter+1] = s_H_pos[j+1];
				temp_val[iter] = s_H_vals[j];
				iter++;
			}
		 
		}

		free(H_pos[row]); //switching out the old ones for the new
		free(H_vals[row]);
		
		H_pos[row] = temp_pos;
		H_vals[row] = temp_val;

		
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

/*Function: FullToCOO - takes a full sparse matrix and transforms it into COO format
Inputs - num_Elem - the total number of nonzero elements
	 H_vals - the Hamiltonian values
	 H_pos - the Hamiltonian positions
	 hamil_Values - a 1D array that will store the values for the COO form

*/
__global__ void FullToCOO(long num_Elem, cuDoubleComplex** H_vals, long** H_pos, cuDoubleComplex* hamil_Values, long* hamil_PosRow, long* hamil_PosCol, long dim){

	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int j = threadIdx.y + blockDim.x*blockIdx.y;

	const int size = H_pos[i][0];

	long start = 0;

	__shared__ long s_H_pos[34]; //hardcoded for now because c++ sucks
	__shared__ cuDoubleComplex s_H_vals[33];

	if (j < size){
			s_H_pos[j] = H_pos[i][j];
			s_H_vals[j] = H_vals[i][j];
	}

	if (j == size){
			s_H_vals[j] = H_vals[i][j];
	} //loading the Hamiltonian information into shared memory


	if (i < size){
		for(int k = 0; k < i; k++){
			start += H_pos[k][0];
		} //need to know where to start sticking values into the COO arrays


		if(j < size){
			hamil_Values[start + j] = s_H_vals[j];
			hamil_PosRow[start + j] = i;
			hamil_PosCol[start + j] = s_H_pos[j+1];
		}
	}
}

