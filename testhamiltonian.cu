#include"testhamiltonian.h"

/* NOTE: this function uses FORTRAN style matrices, where the values and positions are stored in a ONE dimensional array! Don't forget this! */

/* Function GetBasis - fills two arrays with information about the basis
Inputs: dim - the initial dimension of the Hamiltonian
	lattice_Size - the number of sites
	Sz - the value of the Sz operator
	basis_Position[] - an empty array that records the positions of the basis
	basis - an empty array that records the basis
Outputs:	basis_Position - a full array now
		basis[] - a full array now

*/
__host__ int GetBasis(long dim, int lattice_Size, int Sz, long basis_Position[], long basis[]){
	unsigned long temp = 0;
	long realdim = 0;

	long iter = 0;

	for (unsigned long i1=0; i1<dim; i1++){
      		temp = 0;
		basis_Position[i1] = -1;
      		for (int sp =0; sp<lattice_Size; sp++){
          		temp += (i1>>sp)&1;
		}  //unpack bra
      		if (temp==(lattice_Size/2+Sz) ){ 
          		basis[iter] = i1;
          		basis_Position[i1] = i1 -1;
			realdim++;
			iter++;
      		}
  	} 

	return realdim;

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

__device__ cuDoubleComplex HDiagPart(const long bra, int lattice_Size, long3* d_Bond, const double JJ){

  int S0b,S1b ;  //spins (bra 
  int T0,T1;  //site
  int P0, P1, P2, P3; //sites for plaquette (Q)
  int s0p, s1p, s2p, s3p;
  cuDoubleComplex valH = make_cuDoubleComplex( 0. , 0.);

  for (int Ti=0; Ti<lattice_Size; Ti++){
    //***HEISENBERG PART

    T0 = (d_Bond[Ti]).x; //lower left spin
    S0b = (bra>>T0)&1;  
    //if (T0 != Ti) cout<<"Square error 3\n";
    T1 = (d_Bond[Ti]).y; //first bond
    S1b = (bra>>T1)&1;  //unpack bra
    valH.x += JJ*(S0b-0.5)*(S1b-0.5);
    T1 = (d_Bond[Ti]).z; //second bond
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

__global__ void SetFirst(long2* H_pos, long stride, long dim, long value){
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i < dim){
      
      (H_pos[ idx(i, 0, stride) ]).y = value;
    }
}

__device__ int nthdigit(long x, int n){
      return (x/10^n)%10;
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


__host__ int ConstructSparseMatrix(int model_Type, int lattice_Size, long* Bond, cuDoubleComplex* hamil_Values, long* hamil_PosRow, long* hamil_PosCol){
	
	unsigned long num_Elem = 0; // the total number of elements in the matrix, will get this (or an estimate) from the input types
	cudaError_t status1, status2, status3;

	long dim = 65536;
	long vdim;
	/*
	switch (model_Type){
		case 0: dim = 65536;
		case 1: dim = 10; //guesses
	}
        */
      
	//for (int ch=1; ch<lattice_Size; ch++) dim *= 2;

        status1 = cudaSetDeviceFlags(cudaDeviceMapHost);

	int stridepos = 2*lattice_Size + 2;
	int strideval = 2*lattice_Size + 1;        

	long basis_Position[dim];
	long basis[dim];

	int Sz = 0;

	//----------------Construct basis and copy it to the GPU --------------------//
	vdim = GetBasis(dim, lattice_Size, Sz, basis_Position, basis);

	cout<<dim<<endl;

	long* d_basis_Position;
	long* d_basis;

	status1 = cudaMalloc(&d_basis_Position, dim*sizeof(long));
	status2 = cudaMalloc(&d_basis, vdim*sizeof(long));

	if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
		cout<<"Memory allocation for basis arrays failed! Error: ";
                cout<<cudaPeekAtLastError()<<endl;
		return 1;
	}


	status1 = cudaMemcpy(d_basis_Position, basis_Position, dim*sizeof(long), cudaMemcpyHostToDevice);
	status2 = cudaMemcpy(d_basis, basis, vdim*sizeof(long), cudaMemcpyHostToDevice);

	if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
		cout<<"Memory copy for basis arrays failed! Error: ";
                cout<<cudaGetErrorString( cudaPeekAtLastError() )<<endl;
		return 1;
	}

	long* d_Bond;
	status1 = cudaMalloc(&d_Bond, 3*lattice_Size*sizeof(long));

	status2 = cudaMemcpy(d_Bond, Bond, 3*lattice_Size*sizeof(long), cudaMemcpyHostToDevice);

	status1 = cudaPeekAtLastError();

	if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
		cout<<"Memory allocation and copy for bond data failed! ";
		return 1;
	}	


	dim3 bpg;
        bpg.x = vdim/64;
        bpg.y = lattice_Size;
	dim3 tpb;
        tpb.x = 64;
        tpb.y = 16; //these are going to need to depend on dim and Nsize

	//--------------Declare the Hamiltonian arrays on the device, and copy the pointers to them to the device -----------//

        cout<<vdim<<endl; 	

	long2* h_H_pos;
	cuDoubleComplex* h_H_vals; 

	h_H_pos = (long2*)malloc(vdim*stridepos*sizeof(long2));
        h_H_vals = (cuDoubleComplex*)malloc(vdim*strideval*sizeof(cuDoubleComplex));

	
        long2* d_H_pos;
        cuDoubleComplex* d_H_vals;

        status1 = cudaMalloc(&d_H_pos, vdim*stridepos*sizeof(long2));
        status2 = cudaMalloc(&d_H_vals, vdim*strideval*sizeof(cuDoubleComplex));

        if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
              cout<<"Memory allocation for device Hamiltonian failed! Error: "<<cudaGetErrorString( cudaPeekAtLastError() )<<endl;
              return 1;
        }

        //the above code should work on devices of any compute capability - YEAAAAH

        cout<<"Running SetFirst"<<endl;	
     
        SetFirst<<<256, 256>>>(d_H_pos, stridepos, vdim, 1); //count the diagonal element
        cout<<cudaGetErrorString( cudaPeekAtLastError() )<<endl;
	// --------------------- Fill up the sparse matrix and compress it to remove extraneous elements ------//
  

        cudaThreadSynchronize();
	
        double JJ = 1.;

        cout<<"Running FillSparse"<<endl;

	FillSparse<<<bpg, tpb>>>(d_basis_Position, d_basis, vdim, d_H_vals, d_H_pos, d_Bond, lattice_Size, JJ);

        if( cudaPeekAtLastError() != 0 ){
		cout<<cudaGetErrorString( cudaPeekAtLastError() )<<endl;
		return 1;
	}
		
	cudaThreadSynchronize(); //need to make sure all elements are initialized before I start compression

	status1 = cudaFree(d_basis);
        status2 = cudaFree(d_basis_Position);
        status3 = cudaFree(d_Bond); // we don't need these later on
	
        if ( (status1 != CUDA_SUCCESS) || 
             (status2 != CUDA_SUCCESS) ||
             (status3 != CUDA_SUCCESS) ){
          cout<<"Freeing bond and basis information failed! Error: "
          <<cudaGetErrorString( cudaPeekAtLastError() )<<endl;
          return 1;
        }

	bpg.x = vdim/32;
        bpg.y = ((2*lattice_Size)/32) + 1;
	tpb.x = 32;
        tpb.y = 32;

        cout<<"Running CompressSparse"<<endl;
       	
	CompressSparse<<<bpg, tpb>>>(d_H_vals, d_H_pos, vdim, lattice_Size);
        
        cudaThreadSynchronize();

        cout<<cudaGetErrorString(cudaPeekAtLastError() )<<endl;
        cout<<"Running SortHamiltonian"<<endl;

        
        SortHamiltonian<<<vdim/2, 64>>>(d_H_pos, d_H_vals, vdim, lattice_Size, 0);
        cudaThreadSynchronize();
        SortHamiltonian<<<vdim/2, 64>>>(d_H_pos, d_H_vals, vdim, lattice_Size, vdim/2);
        cudaThreadSynchronize();

        if (cudaPeekAtLastError != 0){
          cout<<cudaGetErrorString(cudaPeekAtLastError())<<endl;
        }

	long2* buffer_H_pos;
	cuDoubleComplex* buffer_H_vals;

	cudaHostAlloc(&buffer_H_pos, vdim*stridepos*sizeof(long2), cudaHostAllocDefault);
	cudaHostAlloc(&buffer_H_vals, vdim*strideval*sizeof(cuDoubleComplex), cudaHostAllocDefault);

        cudaThreadSynchronize();

	cudaMemcpy(buffer_H_pos, d_H_pos, vdim*stridepos*sizeof(long2), cudaMemcpyDeviceToHost);
	cudaMemcpy(buffer_H_vals, d_H_vals, vdim*strideval*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

	cudaMemcpy(h_H_pos, buffer_H_pos, vdim*stridepos*sizeof(long2), cudaMemcpyHostToHost);
	cudaMemcpy(h_H_vals, buffer_H_vals, vdim*strideval*sizeof(cuDoubleComplex), cudaMemcpyHostToHost);

	cudaFreeHost(buffer_H_pos);
	cudaFreeHost(buffer_H_vals);

	for(int ii = 0; ii < vdim; ii++){
		num_Elem += (h_H_pos[ idx(ii, 0, stridepos) ]).y;
	}

        UpperHalfToFull(h_H_pos, h_H_vals, buffer_H_pos, buffer_H_vals, num_Elem, vdim, lattice_Size);
	
	cudaFreeHost(h_H_vals);
	cudaFreeHost(h_H_pos);

	dim3 tpb2 = ( tpb.x);	
	dim3 bpg2 = ( (2*num_Elem - vdim)/tpb.x );

	status1 = cudaMalloc(&d_H_vals, (2*num_Elem - vdim)*sizeof(cuDoubleComplex));
	status2 = cudaMalloc(&d_H_pos, (2*num_Elem - vdim)*sizeof(long2));

	if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
        	cout<<"Reallocating device Hamiltonian arrays failed! Error: "<< cudaGetErrorString( cudaPeekAtLastError() )<<endl;     
            	return 1;                                                                       
        }

        status1 = cudaMemcpy(d_H_vals, buffer_H_vals, (2*num_Elem - vdim)*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        status2 = cudaMemcpy(d_H_pos, buffer_H_pos, (2*num_Elem - vdim)*sizeof(long2), cudaMemcpyHostToDevice);
                                                                                       
	if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
            cout<<"Hamiltonian copy from host buffer to device failed!"<< cudaGetErrorString( cudaPeekAtLastError() )<<endl;     
            return 1;                                                                        
        }
        	
        cudaFreeHost(buffer_H_vals);
        cudaFreeHost(buffer_H_pos);
        
	status1 = cudaMalloc(&hamil_Values, num_Elem*sizeof(cuDoubleComplex));
	status2 = cudaMalloc(&hamil_PosRow, num_Elem*sizeof(long));
	status3 = cudaMalloc(&hamil_PosCol, num_Elem*sizeof(long));

	if ( (status1 != CUDA_SUCCESS) ||
	     (status2 != CUDA_SUCCESS) ||
	     (status3 != CUDA_SUCCESS) ){
		cout<<"Memory allocation for COO representation failed! Error: "<<cudaGetErrorString( cudaPeekAtLastError() )<<endl;
		return 1;
	}
        cout<<"Running FullToCOO."<<endl;
	
        FullToCOO<<<bpg2, tpb2>>>(num_Elem, d_H_vals, d_H_pos, hamil_Values, hamil_PosRow, hamil_PosCol, vdim); // csr and description initializations happen somewhere else


	cudaFree(d_H_vals); //cleanup
	cudaFree(d_H_pos);
       

	return 0;
}

int main(){
        //cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024); //have to set a heap size or malloc()s on the device will fail
        
	long* Bond;
	Bond = (long*)malloc(16*3*sizeof(long));
	
	Bond[0] = 0; 	Bond[1] = 1; 	Bond[2] = 2;	Bond[3] = 3;	Bond[4] = 4;	Bond[5] = 5;
	Bond[6] = 6; 	Bond[7] = 7;	Bond[8] = 8;	Bond[9] = 9;	Bond[10] = 10;	Bond[11] = 11;
	Bond[12] = 12;	Bond[13] = 13;	Bond[14] = 14;	Bond[15] = 15;	Bond[16] = 1; 	Bond[17] = 2;
	Bond[18] = 3; 	Bond[19] = 0;	Bond[20] = 5; 	Bond[21] = 6;	Bond[22] = 7;	Bond[23] = 4;
	Bond[24] = 9;	Bond[25] = 10; 	Bond[26] = 11;	Bond[27] = 8;	Bond[28] = 13;	Bond[29] = 14;
	Bond[30] = 15; 	Bond[31] = 12; 	Bond[32] = 4;	Bond[33] = 5;	Bond[34] = 6;	Bond[35] = 7;
	Bond[36] = 8;	Bond[37] = 9;	Bond[38] = 10;	Bond[39] = 11;	Bond[40] = 12;	Bond[41] = 13;
	Bond[42] = 14;	Bond[43] = 15;	Bond[44] = 0;	Bond[45] = 1;	Bond[46] = 2;	Bond[47] = 3;

	cuDoubleComplex* hamil_Values;

	long* hamil_PosRow;

	long* hamil_PosCol;


	int rtn = ConstructSparseMatrix(0, 16, Bond, hamil_Values, hamil_PosRow, hamil_PosCol);

	free(Bond);

	return rtn;
}
/* Function FillSparse: this function takes the empty Hamiltonian arrays and fills them up. Each thread in x handles one ket |i>, and each thread in y handles one site T0
Inputs: d_basis_Position - position information about the basis
	d_basis - other basis infos
	d_dim - the number of kets
	H_vals - an array that will store the values
	H_pos - an array that will store the positions of things
	d_Bond - the bond information
	d_lattice_Size - the number of lattice sites
	JJ - the coupling parameter 

*/

__global__ void FillSparse(long* d_basis_Position, long* d_basis, int dim, cuDoubleComplex* H_vals, long2* H_pos, long* d_Bond, int lattice_Size, const double JJ){

	int jj = blockIdx.y; //my indices!
	long ii = threadIdx.x + blockDim.x*blockIdx.x;
        int T0 = threadIdx.y;

        __shared__ long3 tempbond[16];
        __shared__ int count;
	count = 0;
        __shared__ long temppos[16];
        __shared__ cuDoubleComplex tempval[16]; //going to eliminate a huge number of read/writes to d_Bond, H_vals, H_pos in global memory

        if(jj < lattice_Size) {
        	(tempbond[jj]).x = d_Bond[jj];
		(tempbond[jj]).y = d_Bond[lattice_Size + jj];
		(tempbond[jj]).z = d_Bond[2*lattice_Size + jj];
        }
        

        __syncthreads();

        int stridepos = 2*lattice_Size + 2;
        int strideval = 2*lattice_Size + 1;

	int si, sj,sk,sl; //spin operators
	unsigned long tempi, tempj, tempod;
	cuDoubleComplex tempD;

        tempi = d_basis[ii];

	if( ii < dim ){
            if (T0 < lattice_Size){

		//Diagonal part----------------
                if (blockIdx.y == 0){
		    
		    temppos[0] = d_basis_Position[tempi];

		    tempval[0] = HDiagPart(tempi, lattice_Size, tempbond, JJ);

                    H_vals[ idx(ii, 0, strideval) ] = tempval[0];
                    (H_pos[ idx(ii, 1, stridepos) ]).y = temppos[0];
                    (H_pos[ idx(ii, 1, stridepos) ]).x = ii;
                }
                

		//-------------------------------
		//Horizontal bond ---------------
		si = (tempbond[T0]).x;
		tempod = tempi;
		sj = (tempbond[T0]).y;
	
		tempod ^= (1<<si);   //toggle bit 
		tempod ^= (1<<sj);   //toggle bit 

		if (d_basis_Position[tempod] > ii){ //build only upper half of matrix
        		temppos[T0] = d_basis_Position[tempod];
        		tempval[T0] = HOffBondX(T0,tempi, JJ);
			atomicAdd(&count,1); 
      		}

		else {
			temppos[T0] = -1;
			tempval[T0] = make_cuDoubleComplex(0., 0.);
		}

                __syncthreads();

                H_vals[ idx(ii, 2*T0, strideval) ] = tempval[T0];
                (H_pos[ idx(ii, 2*T0, stridepos) ]).y = temppos[T0];
                (H_pos[ idx(ii, 2*T0, stridepos) ]).x = ii;

		//Vertical bond -----------------
		tempod = tempi;
      		sj = (tempbond[T0]).z;

      		tempod ^= (1<<si);   //toggle bit 
     		tempod ^= (1<<sj);   //toggle bit
                 
      		if (d_basis_Position[tempod] > ii){ 
        		temppos[T0] = d_basis_Position[tempod];
   
        		tempval[T0] = HOffBondY(T0,tempi, JJ);
			atomicAdd(&count,1);
      		}

		else {
			temppos[T0] = -1;
			tempval[T0] = make_cuDoubleComplex(0., 0.);
		}

                //time to write back to global memory
                __syncthreads;
                (H_pos[ idx(ii, 0, stridepos) ]).x = ii;
                (H_pos[ idx(ii, 0, stridepos) ]).y, count;
                
                (H_pos[ idx(ii, 2*T0 + 1, stridepos) ]).x = ii;
                (H_pos[ idx(ii, 2*T0 + 1, stridepos) ]).y = temppos[T0];
                
                H_vals[ idx(ii, 2*T0 + 1, strideval) ] = tempval[T0];
                                 
            }//end of T0 

	}//end of ii
}//end of FillSparse

/* Function: CompressSparse - this function takes the sparse matrix with lots of "buffer" memory sitting on the end of each array, and compresses it down to get rid of the extra memory
Inputs:	H_vals - an array of arrays of Hamiltonian values
	H_pos - an array of arrays of value positions in columns
	d_dim - the dimension of the Hamiltonian, stored on the device
	lattice_Size - the number of lattice sites
Outputs: H_vals - an array of smaller arrays than before
	 H_pos - see above

*/
__global__ void CompressSparse(cuDoubleComplex* H_vals, long2* H_pos, long d_dim, const int lattice_Size){

	long row = blockDim.x*blockIdx.x + threadIdx.x;
        long col = blockDim.y*blockIdx.y + threadIdx.y;

	__shared__ int iter;
	iter = 0;

	if (row < d_dim){

		// the basic idea here is to have each x thread go to the ith row, and each y thread go to the jth element of that row. then using a set of __shared__ temp arrays, we read in the Hamiltonian values and do our comparisons n stuff on them


		const int size1 = 2*lattice_Size + 2;
		const int size2 = 2*lattice_Size + 1;
	
		__shared__ long s_H_pos[34]; //hardcoded for now because c++ sucks - should be able to change this later by putting all these functions in a separate .cu that doesn't use c++ functionality
		__shared__ cuDoubleComplex s_H_vals[33];
		

		if (col < size2){
			s_H_pos[col] = (H_pos[ idx( row, col, size1 ) ]).y;
			s_H_vals[col] = H_vals[ idx( row, col, size2) ];
                
                }

		if (col == size2){
			s_H_pos[col] = (H_pos[ idx(row, col, size1) ]).y;
		} //loading the Hamiltonian information into shared memory

                

		__syncthreads(); // have to make sure all loading is done before we start anything else


                

	        if (col < size2){
			if( (s_H_pos[col+1] != -1) ){
				
                                (H_pos[ idx(row, iter+1, size1) ]).y = s_H_pos[col+1];
                                H_vals[ idx(row, iter, size2) ] = s_H_vals[col];
                                //printf("%d %d %d \t", row, col, s_H_pos[col + 1]);
				atomicAdd(&iter,1);
			}
		 
		}

		
	}
}

//this function takes the upper half form I had from FillSparse and CompressSparse and fills out the bottom half of the matrix - since there are so many comparisons it's probably faster to just do this on CPU
__host__ void UpperHalfToFull(long2* H_pos, cuDoubleComplex* H_vals, long2* buffer_pos, cuDoubleComplex* buffer_val, long num_Elem, long dim, int lattice_Size) {

	int stridepos = 2*lattice_Size + 2;
	int strideval = 2*lattice_Size + 1;

	cudaHostAlloc(&buffer_pos, (2*num_Elem - dim)*sizeof(long2), cudaHostAllocDefault);
	cudaHostAlloc(&buffer_val, (2*num_Elem - dim)*sizeof(cuDoubleComplex), cudaHostAllocDefault);

	unsigned long start = 0;

	for(long ii = 0; ii<dim; ii++){
                //cout<<start<<endl;         
		long size = dim - (H_pos[ idx(ii, 0, stridepos) ]).y ; // the maximum number of nonzero elements we could pull from the columns
		cuDoubleComplex* temp;
		long* temp_col;

		temp = (cuDoubleComplex*)malloc(size*sizeof(cuDoubleComplex));
		temp_col = (long*)malloc(size*sizeof(long));

		long iter = 0;

		for(int jj = 0; jj<ii; jj++){
			for(int kk = 1; kk <= (H_pos[ idx(jj, 0, stridepos) ]).y; kk++){

				if((H_pos[ idx(jj, kk, stridepos) ]).y = ii){
					cout<<iter<<endl;
					temp[iter] = H_vals[ idx(jj, kk-1, strideval) ];
					temp_col[iter] = jj;
					iter++;
				}

			}
		}


		cuDoubleComplex temp_vals[(H_pos[ idx(ii, 0, stridepos) ]).y + iter];
		long temp_pos[(H_pos[ idx(ii, 0, stridepos) ]).y + iter + 1];
		temp_pos[0] = iter + (H_pos[ idx(ii , 0, stridepos) ] ).y; //we'll need this number later!
		//cout<<iter<<" "<<(H_pos[ idx(ii, 0, stridepos) ].y)<<endl;
		for(int ll = 0; ll < (H_pos[ idx(ii, 0, stridepos) ]).y + iter; ll++){
			if (ll < iter){
				temp_vals[ll] = temp[ll];
				temp_pos[ll+1] = temp_col[ll];

			}
			else{
				temp_vals[ll] = H_vals[ idx(ii, ll - iter, strideval) ];
				temp_pos[ll+1] = (H_pos[ idx(ii, ll - iter + 1, stridepos) ]).y;
			}

                        
			buffer_val[ start + ll ] = temp_vals[ll];
			(buffer_pos[ start + ll ]).x = ii;
			(buffer_pos[ start + ll ]).y = temp_pos[ll + 1];
		}
				
		start += temp_pos[0];
	}

} 

/*Function: FullToCOO - takes a full sparse matrix and transforms it into COO format
Inputs - num_Elem - the total number of nonzero elements
	 H_vals - the Hamiltonian values
	 H_pos - the Hamiltonian positions
	 hamil_Values - a 1D array that will store the values for the COO form

*/
__global__ void FullToCOO(long num_Elem, cuDoubleComplex* H_vals, long2* H_pos, cuDoubleComplex* hamil_Values, long* hamil_PosRow, long* hamil_PosCol, long dim){

	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int j = threadIdx.x;

	long size = 2*num_Elem - dim;

	long start = 0;

	__shared__ long2 s_H_pos[256]; //hardcoded for now because c++ sucks
	__shared__ cuDoubleComplex s_H_vals[256];

	if (i < size){
		(s_H_pos[j]).x = (H_pos[i]).x;
		(s_H_pos[j]).y = (H_pos[i]).y;
		s_H_vals[j] = H_vals[i];
	
		__syncthreads();
		
		hamil_Values[i] = s_H_vals[j];
		hamil_PosRow[i] = (s_H_pos[j]).x;
		hamil_PosCol[i] = (s_H_pos[j]).y;
		
	}
}

__global__ void SortHamiltonian( long2* H_pos, cuDoubleComplex* H_vals, long dim, int lattice_Size, long start){
	long i = blockIdx.x + start;
	int j = threadIdx.x;

	__shared__ int maxpos;
	maxpos = 0;
	__shared__ int maxnumdigits;
	__shared__ int rowlength;

	__shared__ hamstruct temparray[33];
	__shared__ int sortarray[10];

	if ( i < dim){

      		rowlength = (H_pos[ idx(i, 0, 2*lattice_Size + 2) ]).y;
      	
		if (j < rowlength ){
      			(temparray[j]).position  = ( H_pos[ idx(i, j + 1, 2*lattice_Size + 2) ] ).y;
              		(temparray[j]).value = H_vals[ idx(i, j, 2*lattice_Size + 1) ];
                        //printf("%d \n", (temparray[j]).position);
              		sortarray[j] = 0;

              		atomicMax(&maxpos, (temparray[j]).position ); //computes maximum using bithax
        	       	__syncthreads(); //need to finish loading everything            
	       		maxnumdigits = floor ( log10f( maxpos ) ) + 1; 

			maxpos = 0; //maxpos now functions as total
	
			__syncthreads();

			for(int k = 1; k < maxnumdigits; k++){
				int key = nthdigit(temparray[j].position, 1);
				sortarray[key] = sortarray[key] + 1;
				
  			
				if ( j < 10 ){				
					int temp = sortarray[j];
					sortarray[j] = maxpos;
					atomicAdd(&maxpos, temp);
				}

				temparray[sortarray[key]] = temparray[j];
				sortarray[key] = sortarray[key] + 1;

                                __syncthreads();

			}//k

			(H_pos[ idx(i, j, 2*lattice_Size + 2) ]).y = temparray[j].position;
			H_vals[ idx(i, j, 2*lattice_Size + 1) ] = temparray[j].value;
			//passing the sorted array back to global memory
					
		}//j

	}//i
}


