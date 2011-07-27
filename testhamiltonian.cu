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
__host__ int GetBasis(int dim, int lattice_Size, int Sz, int basis_Position[], int basis[]){
	unsigned int temp = 0;
	int realdim = 0;

	for (unsigned int i1=0; i1<dim; i1++){
      		temp = 0;
		basis_Position[i1] = -1;
      		for (int sp =0; sp<lattice_Size; sp++){
          		temp += (i1>>sp)&1;
		}  //unpack bra
      		if (temp==(lattice_Size/2+Sz) ){ 
          		basis[realdim] = i1;
          		basis_Position[i1] = realdim;
			realdim++;
			//cout<<basis[realdim]<<" "<<basis_Position[i1]<<endl;
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

__device__ cuDoubleComplex HOffBondX(const int si, const int bra, const double JJ){

	cuDoubleComplex valH;
  	//int S0, S1;
  	//int T0, T1;

  	valH = make_cuDoubleComplex( JJ*0.5 , 0.); //contribution from the J part of the Hamiltonian

  	return valH;


} 

__device__ cuDoubleComplex HOffBondY(const int si, const int bra, const double JJ){

	cuDoubleComplex valH;
  	//int S0, S1;
  	//int T0, T1;

  	valH = make_cuDoubleComplex( JJ*0.5 , 0. ); //contribution from the J part of the Hamiltonian

  	return valH;


}

__device__ cuDoubleComplex HDiagPart(const int bra, int lattice_Size, int3* d_Bond, const double JJ){

  int S0b,S1b ;  //spins (bra 
  int T0,T1;  //site
  //int P0, P1, P2, P3; //sites for plaquette (Q)
  //int s0p, s1p, s2p, s3p;
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


__host__ int ConstructSparseMatrix(int model_Type, int lattice_Size, int* Bond, cuDoubleComplex* hamil_Values, int* hamil_PosRow, int* hamil_PosCol, int* vdim, double JJ, int Sz){


        std::ofstream fout;
        fout.open("testhamiltonian.log"); //creating a log file to store timing information

	int num_Elem = 0; // the total number of elements in the matrix, will get this (or an estimate) from the input types
	cudaError_t status1, status2, status3;

	int dim = 65536;
	
	/*
	switch (model_Type){
		case 0: 
			dim = 65536;
			break;
		case 1: dim = 10; //guesses
	}
        */
      
	//for (int ch=1; ch<lattice_Size; ch++) dim *= 2;

	int stridepos = 2*lattice_Size + 2;
	int strideval = 2*lattice_Size + 1;        

	int basis_Position[dim];
	int basis[dim];

	int Sz = 0;

	//----------------Construct basis and copy it to the GPU --------------------//
	cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start,0);
        *vdim = GetBasis(dim, lattice_Size, Sz, basis_Position, basis);
        cudaEventRecord(stop,0);

        float elapsed;

        cudaEventElapsedTime(&elapsed, start, stop);

        fout<<"Run time for GetBasis: "<<elapsed<<std::endl;

	int* d_basis_Position;
	int* d_basis;

	status1 = cudaMalloc(&d_basis_Position, dim*sizeof(int));
	status2 = cudaMalloc(&d_basis, *vdim*sizeof(int));

	if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
		std::cout<<"Memory allocation for basis arrays failed! Error: ";
                std::cout<<cudaPeekAtLastError()<<std::endl;
		return 1;
	}

	status1 = cudaMemcpy(d_basis_Position, basis_Position, dim*sizeof(int), cudaMemcpyHostToDevice);
	status2 = cudaMemcpy(d_basis, basis, *vdim*sizeof(int), cudaMemcpyHostToDevice);

	if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
		std::cout<<"Memory copy for basis arrays failed! Error: ";
                std::cout<<cudaGetErrorString( cudaPeekAtLastError() )<<std::endl;
		return 1;
	}

	int* d_Bond;
	status1 = cudaMalloc(&d_Bond, 3*lattice_Size*sizeof(int));

	status2 = cudaMemcpy(d_Bond, Bond, 3*lattice_Size*sizeof(int), cudaMemcpyHostToDevice);

	if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
		std::cout<<"Memory allocation and copy for bond data failed! Error: ";
                std::cout<<cudaGetErrorString( cudaPeekAtLastError() )<<std::endl;
		return 1;
	}	

	dim3 bpg;

	if (*vdim <= 65336);
                bpg.x = *vdim;
        
	dim3 tpb;
        tpb.x = 32;
        //these are going to need to depend on dim and Nsize

	//--------------Declare the Hamiltonian arrays on the device, and copy the pointers to them to the device -----------//
	
        int2* d_H_pos;
        cuDoubleComplex* d_H_vals;

        status1 = cudaMalloc(&d_H_pos, *vdim*stridepos*sizeof(int2));
        status2 = cudaMalloc(&d_H_vals, *vdim*strideval*sizeof(cuDoubleComplex));

        if ( (status1 != CUDA_SUCCESS) || (status2 != CUDA_SUCCESS) ){
              std::cout<<"Memory allocation for device Hamiltonian failed! Error: "<<cudaGetErrorString( cudaPeekAtLastError() )<<std::endl;
              return 1;
        }

        //the above code should work on devices of any compute capability - YEAAAAH

	// --------------------- Fill up the sparse matrix and compress it to remove extraneous elements ------//

        //std::cout<<"Running FillSparse"<<std::endl;
      
        cudaEventRecord(start, 0);
	FillSparse<<<bpg, tpb>>>(d_basis_Position, d_basis, *vdim, d_H_vals, d_H_pos, d_Bond, lattice_Size, JJ);

        if( cudaPeekAtLastError() != 0 ){
		std::cout<<"Error in FillSparse! Error: "<<cudaGetErrorString( cudaPeekAtLastError() )<<std::endl;
		return 1;
	}
		
	cudaThreadSynchronize(); //need to make sure all elements are initialized before I start compression
        cudaEventRecord(stop, 0);
        cudaEventElapsedTime(&elapsed, start, stop);

        fout<<"Runtime for FillSparse: "<<elapsed<<std::endl;

	int* num_ptr;
	cudaGetSymbolAddress((void**)&num_ptr, (const char*)"d_num_Elem");
	cudaMemset(num_ptr, 0, sizeof(int));

	thrust::device_ptr<int> thrust_num_ptr(num_ptr);
	thrust::device_vector<int> num_array(*vdim);
        int* num_array_ptr = raw_pointer_cast(&num_array[0]);

        Copy<<<*vdim/512 + 1, 512>>>(num_array_ptr, d_H_pos, lattice_Size, vdim);
	cudaThreadSynchronize();
        *thrust_num_ptr = thrust::reduce(num_array.begin(), num_array.end());
	*thrust_num_ptr = 2*(*thrust_num_ptr) - *vdim;

	cudaMemcpy(&num_Elem, num_ptr, sizeof(int), cudaMemcpyDeviceToHost);
	std::cout<<"Number of elements from thrust : "<<num_Elem<<std::endl;

	status1 = cudaFree(d_basis);
        status2 = cudaFree(d_basis_Position);
        status3 = cudaFree(d_Bond); // we don't need these later on
	
        if ( (status1 != CUDA_SUCCESS) || 
             (status2 != CUDA_SUCCESS) ||
             (status3 != CUDA_SUCCESS) ){
          std::cout<<"Freeing bond and basis information failed! Error: "<<cudaGetErrorString( cudaPeekAtLastError() )<<std::endl;
          return 1;
        }

        hamstruct* d_H_sort;
        status2 = cudaMalloc(&d_H_sort, num_Elem*sizeof(hamstruct));

	if (status2 != CUDA_SUCCESS){
                std::cout<<"Allocating d_H_sort failed! Error: ";
                std::cout<<cudaGetErrorString( status1 )<<std::endl;
                return 1;
        }
      
        cudaEventRecord(start, 0);	
	CompressSparse<<<*vdim, 32>>>(d_H_vals, d_H_pos, d_H_sort, vdim, lattice_Size, num_Elem);
        
        cudaThreadSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventElapsedTime(&elapsed, start, stop);

        fout<<"Runtime for CompressSparse: "<<elapsed<<std::endl;

        if (cudaPeekAtLastError() != 0){
              std::cout<<"Error in CompressSparse! Error: "<<cudaGetErrorString( cudaPeekAtLastError() )<<std::endl;
              return 1;
        }

	cudaFree(d_H_vals); //cleanup
	cudaFree(d_H_pos);

        //----------------Sorting Hamiltonian--------------------------//

        thrust::device_ptr<hamstruct> sort_ptr(d_H_sort);

        cudaEventRecord(start,0);
        thrust::sort(sort_ptr, sort_ptr + num_Elem, ham_sort_function());
        
        //--------------------------------------------------------------

        cudaThreadSynchronize();
        cudaEventRecord(stop,0);
        cudaEventElapsedTime(&elapsed, start, stop);

        fout<<"Runtime for sorting: "<<elapsed<<std::endl;

        if (cudaPeekAtLastError() != 0){
                std::cout<<"Error in sorting! Error: "<<cudaGetErrorString( cudaPeekAtLastError() )<<std::endl;
                return 1;
        }

        //std::cout<<"Sorting complete"<<std::endl;

        
	status1 = cudaMalloc(&hamil_Values, num_Elem*sizeof(cuDoubleComplex));
	status2 = cudaMalloc(&hamil_PosRow, num_Elem*sizeof(int));
	status3 = cudaMalloc(&hamil_PosCol, num_Elem*sizeof(int));

	if ( (status1 != CUDA_SUCCESS) ||
	     (status2 != CUDA_SUCCESS) ||
	     (status3 != CUDA_SUCCESS) ){
		std::cout<<"Memory allocation for COO representation failed! Error: "<<cudaGetErrorString( cudaPeekAtLastError() )<<std::endl;
		return 1;
	}
        
				
	//std::cout<<"Running FullToCOO."<<std::endl;

        cudaEventRecord(start, 0);	
        FullToCOO<<<num_Elem/512 + 1, 512>>>(num_Elem, d_H_sort, hamil_Values, hamil_PosRow, hamil_PosCol, *vdim); // csr and description initializations happen somewhere else

        cudaThreadSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventElapsedTime(&elapsed, start, stop);

        fout<<"Runtime for FullToCOO: "<<elapsed<<std::endl;
	
        cudaFree(d_H_sort);

        //free(h_H_pos);
        //free(h_H_vals);
        //fdata.close();
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        fout.close();

	return num_Elem;
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

__global__ void FillSparse(int* d_basis_Position, int* d_basis, int dim, cuDoubleComplex* H_vals, int2* H_pos, int* d_Bond, int lattice_Size, const double JJ){

	
	int ii = blockIdx.x;
        int T0 = threadIdx.x;

        __shared__ int3 tempbond[16];
        __shared__ int count;
	atomicExch(&count, 0);
        __shared__ int temppos[32];
        __shared__ cuDoubleComplex tempval[32]; //going to eliminate a huge number of read/writes to d_Bond, H_vals, H_pos in global memory

        int stridepos = 2*lattice_Size + 2;
        int strideval = 2*lattice_Size + 1;


	int si, sj;//sk,sl; //spin operators
	unsigned int tempi, tempod; //tempj;
	//cuDoubleComplex tempD;

        tempi = d_basis[ii];

	__syncthreads();

	if( ii < dim ){
            if (T0 < lattice_Size){
    		
		//Putting bond info in shared memory
		(tempbond[T0]).x = d_Bond[T0];
		(tempbond[T0]).y = d_Bond[lattice_Size + T0];
		(tempbond[T0]).z = d_Bond[2*lattice_Size + T0];
		__syncthreads();
		//Diagonal Part

		temppos[0] = d_basis_Position[tempi];

		tempval[0] = HDiagPart(tempi, lattice_Size, tempbond, JJ);

                H_vals[ idx(ii, 0, strideval) ] = tempval[0];
                (H_pos[ idx(ii, 1, stridepos) ]).y = temppos[0];
                (H_pos[ idx(ii, 1, stridepos) ]).x = ii;
                
                

		//-------------------------------
		//Horizontal bond ---------------
		si = (tempbond[T0]).x;
		tempod = tempi;
		sj = (tempbond[T0]).y;
	
		tempod ^= (1<<si);   //toggle bit 
		tempod ^= (1<<sj);   //toggle bit 

		if (d_basis_Position[tempod] > ii){ //build only upper half of matrix
        		temppos[2*T0] = d_basis_Position[tempod];
        		tempval[2*T0] = HOffBondX(T0,tempi, JJ);
                        atomicAdd(&count,1);
                                                                        
      		}

		else {
			temppos[2*T0] = -1;
			tempval[2*T0] = make_cuDoubleComplex(0., 0.);
		}

		//Vertical bond -----------------
		tempod = tempi;
      		sj = (tempbond[T0]).z;

      		tempod ^= (1<<si);   //toggle bit 
     		tempod ^= (1<<sj);   //toggle bit
                 
      		if (d_basis_Position[tempod] > ii){ 
        		temppos[2*T0 + 1] = d_basis_Position[tempod];
        		tempval[2*T0 + 1] = make_cuDoubleComplex(JJ*0.5, 0.);//HOffBondY(T0,tempi, JJ);
			atomicAdd(&count,1);
      		}

		else {
			temppos[2*T0 + 1] = -1;
			tempval[2*T0 + 1] = make_cuDoubleComplex(0., 0.);
		}

                //time to write back to global memory
                __syncthreads();

		
                H_vals[ idx(ii, 2*T0 + 1, strideval) ] = tempval[2*T0];
                (H_pos[ idx(ii, 2*T0 + 2, stridepos) ]).y = temppos[2*T0];
                (H_pos[ idx(ii, 2*T0 + 2, stridepos) ]).x = ii;                

		(H_pos[ idx(ii, 0, stridepos) ]).x = ii;
                (H_pos[ idx(ii, 0, stridepos) ]).y = count + 1;
                               
                (H_pos[ idx(ii, 2*T0 + 3, stridepos) ]).x = ii;
                (H_pos[ idx(ii, 2*T0 + 3, stridepos) ]).y = temppos[2*T0 + 1];
                
                (H_vals[ idx(ii, 2*T0 + 2, strideval) ]).x = (tempval[2*T0 + 1]).x;
                (H_vals[ idx(ii, 2*T0 + 2, strideval) ]).y = (tempval[2*T0 + 1]).y;
                                                 
            }//end of T0 

	}//end of ii
}//end of FillSparse

/* Function: CompressSparse - this function takes the sparse matrix with lots of "buffer" memory sitting on the end of each array, and compresses it down to get rid of the extra memory
Parameters:	H_vals - an array of Hamiltonian values
	        H_pos - an array of value positions 
                H_sort - an array of hamstructs that is filled with the upper and lower halves of the Hamiltonian
	        d_dim - the dimension of the Hamiltonian
	        lattice_Size - the number of lattice sites

*/
__global__ void CompressSparse(const cuDoubleComplex* H_vals, const int2* H_pos, hamstruct* H_sort, int d_dim, const int lattice_Size, const int num_Elem){

	int row = blockIdx.x;
        int col = threadIdx.x;

        __shared__ int s_H_pos[32];
        __shared__ cuDoubleComplex s_H_vals[32];
        //__shared__ hamstruct s_H_sort[32];

        __shared__ int iter;
        iter = 1;

	const int size1 = 2*lattice_Size + 2;
       	const int size2 = 2*lattice_Size + 1;

        __syncthreads();

        if (row < d_dim){
        
                // the basic idea here is to have each x thread go to the ith row, and each y thread go to the jth element of that row. then using a set of __shared__ temp arrays, we read in the Hamiltonian values and do our comparisons n stuff on them

		int start = 0;
	
                for (int ii = 0; ii < row; ii++){ 
                        start += 2*(H_pos[ idx(ii, 0 , size1) ]).y - 1 ;
        	}

                //if (start > num_Elem) printf("%d %d \n", start, num_Elem);		
		(H_sort[ start ]).rowindex = row;
                (H_sort[ start ]).colindex = row;
                (H_sort[ start ]).value = H_vals[ idx( row, 0, size2) ];
                (H_sort[ start ]).dim = d_dim; //doing the diagonals


		int temp;

                if (col < size2  - 1){
                        s_H_pos[col] = (H_pos[ idx( row, col + 2, size1 ) ]).y;
			s_H_vals[col] = H_vals[ idx( row, col + 1, size2) ];

                	if (s_H_pos[col] != -1){
				temp = atomicAdd(&iter, 1);
				(H_sort[start + temp]).rowindex = row;
				(H_sort[start + temp]).colindex = s_H_pos[col];
				(H_sort[start + temp]).value = s_H_vals[col];
				(H_sort[start + temp]).dim = d_dim;

				temp = atomicAdd(&iter, 1);

				(H_sort[ start + temp ]).rowindex = s_H_pos[col]; //the conjugate
                                (H_sort[ start + temp ]).colindex = row;
                                (H_sort[ start + temp ]).value = make_cuDoubleComplex( (s_H_vals[col]).x , -( s_H_vals[col]).y );
                                (H_sort[ start + temp ]).dim = d_dim;                 
			
			} 
		}

	}
}

/*Function: FullToCOO - takes a full sparse matrix and transforms it into COO format
Inputs - num_Elem - the total number of nonzero elements
	 H_vals - the Hamiltonian values
	 H_pos - the Hamiltonian positions
	 hamil_Values - a 1D array that will store the values for the COO form

*/
__global__ void FullToCOO(int num_Elem, hamstruct* H_sort, cuDoubleComplex* hamil_Values, int* hamil_PosRow, int* hamil_PosCol, int dim){

	int i = threadIdx.x + blockDim.x*blockIdx.x;

	int start = 0;

	if (i < num_Elem){
			
		hamil_Values[i] = H_sort[i].value;
		hamil_PosRow[i] = H_sort[i].rowindex;
		hamil_PosCol[i] = H_sort[i].colindex;
		
	}
}

__global__ void Copy(int* thrust_ptr, int2* H_pos, int lattice_Size, int vdim){

  int row = blockIdx.x*blockDim.x + threadIdx.x;

  if (row < vdim) thrust_ptr[row] = (H_pos[ idx(row, 0, 2*lattice_Size + 2) ]).y;
}
