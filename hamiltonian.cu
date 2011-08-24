#include"hamiltonian.h"

/* NOTE: this function uses FORTRAN style matrices, where the values and positions are stored in a ONE dimensional array! Don't forget this! */


int main(){


int* Bond;
  Bond = (int*)malloc(16*3*sizeof(int));
	
  Bond[0] = 0;    Bond[1] = 1;    Bond[2] = 2;    Bond[3] = 3;    Bond[4] = 4;
  Bond[5] = 5;    Bond[6] = 6;    Bond[7] = 7;    Bond[8] = 8;    Bond[9] = 9;
  Bond[10] = 10;  Bond[11] = 11;  Bond[12] = 12;  Bond[13] = 13;  Bond[14] = 14;
  Bond[15] = 15;  Bond[16] = 1;   Bond[17] = 2;   Bond[18] = 3;   Bond[19] = 0;
  Bond[20] = 5;   Bond[21] = 6;   Bond[22] = 7;   Bond[23] = 4;   Bond[24] = 9;
  Bond[25] = 10;  Bond[26] = 11;  Bond[27] = 8;   Bond[28] = 13;  Bond[29] = 14;
  Bond[30] = 15;  Bond[31] = 12;  Bond[32] = 4;   Bond[33] = 5;   Bond[34] = 6;
  Bond[35] = 7;   Bond[36] = 8;   Bond[37] = 9;   Bond[38] = 10;  Bond[39] = 11;
  Bond[40] = 12;  Bond[41] = 13;  Bond[42] = 14;  Bond[43] = 15;  Bond[44] = 0;
  Bond[45] = 1;   Bond[46] = 2;   Bond[47] = 3;

  cuDoubleComplex* hamil_Values;

  int* hamil_PosRow;

  int* hamil_PosCol;

  int nsite = 16;

  int dim;

  int Sz = 0;
  double JJ = 1.;



  int num_Elem = ConstructSparseMatrix( 0, nsite, Bond, hamil_Values, hamil_PosRow, hamil_PosCol, &dim, JJ, Sz );

  return 0;
}

__host__ __device__ int idx(int i, int j, int lda){
  
  return (j + (i*lda));
}


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


	//cudaSetDevice(1);

	int num_Elem = 0; // the total number of elements in the matrix, will get this (or an estimate) from the input types
	cudaError_t status1, status2, status3;

	//int dim = 65536;
	
	/*
	switch (model_Type){
		case 0: 
			dim = 65536;
			break;
		case 1: dim = 10; //guesses
	}
        */
	int dim = 2;

	for (int ch=1; ch<lattice_Size; ch++) dim *= 2;

	int stride = 4*lattice_Size + 1;        

	int basis_Position[dim];
	int basis[dim];
	//----------------Construct basis and copy it to the GPU --------------------//

	*vdim = GetBasis(dim, lattice_Size, Sz, basis_Position, basis);

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

	bpg.x = *vdim/1024 + 1;
        
	dim3 tpb;
	tpb.x = 1024;
	//these are going to need to depend on dim and Nsize
      
	thrust::device_vector<int> num_array(*vdim, 1);
	int* num_array_ptr = raw_pointer_cast(&num_array[0]);

	hamstruct* d_H_sort;
	status2 = cudaMalloc(&d_H_sort, *vdim*stride*sizeof(hamstruct));

	if (status2 != CUDA_SUCCESS){
                std::cout<<"Allocating d_H_sort failed! Error: ";
                std::cout<<cudaGetErrorString( status1 )<<std::endl;
                return 1;
	}

	FillSparse<<<bpg, tpb>>>(d_basis_Position, d_basis, *vdim, d_H_sort, num_array_ptr, d_Bond, lattice_Size, JJ);

	if( cudaPeekAtLastError() != 0 ){
		std::cout<<"Error in FillSparse! Error: "<<cudaGetErrorString( cudaPeekAtLastError() )<<std::endl;
		return 1;
	}
		
	cudaThreadSynchronize();

	int* num_ptr;
	cudaGetSymbolAddress((void**)&num_ptr, (const char*)"d_num_Elem");
	cudaMemset(num_ptr, 0, sizeof(int));

	thrust::device_ptr<int> thrust_num_ptr(num_ptr);
	*thrust_num_ptr = thrust::reduce(num_array.begin(), num_array.end());

	cudaMemcpy(&num_Elem, num_ptr, sizeof(int), cudaMemcpyDeviceToHost);

	status1 = cudaFree(d_basis);
	status2 = cudaFree(d_basis_Position);
	status3 = cudaFree(d_Bond); // we don't need these later on
	
	if ( (status1 != CUDA_SUCCESS) || 
			 (status2 != CUDA_SUCCESS) ||
			 (status3 != CUDA_SUCCESS) ){
		std::cout<<"Freeing bond and basis information failed! Error: "<<cudaGetErrorString( cudaPeekAtLastError() )<<std::endl;
		return 1;
	}

	//----------------Sorting Hamiltonian--------------------------//
	
	thrust::device_ptr<hamstruct> sort_ptr(d_H_sort);

	thrust::sort(sort_ptr, sort_ptr + *vdim*stride, ham_sort_function());
        
	//--------------------------------------------------------------

	if (cudaPeekAtLastError() != 0){
		std::cout<<"Error in sorting! Error: "<<cudaGetErrorString( cudaPeekAtLastError() )<<std::endl;
		return 1;
	}

	status1 = cudaMalloc(&hamil_Values, num_Elem*sizeof(cuDoubleComplex));
	status2 = cudaMalloc(&hamil_PosRow, num_Elem*sizeof(int));
	status3 = cudaMalloc(&hamil_PosCol, num_Elem*sizeof(int));

	if ( (status1 != CUDA_SUCCESS) ||
	     (status2 != CUDA_SUCCESS) ||
	     (status3 != CUDA_SUCCESS) ){
		std::cout<<"Memory allocation for COO representation failed! Error: "<<cudaGetErrorString( cudaPeekAtLastError() )<<std::endl;
		return 1;
	}
	
	FullToCOO<<<num_Elem/512 + 1, 512>>>(num_Elem, d_H_sort, hamil_Values, hamil_PosRow, hamil_PosCol, *vdim); // csr and description initializations happen somewhere else
	
	cudaFree(d_H_sort);	

	return num_Elem;
}


/* Function FillSparse: this function takes the empty Hamiltonian arrays and fills them up. Each thread in x handles one ket |i>, and each thread in y handles one site T0
Inputs: d_basis_Position - position information about the basis
	d_basis - other basis infos
	d_dim - the number of kets
	H_sort - an array that will store the Hamiltonian
	d_Bond - the bond information
	d_lattice_Size - the number of lattice sites
	JJ - the coupling parameter 

*/

__global__ void FillSparse(int* d_basis_Position, int* d_basis, int dim, hamstruct* H_sort, int* elem_num_array, int* d_Bond, const int lattice_Size, const double JJ){

	int ii = (blockDim.x/32)*blockIdx.x + threadIdx.x/32;
	int jj = threadIdx.x;
	int T0 = threadIdx.x%32;

	__shared__ int3 tempbond[16];
	__shared__ int count[1024];
	__shared__ int temppos[1024];
	__shared__ cuDoubleComplex tempval[1024];
	//__shared__ uint tempi[32];
	__shared__ uint tempod[1024];

	int stride = 4*lattice_Size + 1;
	int tempcount;
	int site = T0%(lattice_Size);
	count[T0] = 0;

	int si, sj;//sk,sl; //spin operators
	unsigned int tempi; //tempod; //tempj;
	//cuDoubleComplex tempD;

	tempi = d_basis[ii];

	__syncthreads();

	bool compare;

	if( ii < dim ){
    	if (site < lattice_Size){	
			//Putting bond info in shared memory
				(tempbond[site]).x = d_Bond[site];
				(tempbond[site]).y = d_Bond[lattice_Size + site];
				(tempbond[site]).z = d_Bond[2*lattice_Size + site];
			
				__syncthreads();
				//Diagonal Part

				temppos[jj] = d_basis_Position[tempi];
	
				tempval[jj] = HDiagPart(tempi, lattice_Size, tempbond, JJ);

				H_sort[ idx(ii, 0, stride) ].value = tempval[jj];
				H_sort[ idx(ii, 0, stride) ].colindex = temppos[jj];
				H_sort[ idx(ii, 0, stride) ].rowindex = ii;
			H_sort[ idx(ii, 0, stride) ].dim = dim;
                
			//-------------------------------
			//Horizontal bond ---------------
			si = (tempbond[site]).x;
			tempod[jj] = tempi;
			sj = (tempbond[site]).y;
		
			tempod[jj] ^= (1<<si);   //toggle bit 
			tempod[jj] ^= (1<<sj);   //toggle bit 
	
			compare = (d_basis_Position[tempod[jj]] > ii);
			temppos[jj] = (compare == 1) ? d_basis_Position[tempod[jj]] : -1;
			tempval[jj] = HOffBondX(site, tempi, JJ);
			count[T0] += compare;
			tempcount = 1 + (T0/lattice_Size);

			H_sort[ idx(ii, 4*site + tempcount, stride) ].value = (T0/lattice_Size) ? tempval[T0] : cuConj(tempval[T0]);
			H_sort[ idx(ii, 4*site + tempcount, stride) ].colindex = (T0/lattice_Size) ? temppos[T0] : ii;
			H_sort[ idx(ii, 4*site + tempcount, stride) ].rowindex = (T0/lattice_Size) ? ii : temppos[T0];
			H_sort[ idx(ii, 4*site + tempcount, stride) ].dim = dim;


			//Vertical bond -----------------
			tempod[jj] = tempi;
			sj = (tempbond[site]).z;

			tempod[jj] ^= (1<<si);   //toggle bit 
			tempod[jj] ^= (1<<sj);   //toggle bit
                 
			compare = (d_basis_Position[tempod[jj]] > ii);
			temppos[jj] = (compare == 1) ? d_basis_Position[tempod[jj]] : -1;
			tempval[jj] = HOffBondY(site,tempi, JJ);
			
			count[T0] += compare;
			//printf("%d %d \n", count, compare);

			tempcount = 1 + (T0/lattice_Size);

			H_sort[ idx(ii, 4*site + 2 + tempcount, stride) ].value = (T0/lattice_Size) ? tempval[T0] : cuConj(tempval[T0]);
			H_sort[ idx(ii, 4*site + 2 + tempcount, stride) ].colindex = (T0/lattice_Size) ? temppos[T0] : ii;
			H_sort[ idx(ii, 4*site + 2 + tempcount, stride) ].rowindex = (T0/lattice_Size) ? ii : temppos[T0];
			H_sort[ idx(ii, 4*site + 2 + tempcount, stride) ].dim = dim;   
			__syncthreads();

			atomicAdd(&elem_num_array[ii], count[T0]);
		}
	}//end of ii
}//end of FillSparse

/*Function: FullToCOO - takes a full sparse matrix and transforms it into COO format
Inputs - num_Elem - the total number of nonzero elements
	 H_vals - the Hamiltonian values
	 H_pos - the Hamiltonian positions
	 hamil_Values - a 1D array that will store the values for the COO form

*/
__global__ void FullToCOO(int num_Elem, hamstruct* H_sort, cuDoubleComplex* hamil_Values, int* hamil_PosRow, int* hamil_PosCol, int dim){

	int i = threadIdx.x + blockDim.x*blockIdx.x;

	if (i < num_Elem){
			
		hamil_Values[i] = H_sort[i].value;
		hamil_PosRow[i] = H_sort[i].rowindex;
		hamil_PosCol[i] = H_sort[i].colindex;
		
	}
}
;
