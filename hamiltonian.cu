#include"hamiltonian.h"

/* NOTE: this function uses FORTRAN style matrices, where the values and positions are stored in a ONE dimensional array! Don't forget this! */


int main(){


int* Bond;
  Bond = (int*)malloc(16*3*sizeof(int));

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

  cuDoubleComplex* hamil_Values;

  int* hamil_PosRow;

  int* hamil_PosCol;

  int nsite = 16;

  int dim;

  int Sz = 0;
  float JJ = 1.f;



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
Outputs: basis_Position - a full array now
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
		} //unpack bra
		//if (temp==(lattice_Size/2+Sz) ){
			basis[realdim] = i1;
			basis_Position[i1] = realdim;
			realdim++;
			//cout<<basis[realdim]<<" "<<basis_Position[i1]<<endl;
		//}
}

return realdim;

}

/* Function HOffBondX
Inputs: si - the spin operator in the x direction
bra - the state
JJ - the coupling constant
Outputs: valH - the value of the Hamiltonian

*/

__device__ float HOffBondX(const int si, const int bra, const float JJ){

	float valH;
	//int S0, S1;
	//int T0, T1;

	valH = JJ*0.5; //contribution from the J part of the Hamiltonian

	return valH;

}

__device__ float HOffBondY(const int si, const int bra, const float JJ){

	float valH;
	//int S0, S1;
	//int T0, T1;

	valH = JJ*0.5; //contribution from the J part of the Hamiltonian

	return valH;


}

__device__ float HDiagPart(const int bra, int lattice_Size, int3* d_Bond, const float JJ){

	int S0b,S1b ; //spins (bra
	int T0,T1; //site
	//int P0, P1, P2, P3; //sites for plaquette (Q)
	//int s0p, s1p, s2p, s3p;
	float valH = 0.f;

	for (int Ti=0; Ti<lattice_Size; Ti++){
    //***HEISENBERG PART

		T0 = (d_Bond[Ti]).x; //lower left spin
		S0b = (bra>>T0)&1;
		//if (T0 != Ti) cout<<"Square error 3\n";
		T1 = (d_Bond[Ti]).y; //first bond
		S1b = (bra>>T1)&1; //unpack bra
		valH += JJ*(S0b-0.5)*(S1b-0.5);
		T1 = (d_Bond[Ti]).z; //second bond
		S1b = (bra>>T1)&1; //unpack bra
		valH += JJ*(S0b-0.5)*(S1b-0.5);

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

Outputs: hamil_Values - a pointer to a device array containing the values
hamil_PosRow - a pointer to a device array containing the locations of each value in a row
hamil_PosCol - a pointer to a device array containing the locations of each values in a column

*/


__host__ int ConstructSparseMatrix(int model_Type, int lattice_Size, int* Bond, cuDoubleComplex* hamil_Values, int* hamil_PosRow, int* hamil_PosCol, int* vdim, float JJ, int Sz){


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

	int padded_dim = (*vdim/1024 + 1)*1024;
	int raw_size = (padded_dim + 4*lattice_Size*(*vdim));

	dim3 bpg;

	bpg.x = (4*lattice_Size*(*vdim))/512 + 1;
        
	dim3 tpb;
	tpb.x = 1024;
	//these are going to need to depend on dim and Nsize
     
	int* d_H_rows;
	int* d_H_cols;
	float* d_H_vals;
	
	cudaMalloc(&d_H_rows, raw_size*sizeof(int));
	cudaMalloc(&d_H_cols, raw_size*sizeof(int));
	cudaMalloc(&d_H_vals, raw_size*sizeof(float));

	/*hamstruct* d_H_sort;
	status2 = cudaMalloc(&d_H_sort, *vdim*stride*sizeof(hamstruct));

	if (status2 != CUDA_SUCCESS){
		std::cout<<"Allocating d_H_sort failed! Error: ";
		std::cout<<cudaGetErrorString( status1 )<<std::endl;
		return 1;
	}*/
	
	FillDiagonals<<<*vdim/1024 + 1, tpb>>>(d_basis, *vdim, d_H_rows, d_H_cols, d_H_vals, d_Bond, lattice_Size, JJ);

	cudaThreadSynchronize();

	if( cudaPeekAtLastError() != 0 ){
		std::cout<<"Error in FillDiagonals! Error: "<<cudaGetErrorString( cudaPeekAtLastError() )<<std::endl;
		return 1;
	}

	FillSparse<<<bpg, tpb>>>(d_basis_Position, d_basis, *vdim, d_H_rows, d_H_cols, d_H_vals, d_Bond, lattice_Size, JJ);

	cudaThreadSynchronize();

	if( cudaPeekAtLastError() != 0 ){
		std::cout<<"Error in FillSparse! Error: "<<cudaGetErrorString( cudaPeekAtLastError() )<<std::endl;
		return 1;
	}

	int* num_ptr;
	cudaGetSymbolAddress((void**)&num_ptr, (const char*)"d_num_Elem");

	cudaMemcpy(&num_Elem, num_ptr, sizeof(int), cudaMemcpyDeviceToHost);
	std::cout<<num_Elem<<std::endl;
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

	sortEngine_t engine;
	sortStatus_t sortstatus = sortCreateEngine("sort/sort/src/cubin64/", &engine);

	MgpuSortData sortdata;

	
	sortdata.AttachKey((uint*)d_H_rows);
	sortdata.AttachVal(0, (uint*)d_H_cols);
	sortdata.AttachVal(1, (uint*)d_H_vals);

	int sortnumber = ((raw_size/2048) + 1)*2048;

	sortdata.Alloc(engine, sortnumber, 2);

	sortdata.firstBit = 0;
	sortdata.endBit = 8*sizeof(dim);

	sortArray(engine, &sortdata);

	/*thrust::device_ptr<int> sort_key_ptr(d_H_rows);
	thrust::device_ptr<int> sort_val_ptr(d_H_cols);

	thrust::sort_by_key(sort_key_ptr, sort_key_ptr + *vdim*stride, sort_val_ptr);*/
        
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

	cudaMemcpy(hamil_PosRow, (int*)sortdata.keys[0], num_Elem*sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(hamil_PosCol, (int*)sortdata.values1[0], num_Elem*sizeof(int), cudaMemcpyDeviceToDevice);

	FullToCOO<<<num_Elem/1024 + 1, 1024>>>(num_Elem, (float*)sortdata.values2[0], hamil_Values, *vdim); // csr and description initializations happen somewhere else

	cudaFree(d_H_rows);
	cudaFree(d_H_cols);
	cudaFree(d_H_vals);

	cuDoubleComplex* h_vals = (cuDoubleComplex*)malloc(num_Elem*sizeof(cuDoubleComplex));
	int* h_rows = (int*)malloc(num_Elem*sizeof(int));
	int* h_cols = (int*)malloc(num_Elem*sizeof(int));

	/*cudaMemcpy(h_vals, hamil_Values, num_Elem*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_rows, hamil_PosRow, num_Elem*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_cols, hamil_PosCol, num_Elem*sizeof(int), cudaMemcpyDeviceToHost);

	std::ofstream fout;
	fout.open("hamiltonian.log");
	for(int i = 0; i < num_Elem; i++){
		fout<<"("<<h_rows[i]<<","<<h_cols[i]<<")";
		fout<<" - "<<h_vals[i].x<<std::endl;
	}

	fout.close();*/

	sortReleaseEngine(engine);

	return num_Elem;
}

__global__ void FillDiagonals(int* d_basis, int dim, int* H_rows, int* H_cols, float* H_vals, int* d_Bond, int lattice_Size, float JJ){

	int row = blockIdx.x*blockDim.x + threadIdx.x;
	int site = threadIdx.x%(lattice_Size);

	unsigned int tempi = d_basis[row];

	__shared__ int3 tempbond[16];

	if (row < dim){
		(tempbond[site]).x = d_Bond[site];
		(tempbond[site]).y = d_Bond[lattice_Size + site];
		(tempbond[site]).z = d_Bond[2*lattice_Size + site];

		H_vals[row] = HDiagPart(tempi, lattice_Size, tempbond, JJ);
		H_rows[row] = row;
		H_cols[row] = row;

	}

	else {
		H_rows[row] = dim;
	}

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

__global__ void FillSparse(int* d_basis_Position, int* d_basis, int dim, int* H_rows, int* H_cols, float* H_vals, int* d_Bond, const int lattice_Size, const float JJ){

	int ii = (blockDim.x/(2*lattice_Size))*blockIdx.x + threadIdx.x/(2*lattice_Size);
	int T0 = threadIdx.x%(2*lattice_Size);

	#if __CUDA_ARCH__ < 200
		const int array_size = 512;
	#elif __CUDA_ARCH__ >= 200
		const int array_size = 1024;
	#else
       		#error your mom
	#endif

	__shared__ int3 tempbond[16];
	int count;
	__shared__ int temppos[array_size];
	__shared__ float tempval[array_size];
	__shared__ uint tempi[array_size];
	__shared__ uint tempod[array_size];

	int stride = 4*lattice_Size;
	int tempcount;
	int site = T0%(lattice_Size);
	count = 0;
	int rowtemp;

	int start = (dim/array_size + 1)*array_size;

	int si, sj;//sk,sl; //spin operators
	//unsigned int tempi;// tempod; //tempj;
	//cuDoubleComplex tempD;

	tempi[threadIdx.x] = d_basis[ii];

	__syncthreads();

	bool compare;

	if( ii < dim ){
		if (T0 < 2*lattice_Size){
			//Putting bond info in shared memory
			(tempbond[site]).x = d_Bond[site];
			(tempbond[site]).y = d_Bond[lattice_Size + site];
			(tempbond[site]).z = d_Bond[2*lattice_Size + site];

			__syncthreads();
			//Diagonal Part

			/*temppos[threadIdx.x] = d_basis_Position[tempi[threadIdx.x]];
			tempval[threadIdx.x] = HDiagPart(tempi[threadIdx.x], lattice_Size, tempbond, JJ);

			H_sort[ idx(ii, 0, stride) ].value = tempval[threadIdx.x];
			H_sort[ idx(ii, 0, stride) ].colindex = temppos[threadIdx.x];
			H_sort[ idx(ii, 0, stride) ].rowindex = ii;
			H_sort[ idx(ii, 0, stride) ].dim = dim;*/
                
			//-------------------------------
			//Horizontal bond ---------------
			si = (tempbond[site]).x;
			tempod[threadIdx.x] = tempi[threadIdx.x];
			sj = (tempbond[site]).y;

			tempod[threadIdx.x] ^= (1<<si); //toggle bit
			tempod[threadIdx.x] ^= (1<<sj); //toggle bit

			compare = (d_basis_Position[tempod[threadIdx.x]] > ii);
			temppos[threadIdx.x] = (compare) ? d_basis_Position[tempod[threadIdx.x]] : dim;
			tempval[threadIdx.x] = HOffBondX(site, tempi[threadIdx.x], JJ);

			count += (int)compare;
			tempcount = (T0/lattice_Size);
			rowtemp = (T0/lattice_Size) ? ii : temppos[threadIdx.x];			
			rowtemp = (compare) ? rowtemp : dim;

			H_vals[ idx(ii, 4*site + tempcount + start, stride) ] = tempval[threadIdx.x]; //(T0/lattice_Size) ? tempval[threadIdx.x] : cuConj(tempval[threadIdx.x]);
			H_cols[ idx(ii, 4*site + tempcount + start, stride) ] = (T0/lattice_Size) ? temppos[threadIdx.x] : ii;
			H_rows[ idx(ii, 4*site + tempcount + start, stride) ] = rowtemp;

//Vertical bond -----------------
			tempod[threadIdx.x] = tempi[threadIdx.x];
			sj = (tempbond[site]).z;

			tempod[threadIdx.x] ^= (1<<si); //toggle bit
			tempod[threadIdx.x] ^= (1<<sj); //toggle bit
                 
			compare = (d_basis_Position[tempod[threadIdx.x]] > ii);
			temppos[threadIdx.x] =  (compare) ? d_basis_Position[tempod[threadIdx.x]] : dim;
			tempval[threadIdx.x] = HOffBondY(site,tempi[threadIdx.x], JJ);

			count += (int)compare;
			tempcount = (T0/lattice_Size);
			rowtemp = (T0/lattice_Size) ? ii : temppos[threadIdx.x];			
			rowtemp = (compare) ? rowtemp : dim;

			H_vals[ idx(ii, 4*site + 2 + tempcount + start, stride) ] =  tempval[threadIdx.x]; // (T0/lattice_Size) ? tempval[threadIdx.x] : cuConj(tempval[threadIdx.x]);
			H_cols[ idx(ii, 4*site + 2 + tempcount + start, stride) ] = (T0/lattice_Size) ? temppos[threadIdx.x] : ii;
			H_rows[ idx(ii, 4*site + 2 + tempcount + start, stride) ] = rowtemp;
			
			__syncthreads();

			atomicAdd(&d_num_Elem, count);
}
}//end of ii
}//end of FillSparse

/*Function: FullToCOO - takes a full sparse matrix and transforms it into COO format
Inputs - num_Elem - the total number of nonzero elements
H_vals - the Hamiltonian values
H_pos - the Hamiltonian positions
hamil_Values - a 1D array that will store the values for the COO form

*/
__global__ void FullToCOO(int num_Elem, float* H_vals, cuDoubleComplex* hamil_Values, int dim){

	int i = threadIdx.x + blockDim.x*blockIdx.x;

	if (i < num_Elem){

		hamil_Values[i].x = H_vals[i];
		

	}
}
;


