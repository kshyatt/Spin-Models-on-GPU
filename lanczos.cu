// Katharine Hyatt
// A set of functions to implement the Lanczos method for a generic Hamiltonian
// Based on the codes Lanczos_07.cpp and Lanczos07.h by Roger Melko
//-------------------------------------------------------------------------------

#include"lanczos.h"

// h_ means this variable is going to be on the host (CPU)
// d_ means this variable is going to be on the device (GPU)
// s_ means this variable is shared between threads on the GPU
// The notation <<x,y>> before a function defined as global tells the GPU how many threads per block to use
// and how many blocks per grid to use
// blah.x means the real part of blah, if blah is a data type from cuComplex.h
// blah.y means the imaginary party of blah, if blah is a data type from cuComplex.h
// threadIdx.x (or block) means the "x"th thread from the left in the block (or grid)
// threadIdx.y (or block) means the "y"th thread from the top in the block (or grid)

//Function vecdiff: calculates the difference between some vectors, two of which is multiplied by a scalar
//Implements w = x - a*y - b*z 
//-------------------------------------------------------------------------------------------------------
//Input: w, a "dummy pointer" to the vector that is changed
//       x, the vector that a*y and b*z are subtracted from 
//       alpha, the scalar the first subtracted vector is multiplied by
//       y, the first subtracted vector
//       beta, the scalar the second subtraced vector is multiplied by
//       z, the second subtracted vector
//       n, the number of elements in all the vectors
//       Note: no control is put in here to make sure all vectors are the same size. 
//------------------------------------------------------------------------------------------------------
//Output: w, the result of the subtractions
//        All other quantities remain unchanged
//------------------------------------------------------------------------------------------------------
__global__ void vecdiff(cuDoubleComplex* w, cuDoubleComplex* x, cuDoubleComplex alpha, cuDoubleComplex* y, cuDoubleComplex beta, cuDoubleComplex* z, int n){

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < n) {
    w[i] = cuCsub(x[i],cuCsub(cuCmul(alpha,y[i]),cuCmul(beta,z[i]))); //this is the dirtiest thing ever
  }
  __syncthreads();
}

//Function assignr: assigns the real parts of an array of double complex numbers the value of some double
//-------------------------------------------------------------------------------------------------------
//Input: a, a vector of double precision complex numbers whose real parts we would like to change
//       b, the real number that will become the real part of the complex numbers in a
//       n, the number of elements in a
//-------------------------------------------------------------------------------------------------------
//Output: a, the vector of complex numbers whose real parts have been changed
//        All other quantities are unchanged
//------------------------------------------------------------------------------------------------------- 
__global__ void assignr(cuDoubleComplex* a, double b, int n){
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < n){
    a[i] = make_cuDoubleComplex(b,0.);
  }
}

__global__ void assign(double* a, double b, int n){
  int i = blockDim.x*blockIdx.x + threadIdx.x;

  if (i < n){
    a[i] = b;
  }
}


//Function complextodoubler: assigns the real parts of complex numbers in an array to doubles in another array
//------------------------------------------------------------------------------------------------------------
//Input: a, the vector of complex numbers whose real parts we are extracting
//       b, the vector of doubles that will hold the real parts
//       n, the number of elements in the vectors
//------------------------------------------------------------------------------------------------------------
//Output: b, the vector of doubles now holding the real parts
//        All other quantities are unchanged
//------------------------------------------------------------------------------------------------------------
__global__ void complextodoubler(cuDoubleComplex* a, double* b, int n){
  int i = blockDim.x*blockIdx.x + threadIdx.x;

  if(i <= n){
    b[i] = a[i].x; 
  }
}

//Same as above, but in this case the parts are shifted by one space in the vector
__global__ void complextodoubler2(cuDoubleComplex* a, double* b, int n){
  int i = blockDim.x*blockIdx.x + threadIdx.x + 1;

  if(i <= n){
    b[i-1] = a[i].x;
  }
  if(i == n+1){
    b[i-1] = 0.;
  }
} 

__global__ void zero(double** a, int m){
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  int j = blockDim.y*blockIdx.y + threadIdx.y;

  if ( i< m && j < m){
    a[i][j] = 0. ;
  }
}
// Note: to get the identity matrix, apply the fuction zero above first
__global__ void eye(double** a, int m){
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  
  if (i < m ){

    a[i][i] = 1.;
  }
}


//Function copyHamiltonian: copies all the CSR data for the Hamiltonian to the device - made this to clean stuff up
__host__ void copyHamiltonian(const int h_num_nonzeroelem, const cuDoubleComplex* h_values, const int* h_rowstart, const int* h_colindex, const cusparseMatDescr_t* h_descrH, int dim, int* d_num_nonzeroelem, cuDoubleComplex* d_values, int* d_rowstart, int* d_colindex, cusparseMatDescr_t* d_descrH, int* d_dim){
  

  cudaError_t error;

  error = cudaMalloc(&d_num_nonzeroelem, sizeof(int));

  if (error != CUDA_SUCCESS){
    printf("Num of elements allocation on device failed! \n");
  }

  error = cudaMemcpy(d_num_nonzeroelem, &h_num_nonzeroelem, sizeof(int), cudaMemcpyHostToDevice);

  if (error != CUDA_SUCCESS){
    printf("Num of elements copy from host to device failed! \n");
  }

  error = cudaMalloc(&d_values, h_num_nonzeroelem*sizeof(cuDoubleComplex));

  if (error != CUDA_SUCCESS){
    printf("Values allocation on device failed! \n");
  }

  error = cudaMemcpy(d_values, h_values, h_num_nonzeroelem*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

  if (error != CUDA_SUCCESS){
    printf("Values copy from host to device failed! \n");
  }

  error = cudaMalloc(&d_rowstart, sizeof(h_rowstart)*sizeof(int));
  
  if (error != CUDA_SUCCESS){
    printf("Row start allocation on device failed! \n");
  }

  error = cudaMemcpy(d_rowstart, h_rowstart, sizeof(h_rowstart)*sizeof(int), cudaMemcpyHostToDevice);

  if (error != CUDA_SUCCESS){
    printf("Row start copy from host to device failed! \n");
  }

  error = cudaMalloc(&d_colindex, h_num_nonzeroelem*sizeof(int));

  if (error != CUDA_SUCCESS){
    printf("Column index allocation on device failed! \n");
  }

  error = cudaMemcpy(d_colindex, h_colindex, h_num_nonzeroelem*sizeof(int), cudaMemcpyHostToDevice);

  if (error != CUDA_SUCCESS){
    printf("Column index copy from host to device failed! \n");
  }

  error = cudaMalloc(&d_descrH, sizeof(cusparseMatDescr_t));
  
  if (error != CUDA_SUCCESS){
    printf("Matrix description allocation on device failed! \n");
  }

  error = cudaMemcpy(d_descrH, h_descrH, sizeof(cusparseMatDescr_t), cudaMemcpyHostToDevice);

  if (error != CUDA_SUCCESS){
    printf("Matrix description copy from host to device failed! \n");
  }

  error = cudaMalloc(&d_dim, sizeof(int));
  if (error != CUDA_SUCCESS){
    printf("Matrix dimension allocation on device failed! \n");
  }

  error = cudaMemcpy(d_dim, &dim, sizeof(int), cudaMemcpyHostToDevice);

  if (error != CUDA_SUCCESS){
    printf("Matrix dimension copy from host to device failed! \n");
  }

}
//Function lanczos: takes a hermitian matrix H, tridiagonalizes it, and finds the n smallest eigenvalues.
//---------------------------------------------------------------------------------------------------------------------------------------------------
// Input: num_nonzeroelem, the number of nonzero elements in the Hamiltonian
//        values, an array of the nonzero values
//        rowstart, an array of the index of the values that lead a row (see CSR matrix representation documentation for more on this)
//        colindex, and array of the column in which each element in values resides
//        descrH, the description of H that cusparse functions need
//        dim, the dimension of the full mxm matrix
//        max_Iter, the starting number of iterations we'll try
//        num_Eig, the number of eigenvalues we're interested in seeing
//        conv_req, the convergence we'd like to see
//---------------------------------------------------------------------------------------------------------------------------------------------------
// Output: h_ordered, the array of the num_Eig smallest eigenvalues, ordered from smallest to largest
//---------------------------------------------------------------------------------------------------------------------------------------------------        



int main(){}

void lanczos(const int h_num_nonzeroelem, const cuDoubleComplex* h_values, const int* h_rowstart, const int* h_colindex, const cusparseMatDescr_t *h_descrH, const int dim, int max_Iter, const int num_Eig, const double conv_req){

  cublasStatus linalgstat;
  linalgstat = cublasInit(); //have to initialize the cuBLAS environment, or my program won't work! I could use this later to check for errors as well

  cusparseHandle_t sparsehandle;
  cusparseStatus_t sparsestatus = cusparseCreate(&sparsehandle); //have to initialize the cusparse environment too! This variable gets passed to all my cusparse functions

  if (linalgstat != CUBLAS_STATUS_SUCCESS){
    printf("Failed to initialize CUBLAS! \n");
  }

  if (sparsestatus != CUSPARSE_STATUS_SUCCESS){
    printf("Failed to initialize CUSPARSE! \n");
  }

  cudaError_t status1, status2, status3, status4; //this is to throw errors in case things (mostly memory) in the code fail!  

  int* d_num_nonzeroelem;
  cuDoubleComplex* d_values;
  int* d_rowstart;
  int* d_colindex; 
  cusparseMatDescr_t* d_descrH;
  int* d_dim;

  copyHamiltonian(h_num_nonzeroelem, h_values, h_rowstart, h_colindex, h_descrH, dim, d_num_nonzeroelem, d_values, d_rowstart, d_colindex, d_descrH, d_dim);

  cuDoubleComplex* d_a; //these are going to store the elements of the tridiagonal matrix
  cuDoubleComplex* d_b; //they have to be cuDoubleComplex because that's the only input type the cublas functions I need will take

  status3 = cudaMalloc(&d_a, max_Iter*sizeof(cuDoubleComplex));
  status4 = cudaMalloc(&d_b, max_Iter*sizeof(cuDoubleComplex));

  if ((status3 != CUDA_SUCCESS) || (status4 != CUDA_SUCCESS)){
    printf("Matrix elements memory allocation failed! \n");
  }


  int tpb = 256; //threads per block - a conventional number
  int bpg = (dim + tpb - 1)/tpb; //blocks per grid

  //Making the "random" starting vector

  cuDoubleComplex** d_eigen_Array; //this thing is an array of pointers to the eigenvectors 
  status1 = cudaMalloc(&d_eigen_Array, max_Iter*sizeof(cuDoubleComplex*)); // making the pointer array

  if (status1 != CUDA_SUCCESS){
    printf("Eigenvector array allocation failed! \n");
  }
	//need to fix the below too
  CDCarraysalloc<<<1, max_Iter>>>(d_eigen_Array, dim, 0); //time to make the actual arrays of the eigenvectors

  assignr<<<bpg,tpb>>>(d_eigen_Array[0], 1., dim); //assigning the values of the "random" starting vector
  
  cuDoubleComplex* alpha; 
  status3 = cudaMalloc(&alpha, sizeof(cuDoubleComplex));
  *alpha = make_cuDoubleComplex(1.,0.);
  cuDoubleComplex* beta;
  status4 = cudaMalloc(&beta, sizeof(cuDoubleComplex));
  *beta = make_cuDoubleComplex(0.,0.);

  if ((status3 != CUDA_SUCCESS) || (status4 != CUDA_SUCCESS)){
    printf("Dummy constants alpha and beta initialization failed! \n");
  }

  cusparseOperation_t* A;
  cudaMalloc(&A, sizeof(cusparseOperation_t));

  *A = CUSPARSE_OPERATION_NON_TRANSPOSE;

  cusparseZcsrmv(sparsehandle, *A, *d_dim, *d_dim, *alpha, *d_descrH, d_values, d_rowstart, d_colindex, d_eigen_Array[0], *beta, d_eigen_Array[1]); // the Hamiltonian is applied here

  //*********************************************************************************************************
  // This is just the first steps so I can do the rest  
  d_a[0] = cublasZdotc(*d_dim, d_eigen_Array[0], sizeof(cuDoubleComplex), d_eigen_Array[1], sizeof(cuDoubleComplex));
  d_b[0] = make_cuDoubleComplex(0.,0.);

  cuDoubleComplex* y;
  status2 = cudaMalloc(&y, dim*sizeof(cuDoubleComplex));

  if (status2 != CUDA_SUCCESS){
    printf("Memory allocation of y dummy vector failed! \n");
  }
  
  assignr<<<bpg,tpb>>>(y, 0., *d_dim); //a dummy vector of 0s that i can stick in my functions

  vecdiff<<<bpg,tpb>>>(d_eigen_Array[1], d_eigen_Array[1], d_a[0], d_eigen_Array[0], y[0], y, *d_dim);
  d_b[1] = make_cuDoubleComplex(sqrt(cublasDznrm2(dim, d_eigen_Array[1], sizeof(cuDoubleComplex))),0.);
  // this function (above) takes the norm
  
  cuDoubleComplex gamma = make_cuDoubleComplex(1./d_b[1].x,0.); //alpha = 1/beta in v1 = v1 - alpha*v0

  cublasZaxpy(dim, gamma, d_eigen_Array[1], sizeof(cuDoubleComplex), y, sizeof(cuDoubleComplex)); // function performs a*x + y

  //Now we're done the first round!
  //*********************************************************************************************************

  double* d_ordered;
  status1 = cudaMalloc(&d_ordered, num_Eig*sizeof(double));

  if (status1 != CUDA_SUCCESS){
    printf("Eigenvalue array memory allocation failed! \n");
  }

  assign<<<bpg,tpb>>>(d_ordered, 0., num_Eig);

  double* gs_Energy;
  *gs_Energy = 1.; //the lowest energy

  int returned;

  int iter = 0;

  // In the original code, we started diagonalizing from iter = 5 and above. I start from iter = 1 to minimize issues of control flow
  double* d_diag;
  double* d_offdia;

  status3 = cudaMalloc(&d_diag, dim*sizeof(double));
  status4 = cudaMalloc(&d_offdia, dim*sizeof(double));

  thrust::device_ptr<double> dev_ptr(d_diag);

  if ((status3 != CUDA_SUCCESS) || (status4 != CUDA_SUCCESS)){
    printf("Second matrix elements array memory allocation failed! \n");
  }

  double* eigtemp;
  *eigtemp = 0.;

  while( abs(*gs_Energy - *eigtemp)> conv_req){ //this is a cleaner version than what was in the original - way fewer if statements

    iter++;

    status1 = cudaMemcpy(eigtemp, &d_ordered[num_Eig - 1], sizeof(double), cudaMemcpyDeviceToHost);

    if (status1 != CUDA_SUCCESS){
      printf("Copying last eigenvalue failed \n");
    }

    cusparseZcsrmv(sparsehandle, *A, *d_dim, *d_dim, *alpha, *d_descrH, d_values, d_rowstart, d_colindex, d_eigen_Array[iter], *beta, d_eigen_Array[iter+1]); // the Hamiltonian is applied here, in this gross expression

    d_a[iter] = cublasZdotc(*d_dim, d_eigen_Array[iter], sizeof(cuDoubleComplex), d_eigen_Array[iter + 1], sizeof(cuDoubleComplex));

    vecdiff<<<bpg,tpb>>>(d_eigen_Array[iter+1], d_eigen_Array[iter+1], d_a[iter], d_eigen_Array[iter], d_b[iter], d_eigen_Array[iter - 1], dim);

    d_b[iter+1] = make_cuDoubleComplex(sqrt(cublasDznrm2(dim, d_eigen_Array[iter+1], sizeof(cuDoubleComplex))),0.);
    
    cuDoubleComplex* gamma;
    cudaMalloc(&gamma, sizeof(cuDoubleComplex));
    *gamma = make_cuDoubleComplex(1./d_b[iter+1].x,0.);
    cublasZaxpy(d_dim, gamma, d_eigen_Array[iter+1], sizeof(cuDoubleComplex), y, sizeof(cuDoubleComplex));
    
    //cublasZcopy(dim, d_v_Mid, sizeof(cuDoubleComplex), d_v_Start, sizeof(cuDoubleComplex)); //switching my vectors around for the next iteration
    //cublasZcopy(dim, d_v_End, sizeof(cuDoubleComplex), d_v_Mid, sizeof(cuDoubleComplex)); unnecesarry now that i'm using the array of pointers

    d_diag[iter] = 0.; //adding another spot in the tridiagonal matrix representation
    d_offdia[iter] = 0.;

    complextodoubler<<<bpg,tpb>>>(d_a, d_diag, iter);
    complextodoubler2<<<bpg,tpb>>>(d_b, d_offdia, iter);

    double** d_H_eigen;
    size_t d_eig_pitch;

    status1 = cudaMallocPitch(&d_H_eigen, &d_eig_pitch, iter*sizeof(double), iter);
    if (status1 != CUDA_SUCCESS){
      printf("tqli eigenvectors matrix memory allocation failed! \n");
    }
    
    zero<<<bpg,tpb>>>(d_H_eigen, iter);
    eye<<<bpg,tpb>>>(d_H_eigen, iter); //set this matrix to the identity

    returned = tqli(d_diag, d_offdia, iter + 1, d_H_eigen); //tqli is in a separate file   

    //assign<<<tpb,bpg>>>(d_ordered, d_diag[0], num_Eig);
    
    
    thrust::sort(dev_ptr, dev_ptr + *d_dim); //sorts the array of eigenvalues    

    cudaMemcpy(d_ordered, d_diag, num_Eig*sizeof(double), cudaMemcpyDeviceToDevice);


    status2 = cudaMemcpy(&gs_Energy, &(d_ordered[num_Eig - 1]), sizeof(double), cudaMemcpyDeviceToHost);

    if (status2 != CUDA_SUCCESS){
      printf("Copying the eigenvalue failed! \n");
    }

    if (iter == sizeof(d_eigen_Array) - 2){// have to use this or d_b will overflow
      //this stuff here is used to resize the main arrays in the case that we aren't converging quickly enough
      //------------------------------------------------------------------------
	cuDoubleComplex* temp;
        status1 = cudaMalloc(&temp, (2*sizeof(d_eigen_Array) + 1)*sizeof(cuDoubleComplex));
        
        status2 = cudaMemcpy(temp, d_a, sizeof(d_eigen_Array)*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        
        cudaFree(d_a);
        status3 = cudaMalloc(&d_a, (2*sizeof(d_eigen_Array) + 1)*sizeof(cuDoubleComplex));
        status4 = cudaMemcpy(d_a, temp, (2*sizeof(d_eigen_Array) + 1)*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        
        if( (status1 != CUDA_SUCCESS) ||
            (status2 != CUDA_SUCCESS) ||
            (status3 != CUDA_SUCCESS) ||
            (status4 != CUDA_SUCCESS) ){
          printf("Resizing d_a failed! \n");
        } 

        status1 = cudaMemcpy(temp, d_b, sizeof(d_eigen_Array)*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);

        cudaFree(d_b);
        status2 = cudaMalloc(&d_b, (2*sizeof(d_eigen_Array) + 1)*sizeof(cuDoubleComplex));
        status3 = cudaMemcpy(d_b, temp, (2*sizeof(d_eigen_Array) + 1)*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);

        if ( (status1 != CUDA_SUCCESS) ||
             (status2 != CUDA_SUCCESS) ||
             (status3 != CUDA_SUCCESS) ){
          printf("Resizing d_b failed! \n");
        }
          
        cudaFree(temp);
        cudaMalloc(&temp, (2*sizeof(d_eigen_Array) + 1)*sizeof(cuDoubleComplex*));
        
        status1 = cudaMemcpy(temp, d_eigen_Array, sizeof(d_eigen_Array)*sizeof(cuDoubleComplex*), cudaMemcpyDeviceToDevice);

        int temp2 = sizeof(d_eigen_Array);

        status2 = cudaFree(d_eigen_Array);
        status3 = cudaMalloc(&d_eigen_Array, (2*temp2 + 1)*sizeof(cuDoubleComplex*));
        status4 = cudaMemcpy(d_eigen_Array, temp, (2*temp2 + 1)*sizeof(cuDoubleComplex*), cudaMemcpyDeviceToDevice);

        if ( (status1 != CUDA_SUCCESS) ||
             (status2 != CUDA_SUCCESS) ||
             (status3 != CUDA_SUCCESS) ||
             (status4 != CUDA_SUCCESS) ){
          printf("Resizing d_eigen_Array failed! \n");
        }

        CDCarraysalloc<<<1, temp2 + 1>>>(d_eigen_Array, dim, temp2);//need to change this
                
        cudaFree(temp); 
        //resizing

    }   
  } 

  double* h_ordered;

  status1 = cudaMallocHost(&h_ordered, num_Eig*sizeof(double)); //a place to put the eigenvalues on the CPU

  if (status1 != CUDA_SUCCESS){
    printf("Memory allocation for host eigenvector array failed! \n");
  }

  status2 = cudaMemcpy(h_ordered, d_ordered, num_Eig*sizeof(double), cudaMemcpyDeviceToHost); // moving the eigenvalues over

  if (status2 != CUDA_SUCCESS){
    printf("Copying eigenvalues from GPU to CPU failed! \n");
  }

  for(int i = 0; i < num_Eig; i++){
    printf("%lf \n", h_ordered[i]);
  } //write out the eigenenergies

  cudaFree(&alpha);
  cudaFree(&beta);
  cudaFree(d_a);
  cudaFree(d_b); //dropping stuff off
  // call the expectation values function
  
  // time to copy back all the eigenvectors
  //int* sizeptr;
  //cudaMemcpy(sizeptr, &sizeof(d_eigen_Array), sizeof(int), cudaMemcpyDeviceToHost);
  
  max_Iter = sizeof(d_eigen_Array);

  cuDoubleComplex** h_eigen_Array;
  status1=cudaMallocHost(&h_eigen_Array, max_Iter*sizeof(cuDoubleComplex*));

  if (status1 != CUDA_SUCCESS){
    printf("CPU eigenvector array memory allocation failed! \n");
  }
  
  for(int i = 0; i < max_Iter; i++){
     status2 = cudaMallocHost(&h_eigen_Array[i], dim*sizeof(cuDoubleComplex*));
     status3 = cudaMemcpy(h_eigen_Array[i], d_eigen_Array[i], dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
     if ( (status2 != CUDA_SUCCESS) || (status3 != CUDA_SUCCESS)){
       printf("Copying an eigenvector array failed! \n");
     }

  } // now the eigenvectors are available on the host CPU

  linalgstat = cublasShutdown();
	
  if (linalgstat != CUBLAS_STATUS_SUCCESS){
    printf("CUBLAS failed to shut down properly! \n");
  }

  sparsestatus = cusparseDestroy(sparsehandle);

  if (sparsestatus != CUSPARSE_STATUS_SUCCESS){
    printf("CUSPARSE failed to release handle! \n");
  }
}
// things left to do:
// write a thing (separate file) to call routines to find expectation values, should be faster on GPU 
// make the tqli thing better!
// do the hamiltonian generating stuff
// change things in here to set device array values properly

