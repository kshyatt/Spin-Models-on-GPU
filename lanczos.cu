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


/*__global__ void zero(double** a, int m){
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  int j = blockDim.y*blockIdx.y + threadIdx.y;

  if ( i< m && j < m){
    a[i][j] = 0. ;
  }
}
// Note: to get the identity matrix, apply the fuction zero above first
__global__ void identity(double** a, int m){
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  
  if (i < m ){

    a[i][i] = 1.;
  }
}

__global__ void arraysalloc(cuDoubleComplex** a, int n, int m){
  int i = threadIdx.x + m;
  a[i] = (cuDoubleComplex*)malloc(n*sizeof(cuDoubleComplex));
}*/

//Function lanczos: takes a hermitian matrix H, tridiagonalizes it, and finds the n smallest eigenvalues - this version only returns eigenvalues, not
// eigenvectors. Doesn't use sparse matrices yet either, derp. Should be a very simple change to make using CUSPARSE, which has functions for operations
// between sparse matrices and dense vectors
//---------------------------------------------------------------------------------------------------------------------------------------------------
// Input: h_H, a Hermitian matrix of complex numbers (not yet sparse)
//        dim, the dimension of the matrix
//        max_Iter, the starting number of iterations we'll try
//        num_Eig, the number of eigenvalues we're interested in seeing
//        conv_req, the convergence we'd like to see
//---------------------------------------------------------------------------------------------------------------------------------------------------
// Output: h_ordered, the array of the num_Eig smallest eigenvalues, ordered from smallest to largest
//---------------------------------------------------------------------------------------------------------------------------------------------------        

void lanczos(const int num_Elem, const cuDoubleComplex* d_H_vals, const int* d_H_rows, const int* d_H_cols, const int dim, int max_Iter, const int num_Eig, const double conv_req){

  cublasStatus_t linalgstat;
  //have to initialize the cuBLAS environment, or my program won't work! I could use this later to check for errors as well
  cublasHandle_t linalghandle;
  linalgstat = cublasCreate(&linalghandle);

  if (linalgstat != CUBLAS_STATUS_SUCCESS){
    std::cout<<"Initializing CUBLAS failed! Error: "<<linalgstat<<std::endl;
  }

  cusparseHandle_t sparsehandle;
  cusparseStatus_t sparsestatus = cusparseCreate(&sparsehandle); //have to initialize the cusparse environment too! This variable gets passed to all my cusparse functions

  if (sparsestatus != CUSPARSE_STATUS_SUCCESS){
    std::cout<<"Failed to initialize CUSPARSE! Error: "<<sparsestatus<<std::endl;
  }

  cusparseMatDescr_t H_descr = 0;
  cusparseCreateMatDescr(&H_descr);
  cusparseSetMatType(H_descr, CUSPARSE_MATRIX_TYPE_HERMITIAN);
  cusparseSetMatIndexBase(H_descr, CUSPARSE_INDEX_BASE_ZERO);

  int* d_H_rowptrs;
  cudaMalloc(&d_H_rowptrs, (dim + 1)*sizeof(int));

  sparsestatus = cusparseXcoo2csr(sparsehandle, d_H_rows, num_Elem, dim, d_H_rowptrs, CUSPARSE_INDEX_BASE_ZERO);

  if (sparsestatus != CUSPARSE_STATUS_SUCCESS){
    std::cout<<"Failed to switch from COO to CSR! Error: "<<sparsestatus<<std::endl;
  }

  cudaError_t status1, status2, status3, status4; //this is to throw errors in case things (mostly memory) in the code fail!  

  thrust::device_vector<cuDoubleComplex> d_a(max_Iter);
  thrust::device_vector<cuDoubleComplex> d_b(max_Iter);
  cuDoubleComplex* d_a_ptr;
  cuDoubleComplex* d_b_ptr; //we need these to pass to kernel functions 


  int tpb = 256; //threads per block - a conventional number
  int bpg = (dim + tpb - 1)/tpb; //blocks per grid

  //Making the "random" starting vector

  thrust::device_vector<cuDoubleComplex> d_lanczvec(dim*max_Iter); //this thing is an array of the Lanczos vectors 

  thrust::device_vector<cuDoubleComplex> v0(dim);
  thrust::device_vector<cuDoubleComplex> v1(dim);
  thrust::device_vector<cuDoubleComplex> v2(dim);
  cuDoubleComplex* v0_ptr = thrust::raw_pointer_cast(&v0[0]);
  cuDoubleComplex* v1_ptr = thrust::raw_pointer_cast(&v1[0]);
  cuDoubleComplex* v2_ptr = thrust::raw_pointer_cast(&v2[0]);

  thrust::fill(v0.begin(), v0.end(), make_cuDoubleComplex(1., 0.));//assigning the values of the "random" starting vector
  
  cuDoubleComplex alpha = make_cuDoubleComplex(1.,0.);
  cuDoubleComplex beta = make_cuDoubleComplex(0.,0.); 

  sparsestatus = cusparseZcsrmv(sparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE, dim, dim, alpha, H_descr, d_H_vals, d_H_rows, d_H_cols, v0_ptr, beta, v1_ptr); // the Hamiltonian is applied here

  if (sparsestatus != CUSPARSE_STATUS_SUCCESS){
    std::cout<<"Getting V1 = H*V0 failed! Error: ";
    std::cout<<sparsestatus<<std::endl;
  }

  //*********************************************************************************************************
  // This is just the first steps so I can do the rest
  d_a_ptr = raw_pointer_cast(&d_a[0]);  
  linalgstat = cublasZdotc(linalghandle, dim, v0_ptr, sizeof(cuDoubleComplex), v0_ptr, sizeof(cuDoubleComplex), d_a_ptr);
  d_b[0] = make_cuDoubleComplex(0.,0.);

  if (linalgstat != CUBLAS_STATUS_SUCCESS){
    std::cout<<"Getting d_a[0] failed! Error: ";
    std::cout<<linalgstat<<std::endl;
  }

  cuDoubleComplex* y;
  status2 = cudaMalloc(&y, dim*sizeof(cuDoubleComplex));

  if (status2 != CUDA_SUCCESS){
          std::cout<<"Memory allocation of y dummy vector failed! Error:";
          std::cout<<cudaGetErrorString( status2 )<<std::endl;
  }
  
  status3 = cudaMemset(&y, 0, dim);
  if( status3 != CUDA_SUCCESS){
          std::cout<<"Setting y failed! Error: ";
          std::cout<<cudaGetErrorString(status3)<<std::endl;
  }
  //assignr<<<bpg,tpb>>>(y,0., dim); //a dummy vector of 0s that i can stick in my functions

  double* double_temp;
  status4 = cudaMalloc(&double_temp, sizeof(double));
  if (status4 != CUDA_SUCCESS){
    std::cout<<"Error allocating double_temp! Error: ";
    std::cout<<cudaGetErrorString(status4)<<std::endl;
  }

  thrust::device_ptr<double> double_temp_ptr(double_temp);

  cuDoubleComplex* cuDouble_temp;
  status1 = cudaMalloc(&cuDouble_temp, sizeof(cuDoubleComplex));
  if (status1 != CUDA_SUCCESS){
    std::cout<<"Error allocating cuDouble_temp! Error: ";
    std::cout<<cudaGetErrorString(status4)<<std::endl;
  }

  thrust::device_ptr<cuDoubleComplex> cuDouble_temp_ptr(cuDouble_temp);


  *cuDouble_temp_ptr = cuCmul(make_cuDoubleComplex(-1., 0), d_a[0]);
  linalgstat = cublasZaxpy(linalghandle, dim, cuDouble_temp, v0_ptr, sizeof(cuDoubleComplex), v1_ptr, sizeof(cuDoubleComplex));
  
  if (linalgstat != CUBLAS_STATUS_SUCCESS){
    std::cout<<"V1 = V1 - alpha*V0 failed! Error: ";
    std::cout<<linalgstat<<std::endl;
  }
 
  d_b_ptr = thrust::raw_pointer_cast(&d_b[1]);
  linalgstat = cublasDznrm2(linalghandle, dim, v1_ptr, sizeof(cuDoubleComplex), double_temp);
  if (linalgstat != CUBLAS_STATUS_SUCCESS){
    std::cout<<"Getting the norm of v1 failed! Error: ";
    std::cout<<linalgstat<<std::endl;
  }
  
  d_b[1] = make_cuDoubleComplex(sqrt(*double_temp_ptr),0.);
  // this function (above) takes the norm
  
  cuDoubleComplex gamma = make_cuDoubleComplex(1./cuCreal(d_b[1]),0.); //alpha = 1/beta in v1 = v1 - alpha*v0

  linalgstat = cublasZaxpy(linalghandle, dim, &gamma, v1_ptr, sizeof(cuDoubleComplex), y, sizeof(cuDoubleComplex)); // function performs a*x + y

  if (linalgstat != CUBLAS_STATUS_SUCCESS){
    std::cout<<"Getting 1/gamma * v1 failed! Error: ";
    std::cout<<linalgstat<<std::endl;
  }

  //Now we're done the first round!
  //*********************************************************************************************************

  thrust::device_vector<double> d_ordered(num_Eig);
  thrust::fill(d_ordered.begin(), d_ordered.end(), 0);
  double* d_ordered_ptr = thrust::raw_pointer_cast(&d_ordered[0]); 

  double gs_Energy = 1.; //the lowest energy

  int returned;

  int iter = 0;

  // In the original code, we started diagonalizing from iter = 5 and above. I start from iter = 1 to minimize issues of control flow
  thrust::device_vector<double> d_diag(max_Iter);
  double* diag_ptr;
  thrust::device_vector<double> d_offdia(max_Iter);
  double* offdia_ptr;

  thrust::device_vector<cuDoubleComplex> temp(dim);
  cuDoubleComplex* temp_ptr = thrust::raw_pointer_cast(&temp[0]);

  double eigtemp = 0.;

  while( fabs(gs_Energy - eigtemp)> conv_req){ //this is a cleaner version than what was in the original - way fewer if statements

    iter++;

    status1 = cudaMemcpy(&eigtemp, d_ordered_ptr, sizeof(double), cudaMemcpyDeviceToHost);

    if (status1 != CUDA_SUCCESS){
      printf("Copying last eigenvalue failed \n");
    }

    sparsestatus = cusparseZcsrmv(sparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE, dim, dim, alpha, H_descr, d_H_vals, d_H_rows, d_H_cols, v1_ptr, beta, v2_ptr); // the Hamiltonian is applied here, in this gross expression

    if (sparsestatus != CUSPARSE_STATUS_SUCCESS){
      std::cout<<"Error applying the Hamiltonian in "<<iter<<"th iteration!";
      std::cout<<"Error: "<<sparsestatus<<std::endl;
    } 

    d_a_ptr = thrust::raw_pointer_cast(&d_a[iter]);
    linalgstat = cublasZdotc(linalghandle, dim, v1_ptr, sizeof(cuDoubleComplex), v2_ptr, sizeof(cuDoubleComplex), d_a_ptr);

    if (linalgstat != CUBLAS_STATUS_SUCCESS){
      std::cout<<"Error getting v1 * v2 in "<<iter<<"th iteration! Error: ";
      std::cout<<linalgstat<<std::endl;
    }

    temp = v1;

    *cuDouble_temp_ptr = cuCdiv(d_b[iter], d_a[iter]);

    linalgstat = cublasZaxpy( linalghandle, dim, cuDouble_temp, v0_ptr, sizeof(cuDoubleComplex), temp_ptr, sizeof(cuDoubleComplex));
    if (linalgstat != CUBLAS_STATUS_SUCCESS){
      std::cout<<"Error getting (d_b/d_a)*v0 + v1 in "<<iter<<"th iteration!";
      std::cout<<"Error: "<<linalgstat<<std::endl;
    }
    
    linalgstat = cublasZaxpy( linalghandle, dim, d_a_ptr, temp_ptr, sizeof(cuDoubleComplex), v2_ptr, sizeof(cuDoubleComplex));
    if (linalgstat != CUBLAS_STATUS_SUCCESS){
      std::cout<<"Error getting v2 + d_a*v1 in "<<iter<<"th iteration! Error: ";
      std::cout<<linalgstat<<std::endl;
    }

    linalgstat = cublasDznrm2( linalghandle, dim, v2_ptr, sizeof(cuDoubleComplex), double_temp);
    if (linalgstat != CUBLAS_STATUS_SUCCESS){
      std::cout<<"Error getting norm of v2 in "<<iter<<"th iteration! Error: ";
      std::cout<<linalgstat<<std::endl;
    }

    d_b[iter + 1] = make_cuDoubleComplex(sqrt(*double_temp_ptr), 0.);
    
    gamma = make_cuDoubleComplex(1./cuCreal(d_b[iter+1]),0.);
    linalgstat = cublasZaxpy(linalghandle, dim, &gamma, v2_ptr, sizeof(cuDoubleComplex), y, sizeof(cuDoubleComplex));
    if (linalgstat != CUBLAS_STATUS_SUCCESS){ 
      std::cout<<"Error getting 1/d_b * v2 in "<<iter<<"th iteration! Error: ";
      std::cout<<linalgstat<<std::endl;
    }

    thrust::copy(v0.begin(), v0.end(), &d_lanczvec[dim*(iter - 1)]);
    thrust::copy(v1.begin(), v1.end(), v0.begin());
    thrust::copy(v2.begin(), v2.end(), v1.begin()); //moving things around

    d_diag[iter] = cuCreal(d_a[iter]); //adding another spot in the tridiagonal matrix representation
    d_offdia[iter + 1] = cuCreal(d_b[iter + 1]);

  //this tqli stuff is a bunch of crap and needs to be fixed  double** d_H_eigen;
   /* size_t d_eig_pitch;

    status1 = cudaMallocPitch(&d_H_eigen, &d_eig_pitch, iter*sizeof(double), iter);
    if (status1 != CUDA_SUCCESS){
      printf("tqli eigenvectors matrix memory allocation failed! \n");
    }
    
    zero<<<bpg,tpb>>>(d_H_eigen, iter);
    identity<<<bpg,tpb>>>(d_H_eigen, iter); //set this matrix to the identity

    returned = tqli(d_diag, d_offdia, iter + 1, d_H_eigen); //tqli is in a separate file   
//
*/
    thrust::sort(d_diag.begin(), d_diag.end());
    thrust::copy(d_diag.begin(), d_diag.begin() + num_Eig, d_ordered.begin());
   
    d_ordered_ptr = thrust::raw_pointer_cast(&d_ordered[num_Eig - 1]);
    status2 = cudaMemcpy(&gs_Energy, d_ordered_ptr, sizeof(double), cudaMemcpyDeviceToHost);

    if (status2 != CUDA_SUCCESS){
      printf("Copying the eigenvalue failed! \n");
    }

    if (iter == max_Iter - 1){// have to use this or d_b will overflow
      //this stuff here is used to resize the main arrays in the case that we aren't converging quickly enough
      d_a.resize(2*max_Iter);
      d_b.resize(2*max_Iter);
      d_diag.resize(2*max_Iter);
      d_offdia.resize(2*max_Iter);
      d_lanczvec.resize(2*max_Iter*dim);
    }   
  } 

  thrust::host_vector<double> h_ordered(num_Eig);
  h_ordered = d_ordered;

  for(int i = 0; i < num_Eig; i++){
    std::cout<<h_ordered[i]<<" ";
  } //write out the eigenenergies
  std::cout<<std::endl;
  cudaFree(double_temp);
  cudaFree(cuDouble_temp);
  // call the expectation values function
  
  // time to copy back all the eigenvectors
  thrust::host_vector<cuDoubleComplex> h_lanczvec(max_Iter*dim);
  h_lanczvec = d_lanczvec;
  
  // now the eigenvectors are available on the host CPU

  linalgstat = cublasDestroy(linalghandle);
	
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
