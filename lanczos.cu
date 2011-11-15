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


__global__ void zero(cuDoubleComplex* a, int m){
  int i = blockDim.x*blockIdx.x + threadIdx.x;

  if ( i < m){
    a[i] = make_cuDoubleComplex(0., 0.) ;
  }
}


__global__ void zero(double* a, int m){
  int i = blockDim.x*blockIdx.x + threadIdx.x;

  if ( i < m ){
    a[i] = 0.;
  }
}

// Note: to get the identity matrix, apply the fuction zero above first
__global__ void unitdiag(double* a, int m){
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  
  if (i < m ){

    a[i + m*i] = 1.;
  }
}


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

__host__ void lanczos(const int num_Elem, cuDoubleComplex*& d_H_vals, int*& d_H_rows, int*& d_H_cols, const int dim, int max_Iter, const int num_Eig, const double conv_req){

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time;

  cublasStatus_t linalgstat;
  //have to initialize the cuBLAS environment, or my program won't work! I could use this later to check for errors as well
  cublasHandle_t linalghandle;
	cudaEventRecord(start, 0);  
	linalgstat = cublasCreate(&linalghandle);

  if (linalgstat != CUBLAS_STATUS_SUCCESS){
    std::cout<<"Initializing CUBLAS failed! Error: "<<linalgstat<<std::endl;
  }

  cusparseHandle_t sparsehandle;
  cusparseStatus_t sparsestatus = cusparseCreate(&sparsehandle); //have to initialize the cusparse environment too! This variable gets passed to all my cusparse functions

  if (sparsestatus != CUSPARSE_STATUS_SUCCESS){
    std::cout<<"Failed to initialize CUSPARSE! Error: "<<sparsestatus<<std::endl;
  }
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	//std::cout<<"Time to initialize libraries: "<<time<<std::endl;

	cudaEventRecord(start,0);
  cusparseMatDescr_t H_descr = 0;
  sparsestatus = cusparseCreateMatDescr(&H_descr);
  if (sparsestatus != CUSPARSE_STATUS_SUCCESS){
    std::cout<<"Error creating matrix description: "<<sparsestatus<<std::endl;
  }
  sparsestatus = cusparseSetMatType(H_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  if (sparsestatus != CUSPARSE_STATUS_SUCCESS){
    std::cout<<"Error setting matrix type: "<<sparsestatus<<std::endl;
  }
  sparsestatus = cusparseSetMatIndexBase(H_descr, CUSPARSE_INDEX_BASE_ZERO);
  if (sparsestatus != CUSPARSE_STATUS_SUCCESS){
    std::cout<<"Error setting matrix index base: "<<sparsestatus<<std::endl;
  }
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	//std::cout<<"Runtime to create description: "<<time<<std::endl;

  cudaError_t status1, status2, status3, status4;

	int* d_H_rowptrs;
	status1 = cudaMalloc(&d_H_rowptrs, (dim + 1)*sizeof(int));
	if (status1 != CUDA_SUCCESS){ 
		std::cout<<"Error allocating d_H_rowptrs: "<<cudaGetErrorString(status1)<<std::endl;
	}

	cudaEventRecord(start, 0);
	cusparseHybMat_t hyb_Ham;	
	cusparseCreateHybMat(&hyb_Ham);

	sparsestatus = cusparseXcoo2csr(sparsehandle, d_H_rows, num_Elem, dim, d_H_rowptrs, CUSPARSE_INDEX_BASE_ZERO);
	sparsestatus = cusparseZcsr2hyb(sparsehandle, dim, dim, H_descr, d_H_vals, d_H_rowptrs, d_H_cols, hyb_Ham, 0, CUSPARSE_HYB_PARTITION_AUTO);
	 
	cudaThreadSynchronize();
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	if (sparsestatus != CUSPARSE_STATUS_SUCCESS){
		std::cout<<"Failed to switch from COO to CSR! CUSPARSE Error: "<<sparsestatus<<std::endl;
	}

	if (cudaPeekAtLastError() != CUDA_SUCCESS){
		std::cout<<"Failed to switch from COO to CSR! Error: "<<cudaGetErrorString( cudaPeekAtLastError())<<std::endl;
	}

	//std::cout<<"Runtime to convert to CSR: "<<time<<std::endl;

	thrust::host_vector<cuDoubleComplex> h_a(max_Iter);

	thrust::host_vector<cuDoubleComplex> h_b(max_Iter);

	//cuDoubleComplex* d_a_ptr;
	//cuDoubleComplex* d_b_ptr; //we need these to pass to kernel functions 

	//Making the "random" starting vector

	thrust::device_vector<cuDoubleComplex> d_lanczvec(dim*max_Iter); //this thing is an array of the Lanczos vectors 
 
	cuDoubleComplex* lancz_ptr = thrust::raw_pointer_cast(&d_lanczvec[0]);

	cuDoubleComplex* v0;
	cuDoubleComplex* v1;
	cuDoubleComplex* v2;
	status1 = cudaMalloc(&v0, dim*sizeof(cuDoubleComplex));
	status2 = cudaMalloc(&v1, dim*sizeof(cuDoubleComplex));
	status3 = cudaMalloc(&v2, dim*sizeof(cuDoubleComplex));

  //thrust::device_ptr<cuDoubleComplex> v0_ptr(v0);
  //thrust::device_ptr<cuDoubleComplex> v1_ptr(v1);
  //thrust::device_ptr<cuDoubleComplex> v2_ptr(v2);

  //thrust::device_vector<cuDoubleComplex> v0(dim);
  //thrust::device_vector<cuDoubleComplex> v1(dim);
  //thrust::device_vector<cuDoubleComplex> v2(dim);
  //cuDoubleComplex* v0_ptr = thrust::raw_pointer_cast(&v0[0]);
  //cuDoubleComplex* v1_ptr = thrust::raw_pointer_cast(&v1[0]);
  //cuDoubleComplex* v2_ptr = thrust::raw_pointer_cast(&v2[0]);

	cudaEventRecord(start, 0);
	cuDoubleComplex* host_v0 = (cuDoubleComplex*)malloc(dim*sizeof(cuDoubleComplex));
  for(int i = 0; i<dim; i++){
    host_v0[i] = make_cuDoubleComplex(0. , 0.);
    if (i%4 == 0) host_v0[i] = make_cuDoubleComplex(1.0, 0.) ;
    else if (i%5 == 0) host_v0[i] = make_cuDoubleComplex(-2.0, 0.);
    else if (i%7 == 0) host_v0[i] = make_cuDoubleComplex(3.0, 0.);
    else if (i%9 == 0) host_v0[i] = make_cuDoubleComplex(-4.0, 0.);

  }

	cudaMemcpy(v0, host_v0, dim*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	//std::cout<<"Time to set and push v0: "<<time<<std::endl;

	double normtemp;

	cudaEventRecord(start,0);
	linalgstat = cublasDznrm2(linalghandle, dim, v0, 1, &normtemp);
	normalize<<<dim/512 + 1, 512>>>(v0, dim, normtemp);

	cuDoubleComplex alpha = make_cuDoubleComplex(1.,0.);
	cuDoubleComplex beta = make_cuDoubleComplex(0.,0.); 

	cudaThreadSynchronize();
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	//std::cout<<"Time to normalize v0: "<<time<<std::endl;

	cudaEventRecord(start,0);
	sparsestatus = cusparseZhybmv(sparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, H_descr, hyb_Ham, v0, &beta, v1); // the Hamiltonian is applied here

	if (sparsestatus != CUSPARSE_STATUS_SUCCESS){
		std::cout<<"Getting V1 = H*V0 failed! Error: ";
		std::cout<<sparsestatus<<std::endl;
	}
	cudaThreadSynchronize();
	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	//std::cout<<"Time to get V1=H*V0: "<<time<<std::endl;
	
	if (sparsestatus != CUSPARSE_STATUS_SUCCESS){
		std::cout<<"Getting V1 = H*V0 failed! Error: ";
		std::cout<<sparsestatus<<std::endl;
	}
	if (cudaPeekAtLastError() != 0 ){
		std::cout<<"Getting V1  = H*V0 failed! Error: ";
		std::cout<<cudaGetErrorString(cudaPeekAtLastError())<<std::endl;
	} 

  //*********************************************************************************************************
  
  // This is just the first steps so I can do the rest
  
  /*try{ 
    d_a_ptr = raw_pointer_cast(&d_a[0]);  
  }
  catch( thrust::system_error e ){
    std::cerr<<"Error settng d_a_ptr: "<<e.what()<<std::endl;
    exit(-1);
  }*/

	cuDoubleComplex dottemp = make_cuDoubleComplex(0. ,0.);
   
	linalgstat = cublasZdotc(linalghandle, dim, v1, 1, v0, 1, &dottemp); 
	if (linalgstat != CUBLAS_STATUS_SUCCESS){
		std::cout<<"Getting d_a[0] failed! Error: ";
		std::cout<<linalgstat<<std::endl;
	  }

	h_a[0] = dottemp;
	//cudaMemcpy(d_a_ptr, &dottemp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

	cudaThreadSynchronize();
	if (linalgstat != CUBLAS_STATUS_SUCCESS){
		std::cout<<"Getting h_a[0] failed! Error: ";
		std::cout<<linalgstat<<std::endl;
	}

	h_b[0] = make_cuDoubleComplex(0., 0.);

	cuDoubleComplex* y;
	status2 = cudaMalloc(&y, dim*sizeof(cuDoubleComplex));

	if (status2 != CUDA_SUCCESS){
		std::cout<<"Memory allocation of y dummy vector failed! Error:";
		std::cout<<cudaGetErrorString( status2 )<<std::endl;
	}
  
	zero<<<dim/512 + 1, 512>>>(y, dim);
	cudaThreadSynchronize();
  
	cuDoubleComplex axpytemp = cuCmul(make_cuDoubleComplex(-1.,0), h_a[0]);

	linalgstat = cublasZaxpy(linalghandle, 0, &axpytemp, v0, 1, v1, 1);
	//std::cout<<axpytemp.x<<" "<<axpytemp.y<<std::endl;
	
	cudaThreadSynchronize();

	if (linalgstat != CUBLAS_STATUS_SUCCESS){
		std::cout<<"V1 = V1 - alpha*V0 failed! Error: ";
		std::cout<<linalgstat<<std::endl;
	}

	if (cudaPeekAtLastError() != 0 ){
		std::cout<<"Getting V1  = V1 - a*V0 failed! Error: ";
		std::cout<<cudaGetErrorString(cudaPeekAtLastError())<<std::endl;
	} 


	std::ofstream fout;
	fout.open("lanczos.log");
	//fout<<normtemp<<std::endl;
	fout<<std::endl;
	//int* h_H_vals = (int*)malloc((dim+1)*sizeof(int));
	cudaMemcpy(host_v0, v1, dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	for(int i = 0; i < dim ; i++){
		fout<<host_v0[i].x<<std::endl;
	}

	fout.close();

	
  
	
	cudaEventRecord(start, 0);
	linalgstat = cublasDznrm2(linalghandle, dim, v1, 1, &normtemp); //this is slow for some reason
  
	cudaThreadSynchronize();
	if (linalgstat != CUBLAS_STATUS_SUCCESS){
		std::cout<<"Getting the norm of v1 failed! Error: ";
		std::cout<<linalgstat<<std::endl;
	}

	
	if (cudaPeekAtLastError() != 0 ){
		std::cout<<"Getting nrm(V1) failed! Error: ";
		std::cout<<cudaGetErrorString(cudaPeekAtLastError())<<std::endl;
	} 

 
	//d_b_ptr = thrust::raw_pointer_cast(&d_b[1]);

	h_b[1] = make_cuDoubleComplex(normtemp,0.);
	// this function (above) takes the norm
  	std::cout<<normtemp<<std::endl;
	normtemp = 1./normtemp;
	cuDoubleComplex gamma = make_cuDoubleComplex(1./cuCreal(h_b[1]),0.); //alpha = 1/beta in v1 = v1 - alpha*v0
  	//normalize<<<dim/512 + 1, 512>>>(v0, dim, normtemp);
	linalgstat = cublasZdscal(linalghandle, dim, &normtemp, v1, 1);
	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	if (linalgstat != CUBLAS_STATUS_SUCCESS){
		std::cout<<"Normalizing v1 failed! Error: ";
		std::cout<<linalgstat<<std::endl;
	}

	
	if (cudaPeekAtLastError() != 0 ){
		std::cout<<"Normalizing V1 failed! Error: ";
		std::cout<<cudaGetErrorString(cudaPeekAtLastError())<<std::endl;
	} 


	//std::cout<<"Time to normalize v1: "<<time<<std::endl;

	
	//Now we're done the first round!
	//*********************************************************************************************************

	/*thrust::device_vector<double> d_ordered(num_Eig);
	thrust::fill(d_ordered.begin(), d_ordered.end(), 0);
	double* d_ordered_ptr = thrust::raw_pointer_cast(&d_ordered[0]); */
	cudaEventRecord(start, 0);
	double gs_Energy = 1.; //the lowest energy

	int returned;

	int iter = 0;

  // In the original code, we started diagonalizing from iter = 5 and above. I start from iter = 1 to minimize issues of control flow
  /*thrust::device_vector<double> d_diag(max_Iter);
  double* diag_ptr;
  thrust::device_vector<double> d_offdia(max_Iter);
  double* offdia_ptr;*/
  
	thrust::host_vector<double> h_diag(max_Iter);
	double* h_diag_ptr = raw_pointer_cast(&h_diag[0]);
	thrust::host_vector<double> h_offdia(max_Iter);
	double* h_offdia_ptr = raw_pointer_cast(&h_offdia[0]);
  
  thrust::device_vector<cuDoubleComplex> temp(dim);
  cuDoubleComplex* temp_ptr = thrust::raw_pointer_cast(&temp[0]);

  double eigtemp = 0.;

  thrust::host_vector<double> h_ordered(num_Eig, 0.);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	//std::cout<<"Time to set up the arrays for iteration: "<<time<<std::endl;

  while( fabs(gs_Energy - eigtemp) > conv_req || iter < 10){ //this is a cleaner version than what was in the original - way fewer if statements

    iter++;

	eigtemp = h_ordered[num_Eig - 1];
    /*status1 = cudaMemcpy(&eigtemp, d_ordered_ptr, sizeof(double), cudaMemcpyDeviceToHost);

    if (status1 != CUDA_SUCCESS){
      printf("Copying last eigenvalue failed \n");
    }*/
    //std::cout<<"Getting V2 = H*V1 for the "<<iter + 1<<"th time"<<std::endl;
		cudaEventRecord(start, 0);
    sparsestatus = cusparseZhybmv(sparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, H_descr, hyb_Ham, v1, &beta, v2); // the Hamiltonian is applied here, in this gross expression
    cudaThreadSynchronize();

		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		//std::cout<<"Time to do csrmv: "<<time<<std::endl;

    if (sparsestatus != CUSPARSE_STATUS_SUCCESS){
      std::cout<<"Error applying the Hamiltonian in "<<iter<<"th iteration!";
      std::cout<<"Error: "<<sparsestatus<<std::endl;
    } 

    //d_a_ptr = thrust::raw_pointer_cast(&d_a[iter]);
    //std::cout<<"Getting V1*V2 for the "<<iter + 1<<"th time"<<std::endl;
    
		cudaEventRecord(start, 0);
		linalgstat = cublasZdotc(linalghandle, dim, v1, 1, v2, 1, &dottemp);
    cudaThreadSynchronize();

		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		//std::cout<<"Time to get v1*v2: "<<time<<std::endl;

    h_a[iter] = dottemp;

    if (linalgstat != CUBLAS_STATUS_SUCCESS){
      std::cout<<"Error getting v1 * v2 in "<<iter<<"th iteration! Error: ";
      std::cout<<linalgstat<<std::endl;
    }

    //cudaMemcpy(temp_ptr, v1, dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    //temp = v1;

    axpytemp = cuCmul(make_cuDoubleComplex(-1., 0.), h_b[iter]);
    cudaEventRecord(start, 0);
    linalgstat = cublasZaxpy( linalghandle, dim, &axpytemp, v0, 1, v2, 1);
    if (linalgstat != CUBLAS_STATUS_SUCCESS){
      std::cout<<"Error getting (d_b/d_a)*v0 + v1 in "<<iter<<"th iteration!";
      std::cout<<"Error: "<<linalgstat<<std::endl;
    }
    cudaThreadSynchronize();
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		//std::cout<<"Time to get v2 - b[i]*v0: "<<time<<std::endl;

    axpytemp = cuCmul(make_cuDoubleComplex(-1., 0.), h_a[iter]);
    linalgstat = cublasZaxpy( linalghandle, dim, &axpytemp, v1, 1, v2, 1);
    if (linalgstat != CUBLAS_STATUS_SUCCESS){
      std::cout<<"Error getting v2 + d_a*v1 in "<<iter<<"th iteration! Error: ";
      std::cout<<linalgstat<<std::endl;
    }

    //std::cout<<"Getting norm of V2 for the "<<iter + 1<<"th time"<<std::endl;
		cudaEventRecord(start, 0);
    linalgstat = cublasDznrm2( linalghandle, dim, v2, 1, &normtemp);
    if (linalgstat != CUBLAS_STATUS_SUCCESS){
      std::cout<<"Error getting norm of v2 in "<<iter<<"th iteration! Error: ";
      std::cout<<linalgstat<<std::endl;
    }

		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		//std::cout<<"Time to get norm of v2: "<<time<<std::endl;

    h_b[iter + 1] = make_cuDoubleComplex(normtemp, 0.);
    gamma = make_cuDoubleComplex(1./normtemp,0.);

    linalgstat = cublasZscal(linalghandle, dim, &gamma, v2, 1);
    if (linalgstat != CUBLAS_STATUS_SUCCESS){ 
      std::cout<<"Error getting 1/d_b * v2 in "<<iter<<"th iteration! Error: ";
      std::cout<<linalgstat<<std::endl;
    }

		cudaEventRecord(start, 0);
	//lancz_ptr = raw_pointer_cast(&d_lanczvec[dim*(iter - 1)]);
	//cudaMemcpy(lancz_ptr, v0, dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
	cudaMemcpy(v0, v1, dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
	cudaMemcpy(v1, v2, dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
	cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		//std::cout<<"Time to copy around the Lanczos vectors: "<<time<<std::endl;
    
    for (int i = 0; i <= iter; i++){
			h_diag[i] = cuCreal(h_a[i]); //adding another spot in the tridiagonal matrix representation
    	h_offdia[i] = cuCreal(h_b[i]);
		}
		
  //this tqli stuff is a bunch of crap and needs to be fixed  
    //double* d_H_eigen;
    //size_t d_eig_pitch;

    /*status1 = cudaMalloc(&d_H_eigen, max_Iter*max_Iter*sizeof(double));
    if (status1 != CUDA_SUCCESS){
      printf("tqli eigenvectors matrix memory allocation failed! \n");
    }
    
    zero<<<(iter*iter)/512 + 1, 512>>>(d_H_eigen, iter*iter);
    unitdiag<<<iter/512 + 1, 512>>>(d_H_eigen, iter); //set this matrix to the identity */
    //h_diag = d_diag;
    //h_offdia = d_offdia;

    double* h_H_eigen = (double*)malloc(max_Iter*max_Iter*sizeof(double));
    //cudaMemcpy(h_H_eigen, d_H_eigen, max_Iter*max_Iter*sizeof(double), cudaMemcpyDeviceToHost);
    for (int ii=1;ii<=iter;ii++){
        h_offdia[ii-1] = h_offdia[ii];
    }
    h_offdia[iter] = 0;
		cudaEventRecord(start, 0);
    		returned = tqli(h_diag_ptr, h_offdia_ptr, iter + 1, max_Iter, h_H_eigen); //tqli is in a separate file   
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		//std::cout<<"Time to run TQLI: "<<time<<std::endl;
//
    //d_diag = h_diag;
    
		cudaEventRecord(start, 0);
		thrust::sort(h_diag.begin(), h_diag.end());
    thrust::copy(h_diag.begin(), h_diag.begin() + num_Eig, h_ordered.begin());
		cudaEventRecord(stop, 0);		
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		//std::cout<<"Runtime for sort and copy: "<<time<<std::endl;

		//std::sort(h_diag.begin(), h_diag.end());
   
	gs_Energy = h_ordered[num_Eig - 1];
    //d_ordered_ptr = thrust::raw_pointer_cast(&d_ordered[num_Eig - 1]);
    //status2 = cudaMemcpy(&gs_Energy, d_ordered_ptr, sizeof(double), cudaMemcpyDeviceToHost);
    
    /*h_ordered = d_ordered;

		if (status2 != CUDA_SUCCESS){
      printf("Copying the eigenvalue failed! \n");
    }*/

	for(int i = 0; i < num_Eig; i++){
   		std::cout<<std::setprecision(12)<<h_ordered[i]<<" ";
  	} 
	std::cout<<std::endl;


    if (iter == max_Iter - 2){// have to use this or d_b will overflow
      //this stuff here is used to resize the main arrays in the case that we aren't converging quickly enough
      h_a.resize(2*max_Iter);
      h_b.resize(2*max_Iter);
      //d_diag.resize(2*max_Iter);
      //d_offdia.resize(2*max_Iter);
      h_diag.resize(2*max_Iter);
      h_offdia.resize(2*max_Iter);
      d_lanczvec.resize(2*max_Iter*dim);
      max_Iter *= 2;
    }
    //cudaFree(d_H_eigen);
       
  } 

  
  for(int i = 0; i < num_Eig; i++){
    std::cout<<h_ordered[i]<<" ";
  } //write out the eigenenergies
  std::cout<<std::endl;
  // call the expectation values function
  
  // time to copy back all the eigenvectors
  //thrust::host_vector<cuDoubleComplex> h_lanczvec(max_Iter*dim);
  //h_lanczvec = d_lanczvec;
  
  // now the eigenvectors are available on the host CPU

  linalgstat = cublasDestroy(linalghandle);
	
  if (linalgstat != CUBLAS_STATUS_SUCCESS){
    printf("CUBLAS failed to shut down properly! \n");
  }

  sparsestatus = cusparseDestroy(sparsehandle);

  if (sparsestatus != CUSPARSE_STATUS_SUCCESS){
    printf("CUSPARSE failed to release handle! \n");
  }
  cudaFree(v0);
  cudaFree(v1);
  cudaFree(v2);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

}
// things left to do:
// write a thing (separate file) to call routines to find expectation values, should be faster on GPU 
// make the tqli thing better!

__global__ void normalize(cuDoubleComplex* v, const int size, double norm){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i < size){
    v[i] = cuCdiv(v[i], make_cuDoubleComplex(norm, 0. ));
  }
}

int tqli(double* d, double* e, int n, int max_Iter, double *z)

{
 
  int m,l,iter,i,k;
  double s,r,p,g,f,dd,c,b;

  for (l=0;l<n;l++) {
    iter=0;
    do { 
      for (m=l;m<n-1;m++) { 
	dd=fabs(d[m])+fabs(d[m+1]);
	if (fabs(e[m])+dd == dd) break;
      }
      if (m!=l) { 
	if (iter++ == 30) { 
	  std::cout <<"Too many iterations in tqli() \n";
	  return 0;
	}
	g=(d[l+1]-d[l])/(2.0*e[l]);
	r=sqrt((g*g)+1.0);
	g=d[m]-d[l]+e[l]/(g+SIGN(r,g));
	s=c=1.0;
	p=0.0;
	for (i=m-1;i>=l;i--) { 
	  f=s*e[i];
	  b=c*e[i];
	  if (fabs(f) >= fabs(g)) { 
	    c=g/f;r=sqrt((c*c)+1.0);
	    e[i + 1]=f*r;
	    c *= (s=1.0/r);
	  }
	  else { 
	    s=f/g;r=sqrt((s*s)+1.0);
	    e[i+1]=g*r;
	    s *= (c=1.0/r);
	  }
	  g=d[i+1]-p;
	  r=(d[i]-g)*s+2.0*c*b;
	  p=s*r;
	  d[i+1]=g+p;
	  g=c*r-b;
	  /*EVECTS*/
	  /*
	    for (k=0;k<n;k++) { 
	      f=z(k,i+1);
	      z(k,i+1)=s*z(k,i)+c*f;
	      z(k,i)=c*z(k,i)-s*f;
	    }
	  */
	}
	d[l]=d[l]-p;
	e[l]=g;
	e[m]=0.0;
      }
    } while (m!=l);
  }
  return 1;
}


double pythag(double a, double b){
  double absa, absb;
  absa=fabs(a);
  absb=fabs(b);
  if (absa > absb) return absa*sqrt(1.0+(absb/absa)*(absb/absa));
  else return (absb == 0.0 ? 0.0 : absb*sqrt(1.0+(absa/absb)*(absa/absb)));
}
