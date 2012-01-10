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


__global__ void zero(cuDoubleComplex* a, int m)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if ( i < m)
    {
        a[i] = make_cuDoubleComplex(0., 0.) ;
    }
}


__global__ void zero(double* a, int m)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if ( i < m )
    {
        a[i] = 0.;
    }
}

// Note: to get the identity matrix, apply the fuction zero above first
__global__ void unitdiag(double* a, int m)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i < m )
    {

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

__host__ void lanczos(const int how_many, const int* num_Elem, d_hamiltonian*& Hamiltonian, int max_Iter, const int num_Eig, const double conv_req)
{

	int* dim = (int*)malloc(how_many*sizeof(int));
	for(int i = 0; i < how_many; i++)
	{
		dim[i] = Hamiltonian[i].sectordim;
	}

	cudaStream_t stream[how_many];

    cublasStatus_t cublas_status[how_many];
    //have to initialize the cuBLAS environment, or my program won't work! I could use this later to check for errors as well
    cublasHandle_t linalghandle;
    cublas_status[0] = cublasCreate(&linalghandle);

    if (cublas_status[0] != CUBLAS_STATUS_SUCCESS)
    {
        std::cout<<"Initializing CUBLAS failed! Error: "<<cublas_status[0]<<std::endl;
    }

    cusparseHandle_t sparsehandle;
    cusparseStatus_t cusparse_status[how_many];
	cusparse_status[0] = cusparseCreate(&sparsehandle); //have to initialize the cusparse environment too! This variable gets passed to all my cusparse functions

    if (cusparse_status[0] != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout<<"Failed to initialize CUSPARSE! Error: "<<cusparse_status[0]<<std::endl;
    }
    
	cout<<"Done initializing libraries"<<endl;
    cusparseMatDescr_t H_descr[how_many];
    for(int i = 0; i<how_many; i++)
	{
		cusparse_status[i] = cusparseCreateMatDescr(&H_descr[i]);
		//H_descr[i] = 0;
	    if (cusparse_status[i] != CUSPARSE_STATUS_SUCCESS)
	    {
        	std::cout<<"Error creating matrix description: "<<cusparse_status[i]<<std::endl;
	    }
	    cusparse_status[i] = cusparseSetMatType(H_descr[i], CUSPARSE_MATRIX_TYPE_GENERAL);
    	if (cusparse_status[i] != CUSPARSE_STATUS_SUCCESS)
    	{
   			std::cout<<"Error setting matrix type: "<<cusparse_status[i]<<std::endl;
    	}
    	cusparse_status[i] = cusparseSetMatIndexBase(H_descr[i], CUSPARSE_INDEX_BASE_ZERO);
    	if (cusparse_status[i] != CUSPARSE_STATUS_SUCCESS)
    	{
        	std::cout<<"Error setting matrix index base: "<<cusparse_status[i]<<std::endl;
    	}
    	
	}
    cudaError_t status[how_many];
	cout<<"Done creating descriptions"<<endl;
    int** d_H_rowptrs;
	d_H_rowptrs = (int**)malloc(how_many*sizeof(int*));

    for(int i = 0; i < how_many; i++)
	{
		status[i] = cudaMalloc(&d_H_rowptrs[i], (dim[i] + 1)*sizeof(int));
    	if (status[i] != CUDA_SUCCESS)
    	{
        	std::cout<<"Error allocating d_H_rowptrs: "<<cudaGetErrorString(status[i])<<std::endl;
    	}
	}

    
    cusparseHybMat_t hyb_Ham[how_many];
    for(int i = 0; i < how_many; i++)
    { 
		cusparse_status[i] = cusparseCreateHybMat(&hyb_Ham[i]);
	if (cusparse_status[i] != CUSPARSE_STATUS_SUCCESS)
    	{
        	std::cout<<"Error creating HYB matrix: "<<cusparse_status[i]<<std::endl;
    	}

	cout<<"Done creating HYB matrices"<<endl;

	cusparse_status[i] = cusparseSetStream(sparsehandle, stream[i]);
	if (cusparse_status[i] != CUSPARSE_STATUS_SUCCESS)
    	{
        	std::cout<<"Error switching streams: "<<cusparse_status[i]<<std::endl;
    	}

		
	cout<<"Done changing streams"<<endl;

	//status[i] = cudaStreamSynchronize(stream[i]);

	//if (status[i] != CUDA_SUCCESS)
    	//{
        	//std::cout<<"Error synchronizing stream: "<<cudaGetErrorString(status[i])<<std::endl;
    	//}

	status[i] = cudaPeekAtLastError();
	if (status[i] != CUDA_SUCCESS)
    	{
        	std::cout<<"Error synchronizing stream: "<<cudaGetErrorString(status[i])<<std::endl;
    	}

	cout<<"Done synchronizing stream"<<endl;

    	cusparse_status[i] = cusparseXcoo2csr(sparsehandle, Hamiltonian[i].rows, num_Elem[i], dim[i], d_H_rowptrs[i], CUSPARSE_INDEX_BASE_ZERO);

	if (cusparse_status[i] != CUSPARSE_STATUS_SUCCESS)
    	{
        	std::cout<<"Error converting to CSR: "<<cusparse_status[i]<<std::endl;
    	}

	//status[i] = cudaStreamSynchronize(stream[i]);

	//if (status[i] != CUDA_SUCCESS)
    	//{
        	//std::cout<<"Error synchronizing stream: "<<cudaGetErrorString(status[i])<<std::endl;
    	//}
	status[i] = cudaPeekAtLastError();
	if (status[i] != CUDA_SUCCESS)
    	{
        	std::cout<<"Error synchronizing stream: "<<cudaGetErrorString(status[i])<<std::endl;
    	}

	cout<<"Done converting to CSR"<<endl;
    	cusparse_status[i] = cusparseZcsr2hyb(sparsehandle, dim[i], dim[i], H_descr[i], Hamiltonian[i].vals, d_H_rowptrs[i], Hamiltonian[i].cols, hyb_Ham[i], 0, CUSPARSE_HYB_PARTITION_AUTO);

		if (cusparse_status[i] != CUSPARSE_STATUS_SUCCESS)
    	{
        	std::cout<<"Error converting to HYB: "<<cusparse_status[i]<<std::endl;
    	}

	}


	cout<<"Done converting to HYB"<<endl;
    cudaThreadSynchronize();
    
	
	/*
    if (sparsestatus != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout<<"Failed to switch from COO to CSR! CUSPARSE Error: "<<sparsestatus<<std::endl;
    }

    if (cudaPeekAtLastError() != CUDA_SUCCESS)
    {
        std::cout<<"Failed to switch from COO to CSR! Error: "<<cudaGetErrorString( cudaPeekAtLastError())<<std::endl;
    }*/

    //std::cout<<"Runtime to convert to CSR: "<<time<<std::endl;

    vector< vector<cuDoubleComplex> > h_a(how_many);

    vector< vector<cuDoubleComplex> > h_b(how_many);
    //Making the "random" starting vector

    //thrust::device_vector<cuDoubleComplex> d_lanczvec(dim*max_Iter); //this thing is an array of the Lanczos vectors

    //cuDoubleComplex* lancz_ptr = thrust::raw_pointer_cast(&d_lanczvec[0]);

    cuDoubleComplex** v0 = (cuDoubleComplex**)malloc(how_many*sizeof(cuDoubleComplex*));
    cuDoubleComplex** v1 = (cuDoubleComplex**)malloc(how_many*sizeof(cuDoubleComplex*));
    cuDoubleComplex** v2 = (cuDoubleComplex**)malloc(how_many*sizeof(cuDoubleComplex*));

	cuDoubleComplex** host_v0 = (cuDoubleComplex**)malloc(how_many*sizeof(cuDoubleComplex*));

    for(int i = 0; i < how_many; i++)
	{
		status[i] = cudaMalloc(&v0[i], dim[i]*sizeof(cuDoubleComplex));
    	status[i] = cudaMalloc(&v1[i], dim[i]*sizeof(cuDoubleComplex));
    	status[i] = cudaMalloc(&v2[i], dim[i]*sizeof(cuDoubleComplex));
		host_v0[i] = (cuDoubleComplex*)malloc(dim[i]*sizeof(cuDoubleComplex));
    	
    	
    	for(int j = 0; j<dim[i]; j++)
    	{
        	host_v0[i][j] = make_cuDoubleComplex(0. , 0.);
        	if (j%4 == 0) host_v0[i][j] = make_cuDoubleComplex(1.0, 0.) ;
        	else if (j%5 == 0) host_v0[i][j] = make_cuDoubleComplex(-2.0, 0.);
        	else if (j%7 == 0) host_v0[i][j] = make_cuDoubleComplex(3.0, 0.);
        	else if (j%9 == 0) host_v0[i][j] = make_cuDoubleComplex(-4.0, 0.);

	    }

    	status[i] = cudaMemcpyAsync(v0[i], host_v0[i], dim[i]*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream[i]);
		if (status[i] != CUDA_SUCCESS)
    	{
        	std::cout<<"Error copying v0 to the device: "<<cudaGetErrorString(status[i])<<std::endl;
    	}
	}
    cout<<"Done creating and copying v0"<<endl;

    double* normtemp = (double*)malloc(how_many*sizeof(double));
	cuDoubleComplex* alpha = (cuDoubleComplex*)malloc(how_many*sizeof(cuDoubleComplex));
	cuDoubleComplex* beta = (cuDoubleComplex*)malloc(how_many*sizeof(cuDoubleComplex));

	cuDoubleComplex* dottemp = (cuDoubleComplex*)malloc(how_many*sizeof(cuDoubleComplex));
	cuDoubleComplex* axpytemp = (cuDoubleComplex*)malloc(how_many*sizeof(cuDoubleComplex));

    for(int i = 0; i < how_many; i++)
	{
		cublasSetStream(linalghandle, stream[i]);
		cusparseSetStream(sparsehandle, stream[i]);
		cublas_status[i] = cublasDznrm2(linalghandle, dim[i], v0[i], 1, &normtemp[i]);
		normtemp[i] = 1./normtemp[i];
		cublas_status[i] = cublasZdscal(linalghandle, dim[i], &normtemp[i], v0[i], 1);

    	//normalize<<<dim[i]/512 + 1, 512, 0, stream[i]>>>(v0[i], dim[i], normtemp[i]);

    	alpha[i] = make_cuDoubleComplex(1.,0.);
    	beta[i] = make_cuDoubleComplex(0.,0.);

    	//cudaThreadSynchronize();
    	
    	cusparse_status[i] = cusparseZhybmv(sparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha[i], H_descr[i], hyb_Ham[i], v0[i], &beta[i], v1[i]); // the Hamiltonian is applied here


		cout<<"Done getting v1 = H*v0"<<endl;
    	if (cusparse_status[i] != CUSPARSE_STATUS_SUCCESS)
    	{
        	std::cout<<"Getting V1 = H*V0 failed! Error: ";
        	std::cout<<cusparse_status[i]<<std::endl;
    	}
    	cudaStreamSynchronize(stream[i]);

    	if (cusparse_status[i] != CUSPARSE_STATUS_SUCCESS)
    	{
        	std::cout<<"Getting V1 = H*V0 failed! Error: ";
       		std::cout<<cusparse_status[i]<<std::endl;
    	}
    	if (cudaPeekAtLastError() != 0 )
    	{
        	std::cout<<"Getting V1  = H*V0 failed! Error: ";
       		std::cout<<cudaGetErrorString(cudaPeekAtLastError())<<std::endl;
    	}

    	//*********************************************************************************************************

    	// This is just the first steps so I can do the rest
	}
    
	cuDoubleComplex** y;
	y = (cuDoubleComplex**)malloc(how_many*sizeof(cuDoubleComplex*));

	for(int i = 0; i < how_many; i++)
	{
		cublasSetStream(linalghandle, stream[i]);
		dottemp[i] = make_cuDoubleComplex(0. ,0.);
	   	cublas_status[i] = cublasZdotc(linalghandle, dim[i], v1[i], 1, v0[i], 1, &dottemp[i]);
	   	if (cublas_status[i] != CUBLAS_STATUS_SUCCESS)
    	{
       		std::cout<<"Getting d_a[0] failed! Error: ";
       		std::cout<<cublas_status[i]<<std::endl;
    	}

    	h_a[i].push_back(dottemp[i]);
    	//cudaMemcpy(d_a_ptr, &dottemp, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    	cudaStreamSynchronize(stream[i]);
    	if (cublas_status[i] != CUBLAS_STATUS_SUCCESS)
    	{
        	std::cout<<"Getting h_a[0] failed! Error: ";
        	std::cout<<cublas_status[i]<<std::endl;
    	}

    	h_b[i].push_back(make_cuDoubleComplex(0., 0.));

    	

    	if (status[i] != CUDA_SUCCESS)
    	{
        	std::cout<<"Memory allocation of y dummy vector failed! Error:";
        	std::cout<<cudaGetErrorString( status[i] )<<std::endl;
    	}

		status[i] = cudaMalloc(&y[i], dim[i]*sizeof(cuDoubleComplex));

    	cublas_status[i] = cublasZscal(linalghandle, dim[i], &beta[i], y[i], 1);
    	cudaStreamSynchronize(stream[i]);

    	axpytemp[i] = cuCmul(make_cuDoubleComplex(-1.,0), h_a[i][0]);

    	cublas_status[i] = cublasZaxpy(linalghandle, 0, &axpytemp[i], v0[i], 1, v1[i], 1);
    	//std::cout<<axpytemp.x<<" "<<axpytemp.y<<std::endl;

    	cudaStreamSynchronize(stream[i]);

    	if (cublas_status[i] != CUBLAS_STATUS_SUCCESS)
    	{
        	std::cout<<"V1 = V1 - alpha*V0 failed! Error: ";
        	std::cout<<cublas_status[i]<<std::endl;
    	}

   		if (cudaPeekAtLastError() != 0 )
    	{
        	std::cout<<"Getting V1  = V1 - a*V0 failed! Error: ";
        	std::cout<<cudaGetErrorString(cudaPeekAtLastError())<<std::endl;
    	}

    	cublas_status[i] = cublasDznrm2(linalghandle, dim[i], v1[i], 1, &normtemp[i]); //this is slow for some reason

    	cudaStreamSynchronize(stream[i]);
    	if (cublas_status[i] != CUBLAS_STATUS_SUCCESS)
    	{
        	std::cout<<"Getting the norm of v1 failed! Error: ";
        	std::cout<<cublas_status[i]<<std::endl;
    	}


    	if (cudaPeekAtLastError() != 0 )
    	{
        	std::cout<<"Getting nrm(V1) failed! Error: ";
        	std::cout<<cudaGetErrorString(cudaPeekAtLastError())<<std::endl;
    	}
	}

	cuDoubleComplex* gamma = (cuDoubleComplex*)malloc(how_many*sizeof(cuDoubleComplex));

	for(int i = 0; i < how_many; i++)
	{
		cublasSetStream(linalghandle, stream[i]);
    	//d_b_ptr = thrust::raw_pointer_cast(&d_b[1]);

    	h_b[i].push_back(make_cuDoubleComplex(normtemp[i], 0.));
    	
    	std::cout<<normtemp[i]<<std::endl;
    	normtemp[i] = 1./normtemp[i];
    	gamma[i] = make_cuDoubleComplex(1./cuCreal(h_b[i][1]),0.); //alpha = 1/beta in v1 = v1 - alpha*v0
    	//normalize<<<dim/512 + 1, 512>>>(v0, dim, normtemp);
    	cublas_status[i] = cublasZdscal(linalghandle, dim[i], &normtemp[i], v1[i], 1);
    	//cudaStreamSynchronize(stream[i]);
    	

    	if (cublas_status[i] != CUBLAS_STATUS_SUCCESS)
    	{
        	std::cout<<"Normalizing v1 failed! Error: ";
        	std::cout<<cublas_status[i]<<std::endl;
    	}


    	if (cudaPeekAtLastError() != 0 )
    	{
        	std::cout<<"Normalizing V1 failed! Error: ";
        	std::cout<<cudaGetErrorString(cudaPeekAtLastError())<<std::endl;
    	}
	}


	cout<<"Done first round"<<endl;
    //std::cout<<"Time to normalize v1: "<<time<<std::endl;


    //Now we're done the first round!
    //*********************************************************************************************************

    /*thrust::device_vector<double> d_ordered(num_Eig);
    thrust::fill(d_ordered.begin(), d_ordered.end(), 0);
    double* d_ordered_ptr = thrust::raw_pointer_cast(&d_ordered[0]); */
    //cudaEventRecord(start, 0);
    double* gs_Energy = (double*)malloc(how_many*sizeof(double));

	double* eigtemp = (double*)malloc(how_many*sizeof(double));

    int* returned = (int*)malloc(how_many*sizeof(int));

    int* iter = (int*)malloc(how_many*sizeof(int));
	
	bool* done_flag = (bool*)malloc(how_many*sizeof(bool));

	double** h_H_eigen = (double**)malloc(how_many*sizeof(double*));
	
    // In the original code, we started diagonalizing from iter = 5 and above. I start from iter = 1 to minimize issues of control flow
    /*thrust::device_vector<double> d_diag(max_Iter);
    double* diag_ptr;
    thrust::device_vector<double> d_offdia(max_Iter);
    double* offdia_ptr;*/

	vector< vector< double > > h_diag(how_many);
	vector< vector< double > > h_offdia(how_many);

    //thrust::host_vector<double> h_diag(max_Iter);
    //double* h_diag_ptr = raw_pointer_cast(&h_diag[0]);
    //thrust::host_vector<double> h_offdia(max_Iter);
    //double* h_offdia_ptr = raw_pointer_cast(&h_offdia[0]);

	vector< vector < double > > h_ordered(how_many);

	for(int i = 0; i<how_many; i++){
		gs_Energy[i] = 1.;
		eigtemp[i] = 0.;
		iter = 0;
		done_flag = false;
		h_ordered[i].resize(num_Eig, 0.);
		h_H_eigen[i] = (double*)malloc(max_Iter*max_Iter*sizeof(double));
	}

    bool all_done = false;
    
	while( !all_done )
	{

		for(int i = 0; i < how_many; i++)
		{
			cublasSetStream(linalghandle, stream[i]);
			cusparseSetStream(sparsehandle, stream[i]);
			cudaStreamSynchronize(stream[i]);
			done_flag[i] = (fabs(gs_Energy[i] - eigtemp[i]) > conv_req || iter[i] < 10);
		
			if (!done_flag[i])
			{
				iter[i]++;
				eigtemp[i] = h_ordered[i][num_Eig - 1];
				cusparse_status[i] = cusparseZhybmv(sparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha[i], H_descr[i], hyb_Ham[i], v1[i], &beta[i], v2[i]); // the Hamiltonian is applied here, in this gross expression
			}
		}

		for(int i = 0; i < how_many; i++)
		{
			cublasSetStream(linalghandle, stream[i]);	
			if (!done_flag[i])
			{	
		   		if (cusparse_status[i] != CUSPARSE_STATUS_SUCCESS)
		    	{
		        	std::cout<<"Error applying the Hamiltonian in "<<iter[i]<<"th iteration!";
		        	std::cout<<"Error: "<<cusparse_status[i]<<std::endl;
		    	}

		    	cublas_status[i] = cublasZdotc(linalghandle, dim[i], v1[i], 1, v2[i], 1, &dottemp[i]);
		    	cudaStreamSynchronize(stream[i]);

		    	h_a[i].push_back(dottemp[i]);

				if (cublas_status[i] != CUBLAS_STATUS_SUCCESS)
		    	{
		        	std::cout<<"Error getting v1 * v2 in "<<iter[i]<<"th iteration! Error: ";
		        	std::cout<<cublas_status[i]<<std::endl;
		    	}

		    	//cudaMemcpy(temp_ptr, v1, dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
		    	//temp = v1;

		    	axpytemp[i] = cuCmul(make_cuDoubleComplex(-1., 0.), h_b[i][iter[i]]);

		    	cublas_status[i] = cublasZaxpy( linalghandle, dim[i], &axpytemp[i], v0[i], 1, v2[i], 1);
		    	if (cublas_status[i] != CUBLAS_STATUS_SUCCESS)
		    	{
		        	std::cout<<"Error getting (d_b/d_a)*v0 + v1 in "<<iter[i]<<"th iteration!";
		        	std::cout<<"Error: "<<cublas_status[i]<<std::endl;
		    	}

			}
		}

		for(int i = 0; i < how_many; i++)
		{
			cublasSetStream(linalghandle, stream[i]);	
			cudaStreamSynchronize(stream[i]);
			if (!done_flag[i])
			{	      
		    	axpytemp[i] = cuCmul(make_cuDoubleComplex(-1., 0.), h_a[i][iter[i]]);
		    	cublas_status[i] = cublasZaxpy( linalghandle, dim[i], &axpytemp[i], v1[i], 1, v2[i], 1);
		    	if (cublas_status[i] != CUBLAS_STATUS_SUCCESS)
		    	{
		        	std::cout<<"Error getting v2 + d_a*v1 in "<<iter[i]<<"th iteration! Error: ";
		        	std::cout<<cublas_status[i]<<std::endl;
		    	}

		    	cublas_status[i] = cublasDznrm2( linalghandle, dim[i], v2[i], 1, &normtemp[i]);
		    	if (cublas_status[i] != CUBLAS_STATUS_SUCCESS)
		    	{
		        	std::cout<<"Error getting norm of v2 in "<<iter[i]<<"th iteration! Error: ";
		        	std::cout<<cublas_status[i]<<std::endl;
		    	}

		    	h_b[i].push_back(make_cuDoubleComplex(normtemp[i], 0.));
		    	gamma[i] = make_cuDoubleComplex(1./normtemp[i], 0.);

		    	cublas_status[i] = cublasZscal(linalghandle, dim[i], &gamma[i], v2[i], 1);
		    	if (cublas_status[i] != CUBLAS_STATUS_SUCCESS)
		    	{
		        	std::cout<<"Error getting 1/d_b * v2 in "<<iter[i]<<"th iteration! Error: ";
		        	std::cout<<cublas_status[i]<<std::endl;
		    	}
		    }
		}

		    
		for(int i = 0; i < how_many; i++)
		{
			cublasSetStream(linalghandle, stream[i]);	
			cudaStreamSynchronize(stream[i]);
			if (!done_flag[i])
			{
			
		    	status[i] = cudaMemcpyAsync(v0[i], v1[i], dim[i]*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice, stream[i]);
				if (status[i] != CUDA_SUCCESS)
    			{
        			std::cout<<"Error copying v1 to v0: "<<cudaGetErrorString(status[i])<<std::endl;
    			}
		    	status[i] = cudaMemcpyAsync(v1[i], v2[i], dim[i]*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice, stream[i]);
				if (status[i] != CUDA_SUCCESS)
    			{
        			std::cout<<"Error copying v2 to v1: "<<cudaGetErrorString(status[i])<<std::endl;
    			}
			}
		}
	
		for(int i = 0; i < how_many; i++)
		{
			if (!done_flag[i])
			{
				h_diag[i].resize(h_a[i].size());
				h_offdia[i].resize(h_b[i].size());
				

		    	for (int ii=1; ii<=iter[i]; ii++)
		    	{
					h_diag[i][ii] = h_a[i][ii].x;
					h_offdia[i][ii] = h_b[i][ii].x;
		        	h_offdia[i][ii-1] = h_offdia[i][ii];
		    	}
		    	h_offdia[i][iter[i]] = 0;
		    	returned[i] = tqli(h_diag[i], h_offdia[i], iter[i] + 1, max_Iter, h_H_eigen[i]); //tqli is in a separate file
		    	
        		sort(h_diag[i].begin(), h_diag[i].end());
				for (int j = 0; j < num_Eig; j++)
				{
					h_ordered[j] = h_diag[j];
				}         

        		gs_Energy[i] = h_ordered[i][num_Eig - 1];

				for(int j = 0; j < num_Eig; j++)
				{
				    std::cout<<std::setprecision(12)<<h_ordered[i][j]<<" ";
				}
				std::cout<<std::endl;

                                std::cout<<iter[i]<<endl;

				if (iter[i] == max_Iter - 2) // have to use this or d_b will overflow
				{
				    //this stuff here is used to resize the main arrays in the case that we aren't converging quickly enough
				    h_a.resize(2*max_Iter);
				    h_b.resize(2*max_Iter);
				    //d_diag.resize(2*max_Iter);
				    //d_offdia.resize(2*max_Iter);
				    h_diag.resize(2*max_Iter);
				    h_offdia.resize(2*max_Iter);
				    //d_lanczvec.resize(2*max_Iter*dim);
				    max_Iter *= 2;
				}
			}
		}

		for(int i = 0; i< how_many; i++){
			all_done = (all_done && done_flag[i]);
		}
    }


    /*for(int i = 0; i < num_Eig[i]; i++)
    {
        std::cout<<h_ordered[i]<<" ";
    } //write out the eigenenergies
    std::cout<<std::endl;*/
    // call the expectation values function

    // time to copy back all the eigenvectors
    //thrust::host_vector<cuDoubleComplex> h_lanczvec(max_Iter*dim);
    //h_lanczvec = d_lanczvec;

    // now the eigenvectors are available on the host CPU

    
	for(int i = 0; i < how_many; i++)
	{
    	        cudaFree(v0[i]);
    	        cudaFree(v1[i]);
    	        cudaFree(v2[i]);
		cudaFree(y[i]);
		free(h_H_eigen[i]);
		free(host_v0[i]);
		cusparseDestroyHybMat(hyb_Ham[i]);
	}

	free(gs_Energy);
	free(eigtemp);
	free(alpha);
	free(beta);
	free(returned);
	free(iter);
	free(done_flag);
	free(h_H_eigen);
	free(gamma);
	free(y);
	free(normtemp);
	free(axpytemp);
	free(dottemp);
	free(host_v0);
	
	cublas_status[0] = cublasDestroy(linalghandle);

    if (cublas_status[0] != CUBLAS_STATUS_SUCCESS)
    {
        printf("CUBLAS failed to shut down properly! \n");
    }

    cusparse_status[0] = cusparseDestroy(sparsehandle);

    if (cusparse_status[0] != CUSPARSE_STATUS_SUCCESS)
    {
        printf("CUSPARSE failed to release handle! \n");
    }


        /*if (iter == 1) {
        	std::ofstream fout;
        	fout.open("lanczos.log");
        	//fout<<normtemp<<std::endl;
        	fout<<std::endl;
        	//int* h_H_vals = (int*)malloc((dim+1)*sizeof(int));
        	cudaMemcpy(host_v0, v2, dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        	for(int i = 0; i < dim ; i++){
        		fout<<host_v0[i].x<<std::endl;
        	}

        	fout.close();
        }*/

}
// things left to do:
// write a thing (separate file) to call routines to find expectation values, should be faster on GPU
// make the tqli thing better!

__global__ void normalize(cuDoubleComplex* v, const int size, double norm)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < size)
    {
        v[i] = cuCdiv(v[i], make_cuDoubleComplex(norm, 0. ));
    }
}

int tqli(vector<double> d, vector<double> e, int n, int max_Iter, double *z)

{

    int m,l,iter,i,k;
    double s,r,p,g,f,dd,c,b;

    for (l=0; l<n; l++)
    {
        iter=0;
        do
        {
            for (m=l; m<n-1; m++)
            {
                dd=fabs(d[m])+fabs(d[m+1]);
                if (fabs(e[m])+dd == dd) break;
            }
            if (m!=l)
            {
                if (iter++ == 30)
                {
                    std::cout <<"Too many iterations in tqli() \n";
                    return 0;
                }
                g=(d[l+1]-d[l])/(2.0*e[l]);
                r=sqrt((g*g)+1.0);
                g=d[m]-d[l]+e[l]/(g+SIGN(r,g));
                s=c=1.0;
                p=0.0;
                for (i=m-1; i>=l; i--)
                {
                    f=s*e[i];
                    b=c*e[i];
                    if (fabs(f) >= fabs(g))
                    {
                        c=g/f;
                        r=sqrt((c*c)+1.0);
                        e[i + 1]=f*r;
                        c *= (s=1.0/r);
                    }
                    else
                    {
                        s=f/g;
                        r=sqrt((s*s)+1.0);
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
        }
        while (m!=l);
    }
    return 1;
}


double pythag(double a, double b)
{
    double absa, absb;
    absa=fabs(a);
    absb=fabs(b);
    if (absa > absb) return absa*sqrt(1.0+(absb/absa)*(absb/absa));
    else return (absb == 0.0 ? 0.0 : absb*sqrt(1.0+(absa/absb)*(absa/absb)));
}
