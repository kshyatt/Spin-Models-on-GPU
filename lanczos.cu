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


/*__global__ void mynorm(int dim, cuDoubleComplex* vector, float* result){

	int i = blockDim.x*blockIdx.x + threadIdx.x;
	__shared__ cuDoubleComplex current[64];
	if (i < dim){
		
		current[threadIdx.x] = vector[i];
		current[threadIdx.x] = cuCmul(current[threadIdx.x], cuConj(current[threadIdx.x]));
		atomicAdd(result, (float)current[threadIdx.x].x);
	}
}*/
	

/*__global__ void zero(cuDoubleComplex* a, int m)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if ( i < m)
    {
        a[i].y = 0;
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
__global__ void identity(double* a, int m)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i < m )
    {

        a[i + m*i] = 1.;
    }
}*/


/*Function lanczos: takes a hermitian matrix H, tridiagonalizes it, and finds the n smallest eigenvalues - this version only returns eigenvalues, not
 eigenvectors. 
---------------------------------------------------------------------------------------------------------------------------------------------------
Input: how_many, the number of Hamiltonians to process
num_Elem - the number of nonzero elements per matrix
Hamiltonian - an array of Hamiltonians, each element being a custom struct containing the rows, cols, and vals in COO format as well as the dimensions      
max_Iter, the starting number of iterations we'll try
num_Eig, the number of eigenvalues we're interested in seeing
conv_req, the convergence we'd like to see
---------------------------------------------------------------------------------------------------------------------------------------------------
Output: h_ordered, the array of the num_Eig smallest eigenvalues, ordered from smallest to largest
---------------------------------------------------------------------------------------------------------------------------------------------------
*/
__host__ void lanczos(const int how_many, const int* num_Elem, d_hamiltonian*& Hamiltonian, int max_Iter, const int num_Eig, const double conv_req)
{

    //----------Initializing CUBLAS and CUSPARSE libraries as well as storage on GPU----------------

	int* dim = (int*)malloc(how_many*sizeof(int));
	for(int i = 0; i < how_many; i++)
	{
		dim[i] = Hamiltonian[i].sectordim;
	}

	cudaStream_t stream[how_many];

    cublasStatus_t cublas_status[how_many];
    
    cublasHandle_t linalghandle;
    cublas_status[0] = cublasCreate(&linalghandle);

    if (cublas_status[0] != CUBLAS_STATUS_SUCCESS)
    {
        std::cout<<"Initializing CUBLAS failed! Error: "<<cublas_status[0]<<std::endl;
    }

    cusparseHandle_t sparsehandle;
    cusparseStatus_t cusparse_status[how_many];
    cusparse_status[0] = cusparseCreate(&sparsehandle);

    if (cusparse_status[0] != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout<<"Failed to initialize CUSPARSE! Error: "<<cusparse_status[0]<<std::endl;
    }
    
    //cout<<"Done initializing libraries"<<endl;
    cusparseMatDescr_t H_descr[how_many];
    for(int i = 0; i<how_many; i++)
	{
		cusparse_status[i] = cusparseCreateMatDescr(&H_descr[i]);

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
	//cout<<"Done creating descriptions"<<endl;
    int** d_H_rowptrs;
	d_H_rowptrs = (int**)malloc(how_many*sizeof(int*));

    for(int i = 0; i < how_many; i++)
	{
		status[i] = cudaStreamCreate(&stream[i]);
    	if (status[i] != cudaSuccess)
    	{
        	std::cout<<"Error creating streams: "<<cudaGetErrorString(status[i])<<std::endl;
    	}
	        status[i] = cudaMalloc(&d_H_rowptrs[i], (dim[i] + 1)*sizeof(int));
    	if (status[i] != cudaSuccess)
    	{
        	std::cout<<"Error allocating d_H_rowptrs: "<<cudaGetErrorString(status[i])<<std::endl;
    	}
	}

    //---------------Converting from COO to CSR format for Hamiltonians----------------    
    //cusparseHybMat_t hyb_Ham[how_many];
    for(int i = 0; i < how_many; i++)
    { 
		/*cusparse_status[i] = cusparseCreateHybMat(&hyb_Ham[i]);
		if (cusparse_status[i] != CUSPARSE_STATUS_SUCCESS)
    	{
        	std::cout<<"Error creating HYB matrix: "<<cusparse_status[i]<<std::endl;
    	}

		cout<<"Done creating HYB matrices"<<endl;*/

		cusparse_status[i] = cusparseSetStream(sparsehandle, stream[i]);
		if (cusparse_status[i] != CUSPARSE_STATUS_SUCCESS)
    	{
        	std::cout<<"Error switching streams: "<<cusparse_status[i]<<std::endl;
    	}

		
		//cout<<"Done changing streams"<<endl;

		status[i] = cudaPeekAtLastError();
		if (status[i] != cudaSuccess)
    	{
        	std::cout<<"Error synchronizing stream: "<<cudaGetErrorString(status[i])<<std::endl;
    	}
        //cout<<Hamiltonian[i].rows<<endl;
        //cout<<num_Elem[i]<<endl;
        //cout<<dim[i]<<endl;
        //cout<<d_H_rowptrs[i]<<endl; 
        cusparse_status[i] = cusparseXcoo2csr(sparsehandle, Hamiltonian[i].rows, num_Elem[i], dim[i], d_H_rowptrs[i], CUSPARSE_INDEX_BASE_ZERO);

	if (cusparse_status[i] != CUSPARSE_STATUS_SUCCESS)
    	{
        	std::cout<<"Error converting to CSR: "<<cusparse_status[i]<<std::endl;
    	}

	status[i] = cudaPeekAtLastError();
	if (status[i] != cudaSuccess)
    	{
        	std::cout<<"Error synchronizing stream: "<<cudaGetErrorString(status[i])<<std::endl;
    	}

	//cout<<"Done converting to CSR"<<endl;
    	/*cusparse_status[i] = cusparseDcsr2hyb(sparsehandle, dim[i], dim[i], H_descr[i], Hamiltonian[i].vals, d_H_rowptrs[i], Hamiltonian[i].cols, hyb_Ham[i], 0, CUSPARSE_HYB_PARTITION_AUTO);

	if (cusparse_status[i] != CUSPARSE_STATUS_SUCCESS)
    	{
        	std::cout<<"Error converting to HYB: "<<cusparse_status[i]<<std::endl;
    	}*/

	}

	status[0] = cudaPeekAtLastError();
	if (status[0] != cudaSuccess)
    	{
        	std::cout<<"Error before thread sync: "<<cudaGetErrorString(status[0])<<std::endl;
    	}

	//cout<<"Done converting to HYB"<<endl;
        /*status[0] = cudaThreadSynchronize();

	if (status[0] != cudaSuccess)
    	{
        	std::cout<<"Error syncing threads: "<<cudaGetErrorString(status[0])<<std::endl;
    	}*/
    //----------------Create three arrays to hold current Lanczos vectors----------
    vector< vector<double> > h_a(how_many);

    vector< vector<double> > h_b(how_many);
    //Making the "random" starting vector

    double** v0 = (double**)malloc(how_many*sizeof(double*));
    double** v1 = (double**)malloc(how_many*sizeof(double*));
    double** v2 = (double**)malloc(how_many*sizeof(double*));
    double*** lanczos_store = (double***)malloc(how_many*sizeof(double**));

    double** host_v0 = (double**)malloc(how_many*sizeof(double*));

    for(int i = 0; i < how_many; i++)
    {
      	status[i] = cudaMalloc(&v0[i], dim[i]*sizeof(double));
	if (status[i] != cudaSuccess)
    	{
        	std::cout<<"Error creating storage for v0 on GPU: "<<cudaGetErrorString(status[i])<<std::endl;
    	}
    	status[i] = cudaMalloc(&v1[i], dim[i]*sizeof(double));
	if (status[i] != cudaSuccess)
    	{
        	std::cout<<"Error creating storage for v1 on GPU: "<<cudaGetErrorString(status[i])<<std::endl;
    	}
    	status[i] = cudaMalloc(&v2[i], dim[i]*sizeof(double));
	if (status[i] != cudaSuccess)
    	{
        	std::cout<<"Error creating storage for v2 on GPU: "<<cudaGetErrorString(status[i])<<std::endl;
    	}
        lanczos_store[i] = (double**)malloc(max_Iter*sizeof(double*));
		host_v0[i] = (double*)malloc(dim[i]*sizeof(double));
    	
    	
    	for(int j = 0; j<dim[i]; j++)
    	{
        	host_v0[i][j] = 0.;
        	if (j%4 == 0) host_v0[i][j] = 1. ;
        	else if (j%5 == 0) host_v0[i][j] = -2.;
        	else if (j%7 == 0) host_v0[i][j] = 3.;
        	else if (j%9 == 0) host_v0[i][j] = -4.;

	    }

        status[i] = cudaMalloc(&lanczos_store[i][0], dim[i]*sizeof(double));
    	
		if (status[i] != cudaSuccess)
    	{
        	std::cout<<"Error creating storage for v0 in lanczos_store: "<<cudaGetErrorString(status[i])<<std::endl;
    	}
        
        status[i] = cudaMemcpyAsync(v0[i], host_v0[i], dim[i]*sizeof(double), cudaMemcpyHostToDevice, stream[i]);
		if (status[i] != cudaSuccess)
    	{
        	std::cout<<"Error copying v0 to the device: "<<cudaGetErrorString(status[i])<<std::endl;
    	}
	}
    //cout<<"Done creating and copying v0"<<endl;

    //--------------Create dummy variables for CUBLAS functions----------------

    double* normtemp = (double*)malloc(how_many*sizeof(double));
	double* alpha = (double*)malloc(how_many*sizeof(double));
	double* beta = (double*)malloc(how_many*sizeof(double));

	double* dottemp = (double*)malloc(how_many*sizeof(double));
	double* axpytemp = (double*)malloc(how_many*sizeof(double));

    //--------------Generate first Lanczos vector--------------------------

    for(int i = 0; i < how_many; i++)
	{
		cublasSetStream(linalghandle, stream[i]);
		cusparseSetStream(sparsehandle, stream[i]);
		cublas_status[i] = cublasDnrm2(linalghandle, dim[i], v0[i], 1, &normtemp[i]);
		
		normtemp[i] = 1./normtemp[i];
		
		cublas_status[i] = cublasDscal(linalghandle, dim[i], &normtemp[i], v0[i], 1);

    	alpha[i] = 1.;
    	beta[i] = 0.;

    	//cudaThreadSynchronize();
    	cudaMemcpyAsync(lanczos_store[i][0], v0[i], dim[i]*sizeof(double), cudaMemcpyDeviceToDevice, stream[i]);
    	cusparse_status[i] = cusparseDcsrmv(sparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE, dim[i], dim[i], num_Elem[i], &alpha[i], H_descr[i], Hamiltonian[i].vals, d_H_rowptrs[i], Hamiltonian[i].cols, v0[i], &beta[i], v1[i]); // the Hamiltonian is applied here


        	/*std::ofstream fout;
        	fout.open("lanczos.log");
        	cudaMemcpy(host_v0[0], v1[0], dim[0]*sizeof(double), cudaMemcpyDeviceToHost);
        	for(int j = 0; j < dim[0] ; j++){
        		fout<<(host_v0[0][j])<<std::endl;
				
        	}

        	fout.close();*/


		//cout<<"Done getting v1 = H*v0"<<endl;
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

    	//------Set up arrays for iteration over many Lanczos steps-------------------

	}
    
	double** y;
	y = (double**)malloc(how_many*sizeof(double*));

	for(int i = 0; i < how_many; i++)
	{
		cublasSetStream(linalghandle, stream[i]);
		dottemp[i] = 0.;
	   	cublas_status[i] = cublasDdot(linalghandle, dim[i], v1[i], 1, v0[i], 1, &dottemp[i]);
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

    	h_b[i].push_back(0.);

    	

    	if (status[i] != cudaSuccess)
    	{
        	std::cout<<"Memory allocation of y dummy vector failed! Error:";
        	std::cout<<cudaGetErrorString( status[i] )<<std::endl;
    	}

		status[i] = cudaMalloc(&y[i], dim[i]*sizeof(double));

    	cublas_status[i] = cublasDscal(linalghandle, dim[i], &beta[i], y[i], 1);
    	cudaStreamSynchronize(stream[i]);

    	axpytemp[i] = -1*h_a[i][0];

    	cublas_status[i] = cublasDaxpy(linalghandle, 0, &axpytemp[i], v0[i], 1, v1[i], 1);
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

		normtemp[i] = 0.;
		//cout<<&normtemp[i]<<endl;
		//cuDoubleComplex testnorm;		
		//cublas_status[i] = cublasZdotc(linalghandle, dim[i], v1[i], 1, v1[i], 1, &testnorm);
		//zero<<<(dim[i]/512 + 1)*512, 512, 0, stream[i]>>>(v1[i], dim[i]);
		//cudaThreadSynchronize();
    	cublas_status[i] = cublasDnrm2(linalghandle, dim[i], v1[i], 1, &normtemp[i]); //this is slow for some reason
		//mynorm<<<((dim[i]/64 + 1)*64), 64, 0, stream[i]>>>(dim[i], v1[i], dev_norm);

    	cudaStreamSynchronize(stream[i]);
    	if (cublas_status[i] != CUBLAS_STATUS_SUCCESS)
    	{
        	std::cout<<"Getting the norm of v1 failed! Error: ";
        	std::cout<<cublas_status[i]<<std::endl;
    	}
		//normtemp[i] = testnorm.x;
		//std::cout<<normtemp[i]<<std::endl;

    	if (cudaPeekAtLastError() != 0 )
    	{
        	std::cout<<"Getting nrm(V1) failed! Error: ";
        	std::cout<<cudaGetErrorString(cudaPeekAtLastError())<<std::endl;
    	}

	}

	double* gamma = (double*)malloc(how_many*sizeof(double));

	for(int i = 0; i < how_many; i++)
	{
		cublasSetStream(linalghandle, stream[i]);
    	//d_b_ptr = thrust::raw_pointer_cast(&d_b[1]);

    	h_b[i].push_back(normtemp[i]);
    	
    	normtemp[i] = 1./normtemp[i];
    	gamma[i] = 1./h_b[i][1]; //alpha = 1/beta in v1 = v1 - alpha*v0
    	//normalize<<<dim/512 + 1, 512>>>(v0, dim, normtemp);
    	cublas_status[i] = cublasDscal(linalghandle, dim[i], &normtemp[i], v1[i], 1);
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
	cudaMalloc(&lanczos_store[i][1], dim[i]*sizeof(double));
        cudaMemcpyAsync(lanczos_store[i][1], v1[i], dim[i]*sizeof(double), cudaMemcpyDeviceToDevice, stream[i]);
        }


	//cout<<"Done first round"<<endl;
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
    /*thrust::device_vector<double> d_diag(max_Iter);
    double* diag_ptr;
    thrust::device_vector<double> d_offdia(max_Iter);
    double* offdia_ptr;*/

	//vector< vector< double > > h_diag(how_many);
	//vector< vector< double > > h_offdia(how_many);

	double** h_diag = (double**)malloc(how_many*sizeof(double*));
	double** h_offdia = (double**)malloc(how_many*sizeof(double*));
    //thrust::host_vector<double> h_diag(max_Iter);
    //double* h_diag_ptr = raw_pointer_cast(&h_diag[0]);
    //thrust::host_vector<double> h_offdia(max_Iter);
    //double* h_offdia_ptr = raw_pointer_cast(&h_offdia[0]);

	vector< vector < double > > h_ordered(how_many);

	for(int i = 0; i<how_many; i++){
		gs_Energy[i] = 1.;
		eigtemp[i] = 0.;
		iter[i] = 0;
		done_flag[i] = false;
		h_ordered[i].resize(num_Eig, 0);
		h_H_eigen[i] = (double*)malloc(max_Iter*max_Iter*sizeof(double));
		h_diag[i] = (double*)malloc(h_a[i].size()*sizeof(double));
		h_offdia[i] = (double*)malloc(h_b[i].size()*sizeof(double));
	}
	//cout<<"Initialized everything for iterations"<<endl;

    //---------Begin Lanczos iteration-----------------------------

    bool all_done = false;
    
	while( !all_done )
	{
		all_done = true;
		//cout<<eigtemp[0]<<endl;
		for(int i = 0; i < how_many; i++)
		{
			cublasSetStream(linalghandle, stream[i]);
			cusparseSetStream(sparsehandle, stream[i]);
			cudaStreamSynchronize(stream[i]);
			//cout<<"Streams set"<<endl;
			//fflush(stdout);
			
			if (!done_flag[i])
			{
				iter[i]++;
				cusparse_status[i] = cusparseDcsrmv(sparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE, dim[i], dim[i], num_Elem[i], &alpha[i], H_descr[i], Hamiltonian[i].vals, d_H_rowptrs[i], Hamiltonian[i].cols, v1[i], &beta[i], v2[i]);
				if( cusparse_status[i] != 0)
                                {
                                    cout<<"Error applying H to V1 in "<<iter[i]<<"th iteration"<<endl;
                                }
				//cusparse_status[i] = cusparseDhybmv(sparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha[i], H_descr[i], hyb_Ham[i], v1[i], &beta[i], v2[i]); // the Hamiltonian is applied here, in this gross expression
			}
		}
		//cout<<"Got v2 for "<<iter[0]<<"th iteration"<<endl;

		/*if (iter[0] == 1) {
        	std::ofstream fout;
        	fout.open("lanczos.log");
        	cudaMemcpy(host_v0[0], v2[0], dim[0]*sizeof(double), cudaMemcpyDeviceToHost);
        	for(int j = 0; j < dim[0] ; j++){
        		fout<<(host_v0[0][j])<<std::endl;
				
        	}

        	fout.close();
        }*/
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

		    	cublas_status[i] = cublasDdot(linalghandle, dim[i], v1[i], 1, v2[i], 1, &dottemp[i]);
		    	cudaStreamSynchronize(stream[i]);
				//cout<<"Got v1*v2 for "<<iter[i]<<"th iteration"<<endl;
		    	h_a[i].push_back(dottemp[i]);
				//cout<<dottemp[i]<<endl;

				if (cublas_status[i] != CUBLAS_STATUS_SUCCESS)
		    	{
		        	std::cout<<"Error getting v1 * v2 in "<<iter[i]<<"th iteration! Error: ";
		        	std::cout<<cublas_status[i]<<std::endl;
		    	}

		    	//cudaMemcpy(temp_ptr, v1, dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
		    	//temp = v1;

		    	axpytemp[i] = -1.*h_b[i][iter[i]];

		    	cublas_status[i] = cublasDaxpy( linalghandle, dim[i], &axpytemp[i], v0[i], 1, v2[i], 1);
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
		    	axpytemp[i] = -1.*h_a[i][iter[i]];
		    	cublas_status[i] = cublasDaxpy( linalghandle, dim[i], &axpytemp[i], v1[i], 1, v2[i], 1);
		    	if (cublas_status[i] != CUBLAS_STATUS_SUCCESS)
		    	{
		        	std::cout<<"Error getting v2 + d_a*v1 in "<<iter[i]<<"th iteration! Error: ";
		        	std::cout<<cublas_status[i]<<std::endl;
		    	}

		    	cublas_status[i] = cublasDnrm2( linalghandle, dim[i], v2[i], 1, &normtemp[i]);
		    	if (cublas_status[i] != CUBLAS_STATUS_SUCCESS)
		    	{
		        	std::cout<<"Error getting norm of v2 in "<<iter[i]<<"th iteration! Error: ";
		        	std::cout<<cublas_status[i]<<std::endl;
		    	}
			//cout<<"Got norm of v2 in "<<iter[i]<<"th iteration"<<endl;

		    	h_b[i].push_back(normtemp[i]);
		    	gamma[i] = 1./normtemp[i];

		    	cublas_status[i] = cublasDscal(linalghandle, dim[i], &gamma[i], v2[i], 1);
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
			status[i] = cudaStreamSynchronize(stream[i]);
			
			if (status[i] != cudaSuccess)
    			{
        			std::cout<<"Error syncing before copying v1 to v0: "<<cudaGetErrorString(status[i])<<std::endl;
    			}
                        
                        if (!done_flag[i])
			{
			//----------------Reorthogonalize the new Lanczos vector-----------------------
		    	for(int j = 0; j < iter[i] + 1; j++){
                            cublasDdot(linalghandle, dim[i], v2[i], 1, lanczos_store[i][j], 1, &dottemp[i]);
                            dottemp[i] *= -1.;
                            cublasDaxpy(linalghandle, dim[i],  &dottemp[i], lanczos_store[i][j], 1, v2[i], 1);
                            dottemp[i] = 1. - dottemp[i]*dottemp[i];
                            cublasDscal(linalghandle, dim[i], &dottemp[i], v2[i], 1);
                        
                        }                 
                        
                        //--------------Copy the Lanczos vectors down to prepare for next iteration--------------
                        status[i] = cudaMemcpyAsync(v0[i], v1[i], dim[i]*sizeof(double), cudaMemcpyDeviceToDevice, stream[i]);
			if (status[i] != cudaSuccess)
    			{
        			std::cout<<"Error copying v1 to v0: "<<cudaGetErrorString(status[i])<<std::endl;
    			}
		    	status[i] = cudaMemcpyAsync(v1[i], v2[i], dim[i]*sizeof(double), cudaMemcpyDeviceToDevice, stream[i]);
			if (status[i] != cudaSuccess)
    			{
        			std::cout<<"Error copying v2 to v1: "<<cudaGetErrorString(status[i])<<std::endl;
    			}
			//cout<<"Done copying vectors in "<<iter[i]<<"th iteration"<<endl;
                        status[i] = cudaMalloc(&lanczos_store[i][iter[i] + 1], dim[i]*sizeof(double));
                        status[i] = cudaMemcpyAsync(lanczos_store[i][iter[i] + 1], v2[i], dim[i]*sizeof(double), cudaMemcpyDeviceToDevice, stream[i]);
			}
		
                }
	
		for(int i = 0; i < how_many; i++)
		{
			if (!done_flag[i] && iter[i] > 5)
			{
				//h_diag[i].resize(h_a[i].size());
				//h_offdia[i].resize(h_b[i].size());
				free(h_diag[i]);
				free(h_offdia[i]);
				h_diag[i] = (double*)malloc(h_a[i].size()*sizeof(double));
				h_offdia[i] = (double*)malloc(h_b[i].size()*sizeof(double));
			        //cout<<"Done resizing h_diag and h_offida in "<<iter[i]<<"th iteration"<<endl;
				h_diag[i][0] = h_a[i][0];
		    	        for (int ii=1; ii<=iter[i]; ii++)
		    	        {
					h_diag[i][ii] = h_a[i][ii];
					h_offdia[i][ii] = h_b[i][ii];
		        	        h_offdia[i][ii-1] = h_offdia[i][ii];
		    	        }
		    	        h_offdia[i][iter[i]] = 0;
                                /*zero<<<(max_Iter*max_Iter/512 + 1), 512, 0, stream[i]>>>(h_H_eigen[i], max_Iter*max_Iter);
                                status[i] = cudaStreamSynchronize(stream[i]);
                        	
                                status[i] = cudaPeekAtLastError();
                                if( status[i] != cudaSuccess)
                                {
                                    cout<<"Error in stream sync! Error: "<<cudaGetErrorString(status[i])<<endl;
                                }
                                
                        	status[i] = cudaPeekAtLastError();
                                if( status[i] != cudaSuccess)
                                {
                                    cout<<"Error in zero! Error: "<<cudaGetErrorString(status[i])<<endl;
                                }
>>>>>>> temp
                                identity<<<(max_Iter/512 + 1), 512, 0, stream[i]>>>(h_H_eigen[i], max_Iter);*/
                                cudaStreamSynchronize(stream[i]);

                                //---------Diagonalize Lanczos matrix and check for convergence------------------
                                returned[i] = tqli(h_diag[i], h_offdia[i], iter[i] + 1, max_Iter, h_H_eigen[i]); 
                        	status[i] = cudaPeekAtLastError();
                                if( status[i] != cudaSuccess)
                                {
                                    cout<<"Error in identity! Error: "<<cudaGetErrorString(status[i])<<endl;
                                }
                                //cout<<"Done tqli in "<<iter[i]<<"th iteration"<<endl;
        		
			        sort(h_diag[i], h_diag[i] + h_a[i].size());
			        for (int j = 0; j < num_Eig; j++)
			        {
				h_ordered[i][j] = h_diag[i][j];
                                //cout<<h_ordered[i][j]<<" ";
				}
                                //cout<<endl;

				gs_Energy[i] = h_ordered[i][num_Eig - 1];
				done_flag[i] = (fabs(gs_Energy[i] - eigtemp[i]) < conv_req);// && iter[i] > 10;// ? (iter[i] > 10) : false;
				//cout<<"Got done flag"<<endl;
				eigtemp[i] = h_ordered[i][num_Eig - 1];

				if (iter[i] == max_Iter - 2) // have to use this or d_b will overflow
				{
				    //this stuff here is used to resize the main arrays in the case that we aren't converging quickly enough
				    h_a[i].resize(2*max_Iter);
				    h_b[i].resize(2*max_Iter);
				    //d_diag.resize(2*max_Iter);
				    //d_offdia.resize(2*max_Iter);
				    //h_diag[i].resize(2*max_Iter);
				    //h_offdia[i].resize(2*max_Iter);
				    //d_lanczvec.resize(2*max_Iter*dim);
				    max_Iter *= 2;
				}
			}
		}
		all_done = true;
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

   //--------------Free arrays to prevent memory leaks------------------------ 
	for(int i = 0; i < how_many; i++)
	{
		for(int j = 0; j < num_Eig; j++)
			{
			    std::cout<<std::setprecision(12)<<h_ordered[i][j]<<" ";
			}
		std::cout<<std::endl;

                for(int j = 0; j < iter[i]; j++)
                {
                    cudaFree(lanczos_store[i][j]);
                }
                free(lanczos_store[i]);
                cudaFree(d_H_rowptrs[i]);
                cudaFree(v0[i]);
		cudaFree(v1[i]);
		cudaFree(v2[i]);
		cudaFree(y[i]);
		free(h_H_eigen[i]);
		free(host_v0[i]);
		free(h_diag[i]);
		free(h_offdia[i]);
		//cusparseDestroyHybMat(hyb_Ham[i]);
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
	free(v0);
	free(v1);
	free(v2);
	free(h_diag);
	free(h_offdia);
        free(lanczos_store);
        free(dim);
        free(d_H_rowptrs);
        	
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

int tqli(double* d, double* e, int n, int max_Iter, double *z)

{

    int m,l,iter,i;//k;
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
                if (iter++ == 60)
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
