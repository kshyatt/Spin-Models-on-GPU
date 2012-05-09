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
__host__ void lanczos(const int how_many, const int* num_Elem, d_hamiltonian*& Hamiltonian, double**& groundstates, int max_Iter, const int num_Eig, const double conv_req)
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



        status[i] = cudaPeekAtLastError();
        if (status[i] != cudaSuccess)
        {
            std::cout<<"Error synchronizing stream: "<<cudaGetErrorString(status[i])<<std::endl;
        }
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
    //----------------Create arrays to hold current Lanczos vectors----------
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

        cudaMemcpyAsync(lanczos_store[i][0], v0[i], dim[i]*sizeof(double), cudaMemcpyDeviceToDevice, stream[i]);

        //-----------Apply Hamiltonian to V0--------------------
        cusparse_status[i] = cusparseDcsrmv(sparsehandle, CUSPARSE_OPERATION_NON_TRANSPOSE, dim[i], dim[i], num_Elem[i], &alpha[i], H_descr[i], Hamiltonian[i].vals, d_H_rowptrs[i], Hamiltonian[i].cols, v0[i], &beta[i], v1[i]); // the Hamiltonian is applied here

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

        //---------Normalize V1 and copy it to Lanczos storage-----------

        normtemp[i] = 0.;
        cublas_status[i] = cublasDnrm2(linalghandle, dim[i], v1[i], 1, &normtemp[i]); //this is slow for some reason

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

    double* gamma = (double*)malloc(how_many*sizeof(double));

    for(int i = 0; i < how_many; i++)
    {
        cublasSetStream(linalghandle, stream[i]);

        h_b[i].push_back(normtemp[i]);

        normtemp[i] = 1./normtemp[i];
        gamma[i] = 1./h_b[i][1]; //alpha = 1/beta in v1 = v1 - alpha*v0
        cublas_status[i] = cublasDscal(linalghandle, dim[i], &normtemp[i], v1[i], 1);


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

    //------Set up storage needed for Lanczos iteration--------

    double* gs_Energy = (double*)malloc(how_many*sizeof(double));

    double* eigtemp = (double*)malloc(how_many*sizeof(double));

    int* returned = (int*)malloc(how_many*sizeof(int));

    int* iter = (int*)malloc(how_many*sizeof(int));

    bool* done_flag = (bool*)malloc(how_many*sizeof(bool));

    double** h_H_eigen = (double**)malloc(how_many*sizeof(double*));
    double** d_H_eigen = (double**)malloc(how_many*sizeof(double*));

    double** h_diag = (double**)malloc(how_many*sizeof(double*));
    double** h_offdia = (double**)malloc(how_many*sizeof(double*));

    vector< vector < double > > h_ordered(how_many);

    //--------Set up quantities for Lanczos iteration---------
    for(int i = 0; i<how_many; i++)
    {
        gs_Energy[i] = 1.;
        eigtemp[i] = 0.;
        iter[i] = 0;
        done_flag[i] = false;
        h_ordered[i].resize(num_Eig, 0);
        h_H_eigen[i] = (double*)malloc(max_Iter*max_Iter*sizeof(double));
        cudaMalloc(&d_H_eigen[i], max_Iter*max_Iter*sizeof(double));
        h_diag[i] = (double*)malloc(h_a[i].size()*sizeof(double));
        h_offdia[i] = (double*)malloc(h_b[i].size()*sizeof(double));
    }

    //---------Begin Lanczos iteration-----------------------------

    bool all_done = false;

    while( !all_done )
    {
        all_done = true;
        for(int i = 0; i < how_many; i++)
        {
            cublasSetStream(linalghandle, stream[i]);
            cusparseSetStream(sparsehandle, stream[i]);
            cudaStreamSynchronize(stream[i]);

            //----------Apply the Hamiltonian to the Lanczos vectors-----
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

        //--------Find next set of elements in Lanczos Hamiltonian----

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
                for(int j = 0; j < iter[i] + 1; j++)
                {
                    cublasDdot(linalghandle, dim[i], v2[i], 1, lanczos_store[i][j], 1, &dottemp[i]);
                    dottemp[i] *= -1.;
                    cublasDaxpy(linalghandle, dim[i],  &dottemp[i], lanczos_store[i][j], 1, v2[i], 1);
                    dottemp[i] = 1. - dottemp[i]*dottemp[i];
                    cublasDscal(linalghandle, dim[i], &dottemp[i], v2[i], 1);

                }

                //--------------Copy the Lanczos vectors to storage to prepare for next iteration--------------
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
                //---Copy Lanczos matrix information for diagonalization-----
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

                cudaStreamSynchronize(stream[i]);

                //---------Diagonalize Lanczos matrix and check for convergence------------------
                returned[i] = tqli(h_diag[i], h_offdia[i], iter[i] + 1, max_Iter, h_H_eigen[i]);
                status[i] = cudaPeekAtLastError();
                if( status[i] != cudaSuccess)
                {
                    cout<<"Error in identity! Error: "<<cudaGetErrorString(status[i])<<endl;
                }
                //cout<<"Done tqli in "<<iter[i]<<"th iteration"<<endl;
                cudaMemcpyAsync(d_H_eigen[i], h_H_eigen[i], max_Iter*max_Iter*sizeof(double), cudaMemcpyHostToDevice, stream[i]);

                sort(h_diag[i], h_diag[i] + h_a[i].size());
                for (int j = 0; j < num_Eig; j++)
                {
                    h_ordered[i][j] = h_diag[i][j];
                    //cout<<h_ordered[i][j]<<" ";
                }
                //cout<<endl;

                gs_Energy[i] = h_ordered[i][num_Eig - 1];
                done_flag[i] = (fabs(gs_Energy[i] - eigtemp[i]) < conv_req);// && iter[i] > 10;// ? (iter[i] > 10) : false;
                //done_flag[i] = iter[i] == max_Iter - 2;
                eigtemp[i] = h_ordered[i][num_Eig - 1];

                if (iter[i] == max_Iter - 2) // have to use this or d_b will overflow
                {
                    //this stuff here is used to resize the main arrays in the case that we aren't converging quickly enough
                    h_a[i].resize(2*max_Iter);
                    h_b[i].resize(2*max_Iter);
                    max_Iter *= 2;
                }
            }
        }
        all_done = true;
        for(int i = 0; i< how_many; i++)
        {
            all_done = (all_done && done_flag[i]);
        }
    }


    //-------------Get groundstates------------------------------------------

    for( int i = 0; i < how_many; i++)
    {
        cudaStreamSynchronize(stream[i]);
        GetGroundstate<<<dim[i]/512 + 1, 512, 0, stream[i]>>>(groundstates[i], lanczos_store[i], d_H_eigen[i], iter[i], dim[i]);
    }

    //--------------Free arrays to prevent memory leaks------------------------
    for(int i = 0; i < how_many; i++)
    {
        for(int j = 0; j < num_Eig; j++)
        {
            std::cout<<std::setprecision(12)<<h_ordered[i][j]/16<<" ";
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
        cudaFree(d_H_eigen[i]);
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
    free(d_H_eigen);
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
    //free(dim);
    free(d_H_rowptrs);

    cublas_status[0] = cublasDestroy(linalghandle);

    //----------Output groundstate to file to check for correctness------

    double* host_groundstate = (double*)malloc(dim[0]*sizeof(double));
    std::ofstream fout;
    fout.open("lanczos.log");
    cudaMemcpy(host_groundstate, groundstates[0], dim[0]*sizeof(double), cudaMemcpyDeviceToHost);
    for(int i = 0; i < dim[0] ; i++)
    {
        fout<<host_groundstate[i]<<std::endl;
    }

    fout.close();
    free(host_groundstate);
    free(dim);
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

int tqli(double* d, double* e, int n, int max_Iter, double *z)

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


                    for (k=0; k<n; k++)
                    {
                        f=z[k * n + i+1];
                        z[k*n + i+1]=s*z[k*n + i]+c*f;
                        z[k*n + i ]=c*z[k*n+i]-s*f;
                    }

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

__global__ void GetGroundstate(double* groundstates, double** lanczos_store, double* H_eigen, int mat_dim, int vec_dim)
{

    int element = blockIdx.x*blockDim.x + threadIdx.x;

    if ( element < vec_dim )
    {
        groundstates[element] = H_eigen[0]*lanczos_store[0][element];
        for (int lanc_iter = 1; lanc_iter < mat_dim; lanc_iter++)
        {
            groundstates[element] += H_eigen[lanc_iter]*lanczos_store[lanc_iter][element];
        }
    }
};
