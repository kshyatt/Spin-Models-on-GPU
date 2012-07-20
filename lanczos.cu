/*!
    \file lanczos.cu
    \brief Controller code for general Lanczos diagonalization
*/

// Katharine Hyatt
// A set of functions to implement the Lanczos method for a generic Hamiltonian
// Based on the codes Lanczos_07.cpp and Lanczos07.h by Roger Melko
//-------------------------------------------------------------------------------

#include"lanczos.h"

/*Function lanczos: takes a hermitian matrix H, tridiagonalizes it, and finds the n smallest eigenvalues - this version only returns eigenvalues, not
 eigenvectors.
---------------------------------------------------------------------------------------------------------------------------------------------------
Input: howMany, the number of Hamiltonians to process
numElem - the number of nonzero elements per matrix
Hamiltonian - an array of Hamiltonians, each element being a custom struct containing the rows, cols, and vals in COO format as well as the dimensions
maxIter, the starting number of iterations we'll try
numEig, the number of eigenvalues we're interested in seeing
convReq, the convergence we'd like to see
---------------------------------------------------------------------------------------------------------------------------------------------------
Output: h_ordered, the array of the numEig smallest eigenvalues, ordered from smallest to largest
---------------------------------------------------------------------------------------------------------------------------------------------------
*/
__host__ void lanczos(const int howMany, const int* numElem, d_hamiltonian*& Hamiltonian, double**& groundstates, double**& eigenvalues, int maxIter, const int numEig, const double convReq)
{

    //----------Initializing CUBLAS and CUSPARSE libraries as well as storage on GPU----------------

    int* dim = (int*)malloc(howMany*sizeof(int));
    for(int i = 0; i < howMany; i++)
    {
        dim[i] = Hamiltonian[i].sectorDim;
    }

    /*! 

    First it is necessary to create handles, streams, and to initialize the two CUDA libraries which will be used:
	\verbatim
    */
    cudaStream_t stream[howMany];

    cublasStatus_t cublasStatus[howMany];

    cublasHandle_t linAlgHandle;
    cublasStatus[0] = cublasCreate(&linAlgHandle);

    if (cublasStatus[0] != CUBLAS_STATUS_SUCCESS)
    {
        std::cout<<"Initializing CUBLAS failed! Error: "<<cublasStatus[0]<<std::endl;
    }

    cusparseHandle_t sparseHandle;
    cusparseStatus_t cusparseStatus[howMany];
    cusparseStatus[0] = cusparseCreate(&sparseHandle);

    if (cusparseStatus[0] != CUSPARSE_STATUS_SUCCESS)
    {
        std::cout<<"Failed to initialize CUSPARSE! Error: "<<cusparseStatus[0]<<std::endl;
    }
    /*!
    \endverbatim
    
    The function also transforms the Hamiltonian into CSR format so that CUSPARSE can use it for matrix-vector multiplications.
    \verbatim
    */
    cusparseMatDescr_t H_descr[howMany];
    for(int i = 0; i<howMany; i++)
    {
        cusparseStatus[i] = cusparseCreateMatDescr(&H_descr[i]);

        if (cusparseStatus[i] != CUSPARSE_STATUS_SUCCESS)
        {
            std::cout<<"Error creating matrix description: "<<cusparseStatus[i]<<std::endl;
        }
        cusparseStatus[i] = cusparseSetMatType(H_descr[i], CUSPARSE_MATRIX_TYPE_GENERAL);
        if (cusparseStatus[i] != CUSPARSE_STATUS_SUCCESS)
        {
            std::cout<<"Error setting matrix type: "<<cusparseStatus[i]<<std::endl;
        }
        cusparseStatus[i] = cusparseSetMatIndexBase(H_descr[i], CUSPARSE_INDEX_BASE_ZERO);
        if (cusparseStatus[i] != CUSPARSE_STATUS_SUCCESS)
        {
            std::cout<<"Error setting matrix index base: "<<cusparseStatus[i]<<std::endl;
        }

    }
    cudaError_t status[howMany];
    int** d_H_rowPtrs;
    d_H_rowPtrs = (int**)malloc(howMany*sizeof(int*));

    for(int i = 0; i < howMany; i++)
    {
        status[i] = cudaStreamCreate(&stream[i]);
        if (status[i] != cudaSuccess)
        {
            std::cout<<"Error creating streams: "<<cudaGetErrorString(status[i])<<std::endl;
        }
        status[i] = cudaMalloc(&d_H_rowPtrs[i], (dim[i] + 1)*sizeof(int));
        if (status[i] != cudaSuccess)
        {
            std::cout<<"Error allocating d_H_rowPtrs: "<<cudaGetErrorString(status[i])<<std::endl;
        }
    }

    //---------------Converting from COO to CSR format for Hamiltonians----------------
    //cusparseHybMat_t hyb_Ham[howMany];
    for(int i = 0; i < howMany; i++)
    {
        /*cusparseStatus[i] = cusparseCreateHybMat(&hyb_Ham[i]);
        if (cusparseStatus[i] != cusparseStatus_SUCCESS)
        {
        	std::cout<<"Error creating HYB matrix: "<<cusparseStatus[i]<<std::endl;
        }

        cout<<"Done creating HYB matrices"<<endl;*/

        cusparseStatus[i] = cusparseSetStream(sparseHandle, stream[i]);
        if (cusparseStatus[i] != CUSPARSE_STATUS_SUCCESS)
        {
            std::cout<<"Error switching streams: "<<cusparseStatus[i]<<std::endl;
        }



        status[i] = cudaPeekAtLastError();
        if (status[i] != cudaSuccess)
        {
            std::cout<<"Error synchronizing stream: "<<cudaGetErrorString(status[i])<<std::endl;
        }
        cusparseStatus[i] = cusparseXcoo2csr(sparseHandle, Hamiltonian[i].rows, numElem[i], dim[i], d_H_rowPtrs[i], CUSPARSE_INDEX_BASE_ZERO);

        if (cusparseStatus[i] != CUSPARSE_STATUS_SUCCESS)
        {
            std::cout<<"Error converting to CSR: "<<cusparseStatus[i]<<std::endl;
        }

        status[i] = cudaPeekAtLastError();
        if (status[i] != cudaSuccess)
        {
            std::cout<<"Error synchronizing stream: "<<cudaGetErrorString(status[i])<<std::endl;
        }

        /*cusparseStatus[i] = cusparseDcsr2hyb(sparseHandle, dim[i], dim[i], H_descr[i], Hamiltonian[i].vals, d_H_rowPtrs[i], Hamiltonian[i].cols, hyb_Ham[i], 0, CUSPARSE_HYB_PARTITION_AUTO);

        	if (cusparseStatus[i] != cusparseStatus_SUCCESS)
        {
        	std::cout<<"Error converting to HYB: "<<cusparseStatus[i]<<std::endl;
        }*/

    }
    /*!
    \endverbatim
    status[0] = cudaPeekAtLastError();
    if (status[0] != cudaSuccess)
    {
        std::cout<<"Error before thread sync: "<<cudaGetErrorString(status[0])<<std::endl;
    }
    */
    //----------------Create arrays to hold current Lanczos vectors----------
    vector< vector<double> > h_a(howMany);

    vector< vector<double> > h_b(howMany);
    //Making the "random" starting vector

    /*! 

    The function then sets up Lanczos diagonalization by initializing a random starting vector on the CPU, creating storage for the Lanczos vectors on the GPU, and copying this starting vector across.
    
    \verbatim
    */
    double** v0 = (double**)malloc(howMany*sizeof(double*));
    double** v1 = (double**)malloc(howMany*sizeof(double*));
    double** v2 = (double**)malloc(howMany*sizeof(double*));
    double*** lanczosStore = (double***)malloc(howMany*sizeof(double**));

    double** host_v0 = (double**)malloc(howMany*sizeof(double*));

    for(int i = 0; i < howMany; i++)
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
        lanczosStore[i] = (double**)malloc(maxIter*sizeof(double*));
        host_v0[i] = (double*)malloc(dim[i]*sizeof(double));


        for(int j = 0; j<dim[i]; j++)
        {
            host_v0[i][j] = 0.;
            if (j%4 == 0) host_v0[i][j] = 1. ;
            else if (j%5 == 0) host_v0[i][j] = -2.;
            else if (j%7 == 0) host_v0[i][j] = 3.;
            else if (j%9 == 0) host_v0[i][j] = -4.;

        }

        status[i] = cudaMalloc(&lanczosStore[i][0], dim[i]*sizeof(double));

        if (status[i] != cudaSuccess)
        {
            std::cout<<"Error creating storage for v0 in lanczosStore: "<<cudaGetErrorString(status[i])<<std::endl;
        }

        status[i] = cudaMemcpyAsync(v0[i], host_v0[i], dim[i]*sizeof(double), cudaMemcpyHostToDevice, stream[i]);
        if (status[i] != cudaSuccess)
        {
            std::cout<<"Error copying v0 to the device: "<<cudaGetErrorString(status[i])<<std::endl;
        }
    }
    /*!
    \endverbatim
    First, storage variables are created to hold the results of the CUBLAS functions.
    \verbatim
    */

    //--------------Create dummy variables for CUBLAS functions----------------

    double* normTemp = (double*)malloc(howMany*sizeof(double));
    double* alpha = (double*)malloc(howMany*sizeof(double));
    double* beta = (double*)malloc(howMany*sizeof(double));

    double* dotTemp = (double*)malloc(howMany*sizeof(double));
    double* axpyTemp = (double*)malloc(howMany*sizeof(double));
    
    double** y = (double**)malloc(howMany*sizeof(double*));
    /*!
    \endverbatim
    */
    //--------------Generate first Lanczos vector--------------------------

    for(int i = 0; i < howMany; i++)
    {
        cublasSetStream(linAlgHandle, stream[i]);
        cusparseSetStream(sparseHandle, stream[i]);
        /*! 
        Then the initial multiplication to generate the first Lanczos vector is performed. 
        \verbatim
        */     
       cublasStatus[i] = cublasDnrm2(linAlgHandle, dim[i], v0[i], 1, &normTemp[i]);

        normTemp[i] = 1./normTemp[i];

        cublasStatus[i] = cublasDscal(linAlgHandle, dim[i], &normTemp[i], v0[i], 1);

        alpha[i] = 1.;
        beta[i] = 0.;

        cudaMemcpyAsync(lanczosStore[i][0], v0[i], dim[i]*sizeof(double), cudaMemcpyDeviceToDevice, stream[i]);

        //-----------Apply Hamiltonian to V0--------------------
        cusparseStatus[i] = cusparseDcsrmv(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, dim[i], dim[i], numElem[i], &alpha[i], H_descr[i], Hamiltonian[i].vals, d_H_rowPtrs[i], Hamiltonian[i].cols, v0[i], &beta[i], v1[i]); // the Hamiltonian is applied here

        /*!
        \endverbatim
        */

        if (cusparseStatus[i] != CUSPARSE_STATUS_SUCCESS)
        {
            std::cout<<"Getting V1 = H*V0 failed! Error: ";
            std::cout<<cusparseStatus[i]<<std::endl;
        }
        //cudaStreamSynchronize(stream[i]);
        if (cudaPeekAtLastError() != 0 )
        {
            std::cout<<"Getting V1  = H*V0 failed! Error: ";
            std::cout<<cudaGetErrorString(cudaPeekAtLastError())<<std::endl;
        }


    }

    for(int i = 0; i < howMany; i++)
    {
        cublasSetStream(linAlgHandle, stream[i]);
        dotTemp[i] = 0.;

        cublasStatus[i] = cublasDdot(linAlgHandle, dim[i], v1[i], 1, v0[i], 1, &dotTemp[i]);
        
        h_a[i].push_back(dotTemp[i]);
        h_b[i].push_back(0.);

        if (cublasStatus[i] != CUBLAS_STATUS_SUCCESS)
        {
            std::cout<<"Getting d_a[0] failed! Error: ";
            std::cout<<cublasStatus[i]<<std::endl;
        }

        

        //cudaStreamSynchronize(stream[i]);
        if (cublasStatus[i] != CUBLAS_STATUS_SUCCESS)
        {
            std::cout<<"Getting h_a[0] failed! Error: ";
            std::cout<<cublasStatus[i]<<std::endl;
        }

        

        if (status[i] != cudaSuccess)
        {
            std::cout<<"Memory allocation of y dummy vector failed! Error:";
            std::cout<<cudaGetErrorString( status[i] )<<std::endl;
        }

        status[i] = cudaMalloc(&y[i], dim[i]*sizeof(double));

        
        /*!
        The new vector must be rescaled and stored before Lanczos iteration can begin. 
        \verbatim
        */

        cublasStatus[i] = cublasDscal(linAlgHandle, dim[i], &beta[i], y[i], 1);
        //cudaStreamSynchronize(stream[i]);

        axpyTemp[i] = -1*h_a[i][0];

        cublasStatus[i] = cublasDaxpy(linAlgHandle, 0, &axpyTemp[i], v0[i], 1, v1[i], 1);
        //cudaStreamSynchronize(stream[i]);

        if (cublasStatus[i] != CUBLAS_STATUS_SUCCESS)
        {
            std::cout<<"V1 = V1 - alpha*V0 failed! Error: ";
            std::cout<<cublasStatus[i]<<std::endl;
        }
        if (cudaPeekAtLastError() != 0 )
        {
            std::cout<<"Getting V1  = V1 - a*V0 failed! Error: ";
            std::cout<<cudaGetErrorString(cudaPeekAtLastError())<<std::endl;
        }

        //---------Normalize V1 and copy it to Lanczos storage-----------

        normTemp[i] = 0.;
        cublasStatus[i] = cublasDnrm2(linAlgHandle, dim[i], v1[i], 1, &normTemp[i]); //this is slow for some reason

        //cudaStreamSynchronize(stream[i]);

        if (cublasStatus[i] != CUBLAS_STATUS_SUCCESS)
        {
            std::cout<<"Getting the norm of v1 failed! Error: ";
            std::cout<<cublasStatus[i]<<std::endl;
        }

        if (cudaPeekAtLastError() != 0 )
        {
            std::cout<<"Getting nrm(V1) failed! Error: ";
            std::cout<<cudaGetErrorString(cudaPeekAtLastError())<<std::endl;
        }

    }

    double* gamma = (double*)malloc(howMany*sizeof(double));

    for(int i = 0; i < howMany; i++)
    {
        cublasSetStream(linAlgHandle, stream[i]);

        h_b[i].push_back(normTemp[i]);

        normTemp[i] = 1./normTemp[i];
        gamma[i] = 1./h_b[i][1]; //alpha = 1/beta in v1 = v1 - alpha*v0
        cublasStatus[i] = cublasDscal(linAlgHandle, dim[i], &normTemp[i], v1[i], 1);


        if (cublasStatus[i] != CUBLAS_STATUS_SUCCESS)
        {
            std::cout<<"Normalizing v1 failed! Error: ";
            std::cout<<cublasStatus[i]<<std::endl;
        }


        if (cudaPeekAtLastError() != 0 )
        {
            std::cout<<"Normalizing V1 failed! Error: ";
            std::cout<<cudaGetErrorString(cudaPeekAtLastError())<<std::endl;
        }
        cudaMalloc(&lanczosStore[i][1], dim[i]*sizeof(double));
        cudaMemcpyAsync(lanczosStore[i][1], v1[i], dim[i]*sizeof(double), cudaMemcpyDeviceToDevice, stream[i]);
    }
    /*!    
    \endverbatim
    */

    /*! 
    Storage space for the tridiagonal matrix is created and flags are initialized to track progress:

	\verbatim
    */
    
    double* gsEnergy = (double*)malloc(howMany*sizeof(double));

    double* eigTemp = (double*)malloc(howMany*sizeof(double));

    int* returned = (int*)malloc(howMany*sizeof(int));

    int* iter = (int*)malloc(howMany*sizeof(int));

    bool* doneFlag = (bool*)malloc(howMany*sizeof(bool));

    double** h_H_eigen = (double**)malloc(howMany*sizeof(double*));
    double** d_H_eigen = (double**)malloc(howMany*sizeof(double*));

    double** h_diag = (double**)malloc(howMany*sizeof(double*));
    double** h_offdia = (double**)malloc(howMany*sizeof(double*));

    vector< vector < double > > h_ordered(howMany);
    /*!
    \endverbatim
    */
    /*! 
    The flags and storage are initialized for the interations
    \verbatim
    */
    for(int i = 0; i<howMany; i++)
    {
        gsEnergy[i] = 1.;
        eigTemp[i] = 0.;
        iter[i] = 0;
        doneFlag[i] = false;
        h_ordered[i].resize(numEig, 0);
        h_H_eigen[i] = (double*)malloc(maxIter*maxIter*sizeof(double));
        cudaMalloc(&d_H_eigen[i], maxIter*maxIter*sizeof(double));
        h_diag[i] = (double*)malloc(h_a[i].size()*sizeof(double));
        h_offdia[i] = (double*)malloc(h_b[i].size()*sizeof(double));
    }
    /*!
    \endverbatim
    */

    //---------Begin Lanczos iteration-----------------------------

    bool allDone = false;

    while( !allDone )
    {
        allDone = true;
        for(int i = 0; i < howMany; i++)
        {
            cublasSetStream(linAlgHandle, stream[i]);
            cusparseSetStream(sparseHandle, stream[i]);
            cudaStreamSynchronize(stream[i]);

            /*! 
            If the current diagonalization is not complete, multiply H*V1 to get a new V2
            \verbatim
            */
            if (!doneFlag[i])
            {
                iter[i]++;
                
                cusparseStatus[i] = cusparseDcsrmv(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, dim[i], dim[i], numElem[i], &alpha[i], H_descr[i], Hamiltonian[i].vals, d_H_rowPtrs[i], Hamiltonian[i].cols, v1[i], &beta[i], v2[i]);
                
                if( cusparseStatus[i] != 0)
                {
                    cout<<"Error applying H to V1 in "<<iter[i]<<"th iteration"<<endl;
                }
                //cusparseStatus[i] = cusparseDhybmv(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha[i], H_descr[i], hyb_Ham[i], v1[i], &beta[i], v2[i]); // the Hamiltonian is applied here, in this gross expression
            }
            /*!
            \endverbatim
            */
        }
        for(int i = 0; i < howMany; i++)
        {
            cublasSetStream(linAlgHandle, stream[i]);
            if (!doneFlag[i])
            {
                if (cusparseStatus[i] != CUSPARSE_STATUS_SUCCESS)
                {
                    std::cout<<"Error applying the Hamiltonian in "<<iter[i]<<"th iteration!";
                    std::cout<<"Error: "<<cusparseStatus[i]<<std::endl;
                }

                cublasStatus[i] = cublasDdot(linAlgHandle, dim[i], v1[i], 1, v2[i], 1, &dotTemp[i]);
                //cudaStreamSynchronize(stream[i]);

                h_a[i].push_back(dotTemp[i]);


                if (cublasStatus[i] != CUBLAS_STATUS_SUCCESS)
                {
                    std::cout<<"Error getting v1 * v2 in "<<iter[i]<<"th iteration! Error: ";
                    std::cout<<cublasStatus[i]<<std::endl;
                }

                axpyTemp[i] = -1.*h_b[i][iter[i]];

                cublasStatus[i] = cublasDaxpy( linAlgHandle, dim[i], &axpyTemp[i], v0[i], 1, v2[i], 1);
                if (cublasStatus[i] != CUBLAS_STATUS_SUCCESS)
                {
                    std::cout<<"Error getting (d_b/d_a)*v0 + v1 in "<<iter[i]<<"th iteration!";
                    std::cout<<"Error: "<<cublasStatus[i]<<std::endl;
                }

            }
        }

        //--------Find next set of elements in Lanczos Hamiltonian----

        for(int i = 0; i < howMany; i++)
        {
            cublasSetStream(linAlgHandle, stream[i]);
            //cudaStreamSynchronize(stream[i]);
            if (!doneFlag[i])
            {
                /*! 
                Similarly to setting up V1, V2 must be rescaled
                \verbatim
                */
                axpyTemp[i] = -1.*h_a[i][iter[i]];
                cublasStatus[i] = cublasDaxpy( linAlgHandle, dim[i], &axpyTemp[i], v1[i], 1, v2[i], 1);
                if (cublasStatus[i] != CUBLAS_STATUS_SUCCESS)
                {
                    std::cout<<"Error getting v2 + d_a*v1 in "<<iter[i]<<"th iteration! Error: ";
                    std::cout<<cublasStatus[i]<<std::endl;
                }

                cublasStatus[i] = cublasDnrm2( linAlgHandle, dim[i], v2[i], 1, &normTemp[i]);
                
                if (cublasStatus[i] != CUBLAS_STATUS_SUCCESS)
                {
                    std::cout<<"Error getting norm of v2 in "<<iter[i]<<"th iteration! Error: ";
                    std::cout<<cublasStatus[i]<<std::endl;
                }


                h_b[i].push_back(normTemp[i]);
                gamma[i] = 1./normTemp[i];
                /*!
                \endverbatim
                */
                cublasStatus[i] = cublasDscal(linAlgHandle, dim[i], &gamma[i], v2[i], 1);
                if (cublasStatus[i] != CUBLAS_STATUS_SUCCESS)
                {
                    std::cout<<"Error getting 1/d_b * v2 in "<<iter[i]<<"th iteration! Error: ";
                    std::cout<<cublasStatus[i]<<std::endl;
                }
            }
        }


        for(int i = 0; i < howMany; i++)
        {
            cublasSetStream(linAlgHandle, stream[i]);
            //status[i] = cudaStreamSynchronize(stream[i]);

            if (status[i] != cudaSuccess)
            {
                std::cout<<"Error syncing before copying v1 to v0: "<<cudaGetErrorString(status[i])<<std::endl;
            }

            if (!doneFlag[i])
            {
                /*!
                Reorthogonalization is performed on v2 to ensure that the excited states do not collapse into the groundstate
                \verbatim
                */
                for(int j = 0; j < iter[i] + 1; j++)
                {
                    cublasDdot(linAlgHandle, dim[i], v2[i], 1, lanczosStore[i][j], 1, &dotTemp[i]);
                    dotTemp[i] *= -1.;
                    cublasDaxpy(linAlgHandle, dim[i],  &dotTemp[i], lanczosStore[i][j], 1, v2[i], 1);
                    dotTemp[i] = 1. - dotTemp[i]*dotTemp[i];
                    cublasDscal(linAlgHandle, dim[i], &dotTemp[i], v2[i], 1);

                }
                /*!
                \endverbatim
                The vectors are copied down one and stored to prepare for the next iteration
                \verbatim
                */

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
             
                status[i] = cudaMalloc(&lanczosStore[i][iter[i] + 1], dim[i]*sizeof(double));
                status[i] = cudaMemcpyAsync(lanczosStore[i][iter[i] + 1], v2[i], dim[i]*sizeof(double), cudaMemcpyDeviceToDevice, stream[i]);
                /*!
                \endverbatim
                */
            }

        }

        for(int i = 0; i < howMany; i++)
        {
            if (!doneFlag[i] && iter[i] > 5)
            {
                //---Copy Lanczos matrix information for diagonalization-----
                free(h_diag[i]);
                free(h_offdia[i]);
                h_diag[i] = (double*)malloc(h_a[i].size()*sizeof(double));
                h_offdia[i] = (double*)malloc(h_b[i].size()*sizeof(double));

                h_diag[i][0] = h_a[i][0];
                for (int ii=1; ii<=iter[i]; ii++)
                {
                    h_diag[i][ii] = h_a[i][ii];
                    h_offdia[i][ii] = h_b[i][ii];
                    h_offdia[i][ii-1] = h_offdia[i][ii];
                }
                h_offdia[i][iter[i]] = 0;

                //cudaStreamSynchronize(stream[i]);

                //---------Diagonalize Lanczos matrix and check for convergence------------------
                returned[i] = tqli(h_diag[i], h_offdia[i], iter[i] + 1, maxIter, h_H_eigen[i]);
                status[i] = cudaPeekAtLastError();
                if( status[i] != cudaSuccess)
                {
                    cout<<"Error in identity! Error: "<<cudaGetErrorString(status[i])<<endl;
                }
                //cout<<"Done tqli in "<<iter[i]<<"th iteration"<<endl;
                cudaMemcpyAsync(d_H_eigen[i], h_H_eigen[i], maxIter*maxIter*sizeof(double), cudaMemcpyHostToDevice, stream[i]);

                std::sort(h_diag[i], h_diag[i] + h_a[i].size());
                for (int j = 0; j < numEig; j++)
                {
                    h_ordered[i][j] = h_diag[i][j];
                    //cout<<h_ordered[i][j]<<" ";
                }
                //cout<<endl;

                gsEnergy[i] = h_ordered[i][numEig - 1];
                doneFlag[i] = (fabs(gsEnergy[i] - eigTemp[i]) < convReq);// && iter[i] > 10;// ? (iter[i] > 10) : false;
                //doneFlag[i] = iter[i] == maxIter - 2;
                eigTemp[i] = h_ordered[i][numEig - 1];

                if (iter[i] == maxIter - 2) // have to use this or d_b will overflow
                {
                    //this stuff here is used to resize the main arrays in the case that we aren't converging quickly enough
                    h_a[i].resize(2*maxIter);
                    h_b[i].resize(2*maxIter);
                    maxIter *= 2;
                }
            }
        }
        allDone = true;
        for(int i = 0; i< howMany; i++)
        {
            allDone = (allDone && doneFlag[i]);
        }
    }


    //-------------Get groundstates------------------------------------------

    for( int i = 0; i < howMany; i++)
    {
        //cudaStreamSynchronize(stream[i]);
        //GetGroundstate<<<dim[i]/512 + 1, 512, 0, stream[i]>>>(groundstates[i], lanczosStore[i], d_H_eigen[i], iter[i], dim[i]);
    }

    //--------------Free arrays to prevent memory leaks------------------------
    for(int i = 0; i < howMany; i++)
    {
        for(int j = 0; j < numEig; j++)
        {
            std::cout<<std::setprecision(12)<<h_ordered[i][j]<<" ";
        }
        std::cout<<std::endl;

        for(int j = 0; j < iter[i]; j++)
        {
            cudaFree(lanczosStore[i][j]);
        }
        free(lanczosStore[i]);
        cudaFree(d_H_rowPtrs[i]);
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

    free(gsEnergy);
    free(eigTemp);
    free(alpha);
    free(beta);
    free(returned);
    free(iter);
    free(doneFlag);
    free(h_H_eigen);
    free(d_H_eigen);
    free(gamma);
    free(y);
    free(normTemp);
    free(axpyTemp);
    free(dotTemp);
    free(host_v0);
    free(v0);
    free(v1);
    free(v2);
    free(h_diag);
    free(h_offdia);
    free(lanczosStore);
    //free(dim);
    free(d_H_rowPtrs);

    cublasStatus[0] = cublasDestroy(linAlgHandle);

    //----------Output groundstate to file to check for correctness------

    double* host_groundstate = (double*)malloc(dim[0]*sizeof(double));
    /*std::ofstream fout;
    fout.open("lanczos.log");
    cudaMemcpy(host_groundstate, groundstates[0], dim[0]*sizeof(double), cudaMemcpyDeviceToHost);
    for(int i = 0; i < dim[0] ; i++)
    {
        fout<<host_groundstate[i]<<std::endl;
    }

    fout.close();*/
    free(host_groundstate);
    free(dim);
    if (cublasStatus[0] != CUBLAS_STATUS_SUCCESS)
    {
        printf("CUBLAS failed to shut down properly! \n");
    }

    cusparseStatus[0] = cusparseDestroy(sparseHandle);

    if (cusparseStatus[0] != CUSPARSE_STATUS_SUCCESS)
    {
        printf("CUSPARSE failed to release handle! \n");
    }


    /*if (iter == 1) {
    	std::ofstream fout;
    	fout.open("lanczos.log");
    	//fout<<normTemp<<std::endl;
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

int tqli(double* d, double* e, int n, int maxIter, double *z)

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

__global__ void GetGroundstate(double* groundstates, double** lanczosStore, double* H_eigen, int mat_dim, int vec_dim)
{

    int element = blockIdx.x*blockDim.x + threadIdx.x;

    if ( element < vec_dim )
    {
        groundstates[element] = H_eigen[0]*lanczosStore[0][element];
        for (int lancIter = 1; lancIter < mat_dim; lancIter++)
        {
            groundstates[element] += H_eigen[lancIter]*lanczosStore[lancIter][element];
        }
    }
};
