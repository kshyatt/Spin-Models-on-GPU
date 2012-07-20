#include"hamiltonian.h"

__host__ int GetBasis(int dim, parameters data, int basisPosition[], int basis[])
{
    unsigned int temp = 0;
    int realDim = 0;

    for (unsigned int i1=0; i1<dim; i1++)
    {
        temp = 0;
        basisPosition[i1] = -1;
        for (int sp =0; sp<data.nsite; sp++)
        {
            temp += (i1>>sp)&1;
        } //unpack bra
        if ( ( (data.modelType == 1) || (data.modelType == 0) ) && ((temp <= data.nsite/2 + data.Sz ) && (temp >= data.nsite/2 - data.Sz) ) )
        {
            basis[ realDim ] = i1;
            basisPosition[ i1 ] = realDim;
            realDim++;

        }
        else if ( data.modelType == 2)
        {
            basis[ realDim ] = i1;
            basisPosition[i1] = realDim;
            realDim++;
        }
    }

    /*#if CHECK == 1
        ofstream basfile;
        ofstream basposfile;
        switch( data.Sz )
        {
            case 0 :
                basfile.open("tests0basis.dat");
                basposfile.open("tests0basisposition.dat");
                break;
            case 1 :
                basfile.open("tests1basis.dat");
                basposfile.open("tests1basisposition.dat");
                break;
            case 2 :
                basfile.open("tests2basis.dat");
                basposfile.open("tests2basisposition.dat");
                break;
            case 3 :
                basfile.open("tests3basis.dat");
                basposfile.open("tests3basisposition.dat");
                break;
            case 4 :
                basfile.open("tests4basis.dat");
                basposfile.open("tests4basisposition.dat");
                break;
            case 5 :
                basfile.open("tests5basis.dat");
                basposfile.open("tests5basisposition.dat");
                break;
            case 6 :
                basfile.open("tests6basis.dat");
                basposfile.open("tests6basisposition.dat");
                break;
            case 7 :
                basfile.open("tests7basis.dat");
                basposfile.open("tests7basisposition.dat");
                break;
            case 8 :
                basfile.open("tests8basis.dat");
                basposfile.open("tests8basisposition.dat");
                break;
        }

        for( unsigned int i = 0; i < dim; i++)
        {
            cout<<i<<endl;
            if (i < realDim) basfile<<basis[i]<<endl;
            basposfile<<basisPosition[i]<<endl;
        }
        basfile.close();
        basposfile.close();
    #endif
    */
    return realDim;

}

__host__ void ConstructSparseMatrix( const int howMany, int** Bond, d_hamiltonian*& hamilLancz, parameters* data, int*& countArray, int device )
{
    cudaFree(0);
    cudaSetDevice(device);

    int* numElem = (int*)malloc( howMany * sizeof(int) );
    f_hamiltonian* d_H = (f_hamiltonian*)malloc( howMany * sizeof(f_hamiltonian) );

    int stride[howMany];

    int** basisPosition = (int**)malloc(howMany*sizeof(int*));
    int** basis = (int**)malloc(howMany*sizeof(int*));

    int** d_basisPosition = (int**)malloc(howMany*sizeof(int*));
    int** d_basis = (int**)malloc(howMany*sizeof(int*));

    int** d_Bond = (int**)malloc(howMany*sizeof(int*));

    int paddedDim[howMany];
    int rawSize[howMany];

    dim3* bpg = (dim3*)malloc(howMany*sizeof(dim3));
    dim3* tpb = (dim3*)malloc( howMany * sizeof(dim3) );

    int* offset = (int*)malloc( howMany * sizeof(int) );

    cudaStream_t stream[ howMany ];

    cudaError_t status[ howMany ];

    for(int i = 0; i < howMany; i++)
    {

        switch( data[i].modelType )
        {
        case 0: //2D spin 1/2 Heisenberg
        case 1: //2D spin 1/2 XY
            numElem[i] = 0;
            stride[i] = 4*data[i].nsite + 1;

            d_H[i].fullDim = 2;
            for (int ch=1; ch<data[i].nsite; ch++) d_H[i].fullDim *= 2;
            break;
        case 2: //1D Transverse Field Ising Model
            numElem[i] = 0;
            stride[i] = 2*data[i].nsite + 1;

            d_H[i].fullDim = 2;
            for (int ch=1; ch<data[i].nsite; ch++) d_H[i].fullDim *= 2;
            break;
        }

        basisPosition[i] = (int*)malloc(d_H[i].fullDim*sizeof(int));
        basis[i] = (int*)malloc(d_H[i].fullDim*sizeof(int));

        d_H[i].sectorDim = GetBasis(d_H[i].fullDim, data[i], basisPosition[i], basis[i]);
        
        status[i] = cudaMalloc((void**)&d_basisPosition[i], d_H[i].fullDim*sizeof(int));
        
        if (status[i] != cudaSuccess)
        {
            cout<<"Error allocating "<<i<<"th d_basisPosition array: "<<cudaGetErrorString(status[i])<<endl;
        }
        
        status[i] = cudaMalloc((void**)&d_basis[i], d_H[i].sectorDim*sizeof(int));

        if (status[i] != cudaSuccess)
        {
            cout<<"Error allocating "<<i<<"th d_basis array: "<<cudaGetErrorString(status[i])<<endl;
        }

        status[i] = cudaStreamCreate(&stream[i]);

        if (status[i] != cudaSuccess)
        {
            cout<<"Error creating "<<i<<"th stream: "<<cudaGetErrorString(status[i])<<endl;
        }
    }

    for(int i = 0; i<howMany; i++)
    {
        //-------Copy basis information to the GPU------------------------
        
        status[i] = cudaMemcpyAsync(d_basisPosition[i], basisPosition[i], d_H[i].fullDim*sizeof(int), cudaMemcpyHostToDevice, stream[i]);

        if (status[i] != cudaSuccess)
        {
            cout<<"Error copying "<<i<<"th basisPosition: "<<cudaGetErrorString(status[i])<<endl;
        }

        status[i] = cudaMemcpyAsync(d_basis[i], basis[i], d_H[i].sectorDim*sizeof(int), cudaMemcpyHostToDevice, stream[i]);

        if (status[i] != cudaSuccess)
        {
            cout<<"Error copying "<<i<<"th basis: "<<cudaGetErrorString(status[i])<<endl;
        }

        //---------Determine the size of arrays which will be needed for Hamiltonian storage --------------

        if ( d_H[i].sectorDim % 512 )
        {
            paddedDim[i] = (d_H[i].sectorDim/512 + 1)*512;
        }
        else
        {
            paddedDim[i] = d_H[i].sectorDim;
        }

        rawSize[i] = paddedDim[i] + ((stride[i] - 1)*d_H[i].sectorDim);

        if (rawSize[i] % 2048 )
        {
            rawSize[i] = (rawSize[i]/2048 + 1)*2048;
        }

        //-------Allocate space on the GPU to store the Hamiltonian---------

        status[i] = cudaMalloc((void**)&d_H[i].rows, rawSize[i]*sizeof(int));
        
        if (status[i] != cudaSuccess)
        {
            cout<<"Error creating "<<i<<"th rows array: "<<cudaGetErrorString(status[i])<<endl;
        }
        
        status[i] = cudaMalloc((void**)&d_H[i].cols, rawSize[i]*sizeof(int));
        
        if (status[i] != cudaSuccess)
        {
            cout<<"Error creating "<<i<<"th cols array: "<<cudaGetErrorString(status[i])<<endl;
        }
        
        status[i] = cudaMalloc((void**)&d_H[i].vals, rawSize[i]*sizeof(float));
        
        if (status[i] != cudaSuccess)
        {
            cout<<"Error creating "<<i<<"th values array: "<<cudaGetErrorString(status[i])<<endl;
        }

        status[i] = cudaMalloc((void**)&d_H[i].set, rawSize[i]*sizeof(int));
        
        if (status[i] != cudaSuccess)
        {
            cout<<"Error creating "<<i<<"th flag array: "<<cudaGetErrorString(status[i])<<endl;
        }

        status[i] = cudaMemset(d_H[i].set, 0, rawSize[i]*sizeof(int));

        if (status[i] != cudaSuccess)
        {
            cout<<"Error memsetting "<<i<<"th flags array: "<<cudaGetErrorString(status[i])<<endl;
        }
        
        status[i] = cudaMalloc((void**)&d_Bond[i], 3*data[i].nsite*sizeof(int));
        
        if (status[i] != cudaSuccess)
        {
            cout<<"Error creating "<<i<<"th bonds array: "<<cudaGetErrorString(status[i])<<endl;
        }

        status[i] = cudaMemcpyAsync(d_Bond[i], Bond[i], 3*data[i].nsite*sizeof(int), cudaMemcpyHostToDevice, stream[i]);

        if (status[i] != cudaSuccess)
        {
            cout<<"Error copying "<<i<<"th bonds array: "<<cudaGetErrorString(status[i])<<endl;
        }

        //------Determine the appropriate number of threads and blocks to launch for Hamiltonian generation------------

        tpb[i].x = data[i].nsite;
        do
        {
            tpb[i].x *= 2;
        }
        while(tpb[i].x < 512);

        if ( ( ( stride[i] - 1 ) * d_H[i].sectorDim ) % tpb[i].x == 1 )
        {
            bpg[i].x = ( ( ( stride[i] - 1 ) * d_H[i].sectorDim ) / tpb[i].x ) + 1;
        }

        else
        {
            bpg[i].x = ( ( stride[i] - 1 ) * d_H[i].sectorDim ) / tpb[i].x;
        }

        if (bpg[i].x > (1<<16 - 1 ))
        {
            offset[i] = (1<<16 - 1);
        }

        status[i] = cudaStreamSynchronize(stream[i]);

        if (status[i] != cudaSuccess)
        {
            cout<<"Error synchronizing "<<i<<"th stream: "<<cudaGetErrorString(status[i])<<endl;
        }

        //------Generate diagonal elements of the Hamiltonian--------

        switch( data[i].modelType )
        {
        case 0:

            FillDiagonalsHeisenberg<<<d_H[i].sectorDim/512 + 1, 512, device, stream[i]>>>(d_basis[i], d_H[i], d_Bond[i], data[i]);
            break;
        
        case 1:
        
            FillDiagonalsXY<<<d_H[i].sectorDim/512 + 1, 512, device, stream[i]>>>(d_basis[i], d_H[i], d_Bond[i], data[i]);
            break;
        
        case 2:
        
            FillDiagonalsTFI<<<d_H[i].sectorDim/512 + 1, 512, device, stream[i]>>>(d_basis[i], d_H[i], d_Bond[i], data[i]);
            break;
        }
    }

    for(int i = 0; i < howMany; i++)
    {
        status[i] = cudaStreamSynchronize(stream[i]);

        if (status[i] != cudaSuccess)
        {
            cout<<"Error synchronizing "<<i<<"th stream: "<<cudaGetErrorString(status[i])<<endl;
        }

        status[i] = cudaPeekAtLastError();
    
        if (status[i] != cudaSuccess)
        {
            cout<<"Error in "<<i<<"th stream: "<<cudaGetErrorString(status[i])<<endl;
        }

        //--------Launch kernel to create offdiagonal Hamiltonian elements-------

        switch( data[i].modelType ) 
        {

        case 0:

            while ( bpg[i].x > (1<<16 - 1) )
            {
                
                FillSparseHeisenberg<<< (1 << 16 - 1), tpb[i].x, device, stream[i]>>>(d_basisPosition[i], d_basis[i], d_H[i], d_Bond[i], data[i], 0);
                
                if ( bpg[i].x - offset[i] > (1<<16 - 1))
                {
                    FillSparseHeisenberg<<< (1<<16 - 1), tpb[i].x, device, stream[i]>>>(d_basisPosition[i], d_basis[i], d_H[i], d_Bond[i], data[i], offset[i]);
                
                    bpg[i].x -= (1<<16 - 1);
                    offset[i] += (1<< 16 - 1);
                }

                else
                {
                    FillSparseHeisenberg<<< bpg[i].x - offset[i], tpb[i].x, device, stream[i]>>>(d_basisPosition[i], d_basis[i], d_H[i], d_Bond[i], data[i], offset[i]);
                    bpg[i].x -= offset[i];
                }
            }
            
            FillSparseHeisenberg<<< bpg[i].x, tpb[i].x, device, stream[i]>>>(d_basisPosition[i], d_basis[i], d_H[i], d_Bond[i], data[i], 0);
            
            break;

        case 1:

            while ( bpg[i].x > (1<<16 - 1) )
            {
            
                FillSparseXY<<< (1 << 16) - 1, tpb[i].x, device, stream[i]>>>(d_basisPosition[i], d_basis[i], d_H[i], d_Bond[i], data[i], offset[i]);
                //if ( bpg[i].x -  > (1<<16 - 1))
                //{

                   // FillSparseXY<<< (1<<16 - 1), tpb[i].x, device, stream[i]>>>(d_basisPosition[i], d_basis[i], d_H[i], d_Bond[i], data[i], offset[i]);
                    bpg[i].x -= (1<<16 - 1);
                    offset[i] += (1 << 16 - 1);
                //}
                //else
                //{
                    //FillSparseXY<<< (1 << 16 - 1), tpb[i].x, device, stream[i]>>>(d_basisPosition[i], d_basis[i], d_H[i], d_Bond[i], data[i], offset[i]);
                 //   bpg[i].x -= offset[i];
                //}
            }
            
            FillSparseXY<<< bpg[i].x, tpb[i].x, device, stream[i]>>>(d_basisPosition[i], d_basis[i], d_H[i], d_Bond[i], data[i], 0);
            break;

        case 2:

            while ( bpg[i].x > (1<<16 - 1) )
            {
                FillSparseTFI<<< (1 << 16 - 1), tpb[i].x, device, stream[i]>>>(d_basisPosition[i], d_basis[i], d_H[i], d_Bond[i], data[i], 0);
            
                if ( bpg[i].x - offset[i] > (1<<16 - 1))
                {

                    FillSparseTFI<<< (1<<16 - 1), tpb[i].x, device, stream[i]>>>(d_basisPosition[i], d_basis[i], d_H[i], d_Bond[i], data[i], offset[i]);
            
                    bpg[i].x -= (1<<16 - 1);
                    offset[i] += (1 << 16 - 1);
                }
            
                else
                {
                    FillSparseTFI<<< bpg[i].x - offset[i], tpb[i].x, device, stream[i]>>>(d_basisPosition[i], d_basis[i], d_H[i], d_Bond[i], data[i], offset[i]);
                    bpg[i].x -= offset[i];
                }
            }
            
            FillSparseTFI<<< bpg[i].x, tpb[i].x, device, stream[i]>>>(d_basisPosition[i], d_basis[i], d_H[i], d_Bond[i], data[i], 0);
            break;
        }

        status[i] = cudaPeekAtLastError();
        
        if (status[i] != cudaSuccess)
        {
            cout<<"Error in "<<i<<"th stream: "<<cudaGetErrorString(status[i])<<endl;
        }

    }

    cudaThreadSynchronize();
    
    for(int i = 0; i < howMany; i++)
    {
        //thrust::device_ptr<int> red_ptr(d_H[i].set);
        //numElem[i] = thrust::reduce(red_ptr, red_ptr + rawSize[i]);
    }

    //----Free GPU storage for basis and bond information which is not needed------

    for(int i = 0; i < howMany; i++)
    {

        status[i] = cudaFree(d_basis[i]);
    
        if ( status[i] != cudaSuccess)
        {
            cout<<"Error freeing "<<i<<"th basis array: "<<cudaGetErrorString(status[i])<<endl;
        }
    
        status[i] = cudaFree(d_basisPosition[i]);
    
        if (status[i] != cudaSuccess)
        {
            cout<<"Error freeing "<<i<<"th basisPosition array: "<<cudaGetErrorString(status[i])<<endl;
        }
    
        status[i] = cudaFree(d_Bond[i]); // we don't need these later on
    
        if (status[i] != cudaSuccess)
        {

            cout<<"Error freeing "<<i<<"th Bond array: "<<cudaGetErrorString(status[i])<<endl;
        }
    
        free(basis[i]);
        free(basisPosition[i]);
    }
    //----------------Sorting Hamiltonian--------------------------//

    float** valsBuffer = (float**)malloc(howMany*sizeof(float*));
    int sortNumber[howMany];

    //-------Row-sort Hamiltonian to eliminate zero-valued elements---------

    for(int i = 0; i<howMany; i++)
    {

        sortEngine_t engine;
    
        sortStatus_t sortstatus = sortCreateEngine("sort/sort/src/cubin64/", &engine);

        MgpuSortData sortData;

        sortData.AttachKey((unsigned int*)d_H[i].rows);
        //sortdata.AttachKey((unsigned int*)d_H[i].index);
        //sortdata.AttachVal(0, (unsigned int*)d_H[i].rows);
        sortData.AttachVal(0, (unsigned int*)d_H[i].cols);
        sortData.AttachVal(1, (unsigned int*)d_H[i].vals);

        sortNumber[i] = rawSize[i];

        sortData.Alloc(engine, sortNumber[i], 2);

        sortData.firstBit = 0;
        sortData.endBit = 31; //2*lattice_Size[i];

        sortArray(engine, &sortData);

        //-----Allocate final Hamiltonian storage and copy data to it-------

        status[i] = cudaMalloc((void**)&hamilLancz[i].vals, numElem[i]*sizeof(cuDoubleComplex));

        if (status[i] != cudaSuccess)
        {
            cout<<"Error allocating "<<i<<"th lancz values array: "<<cudaGetErrorString(status[i])<<endl;
        }

        status[i] = cudaMalloc((void**)&hamilLancz[i].rows, numElem[i]*sizeof(int));

        if (status[i] != cudaSuccess)
        {
            cout<<"Error allocating "<<i<<"th lancz rows array: "<<cudaGetErrorString(status[i])<<endl;
        }

        status[i] = cudaMalloc((void**)&hamilLancz[i].cols, numElem[i]*sizeof(int));

        if (status[i] != cudaSuccess)
        {
            cout<<"Error allocating "<<i<<"th lancz cols array: "<<cudaGetErrorString(status[i]);
        }

        //half of these are commented out to get check indexed sort vs nonindexed
        cudaMemcpy(hamilLancz[i].rows, (int*)sortData.keys[0], numElem[i]*sizeof(int), cudaMemcpyDeviceToDevice);

        //cudaMemcpy(hamilLancz[i].rows, (int*)sortdata.values1[0], numElem[i]*sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(hamilLancz[i].cols, (int*)sortData.values1[0], numElem[i]*sizeof(int), cudaMemcpyDeviceToDevice);

        //cudaMemcpy(hamilLancz[i].cols, (int*)sortdata.values2[0], numElem[i]*sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMalloc((void**)&valsBuffer[i], numElem[i]*sizeof(float));

        cudaMemcpy(valsBuffer[i], (float*)sortData.values2[0], numElem[i]*sizeof(float), cudaMemcpyDeviceToDevice);

        //cudaMemcpy(vals_buffer[i], (float*)sortdata.values3[0], numElem[i]*sizeof(float), cudaMemcpyDeviceToDevice);

        FullToCOO<<<numElem[i]/1024 + 1, 1024>>>(numElem[i], valsBuffer[i], hamilLancz[i].vals, d_H[i].sectorDim);

        //int* h_index = (int*)malloc(numElem[i]*sizeof(int));
        // status[i] = cudaMemcpy(h_index, d_H[i].index, numElem[i]*sizeof(int), cudaMemcpyDeviceToHost);

        sortReleaseEngine(engine);
    
        cudaFree(d_H[i].rows);
        cudaFree(d_H[i].cols);
        cudaFree(d_H[i].vals);
        cudaFree(d_H[i].set);

        hamilLancz[i].fullDim = d_H[i].fullDim;
        hamilLancz[i].sectorDim = d_H[i].sectorDim;
        countArray[i] = numElem[i];

        //----This code dumps the Hamiltonian to a file-------------
        /*
        #if CHECK == 1

            double* h_vals = (double*)malloc(numElem[i]*sizeof(double));
            int* h_rows = (int*)malloc(numElem[i]*sizeof(int));
            int* h_cols = (int*)malloc(numElem[i]*sizeof(int));

            status[i] = cudaMemcpy(h_vals, hamilLancz[i].vals, numElem[i]*sizeof(double), cudaMemcpyDeviceToHost);

            if (status[i] != cudaSuccess)
            {
                cout<<"Error copying to h_vals: "<<cudaGetErrorString(status[i])<<endl;
            }

            status[i] = cudaMemcpy(h_rows, hamilLancz[i].rows, numElem[i]*sizeof(int), cudaMemcpyDeviceToHost);

            if (status[i] != cudaSuccess)
            {
                cout<<"Error copying to h_rows: "<<cudaGetErrorString(status[i])<<endl;
            }

            status[i] = cudaMemcpy(h_cols, hamilLancz[i].cols, numElem[i]*sizeof(int), cudaMemcpyDeviceToHost);

            if (status[i] != cudaSuccess)
            {
                cout<<"Error copying to h_cols: "<<cudaGetErrorString(status[i])<<endl;
            }


            ofstream hamrows;
            ofstream hamcols;
            ofstream hamvals;
            
            switch (data[0].modelType)
            {
                case 0 :
                    hamrows.open("heistestrows.dat");
                    hamcols.open("heistestcols.dat");
                    hamvals.open("heistestvals.dat");
                    break;
                case 1 :
                    hamrows.open("xytestrows.dat");
                    hamcols.open("xytestcols.dat");
                    hamvals.open("xytestvals.dat");
                    break;
                case 2 :
                    hamrows.open("isingtestrows.dat");
                    hamcols.open("isingtestcols.dat");
                    hamvals.open("isingtestvals.dat");
                    break;
            }
            for(int j = 0; j < numElem[i]; j++)
            {
                
                hamrows<<h_rows[j]<<endl;
                hamcols<<h_cols[j]<<endl;
                hamvals<<h_vals[j]<<endl;
            }
            hamrows.close();
            hamcols.close();
            hamvals.close();
            free(h_rows);
            free(h_cols);
            free(h_vals);

        #endif
        */

        cudaStreamSynchronize(stream[i]);
        cudaFree(valsBuffer[i]);
        free(Bond[i]);
    }
    cudaDeviceSynchronize();

    //----Free all the array storage to avoid memory leaks---------

    free(d_basisPosition);
    free(d_Bond);
    free(d_basis);
    free(basis);
    free(basisPosition);
    free(d_H);
    free(bpg);
    free(tpb);
    free(valsBuffer);
    memcpy(countArray, numElem, howMany*sizeof(int));
    //cout<<numElem[0]<<endl;
    free(numElem);
    //cudaFree(d_numElem);
    //return numElem;
}

/*Function: FullToCOO - takes a full sparse matrix and transforms it into COO format
Inputs - numElem - the total number of nonzero elements
H_vals - the Hamiltonian values
H_pos - the Hamiltonian positions
hamil_Values - a 1D array that will store the values for the COO form

*/
__global__ void FullToCOO(int numElem, float* H_vals, double* hamilValues, int dim)
{

    int i = threadIdx.x + blockDim.x*blockIdx.x;

    if (i < numElem)
    {

        hamilValues[i] = H_vals[i];


    }
}
;
