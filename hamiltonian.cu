#include"hamiltonian.h"

/* NOTE: this function uses FORTRAN style matrices, where the values and positions are stored in a ONE dimensional array! Don't forget this! */
/* Function GetBasis - fills two arrays with information about the basis
Inputs: dim - the initial dimension of the Hamiltonian
lattice_Size - the number of sites
Sz - the value of the Sz operator
basis_Position[] - an empty array that records the positions of the basis
basis - an empty array that records the basis
Outputs: basis_Position - a full array now
basis[] - a full array now

*/
__host__ int GetBasis(int dim, parameters data, int basis_Position[], int basis[])
{
    unsigned int temp = 0;
    int realdim = 0;

    for (unsigned int i1=0; i1<dim; i1++)
    {
        temp = 0;
        basis_Position[i1] = -1;
        for (int sp =0; sp<data.nsite; sp++)
        {
            temp += (i1>>sp)&1;
        } //unpack bra
        if ( ( (data.model_type == 1) || (data.model_type == 0) ) && temp == data.nsite/2 + data.Sz)
        {
            basis[realdim] = i1;
            basis_Position[i1] = realdim;
            realdim++;

        }
        else if ( data.model_type == 2)
        {
            basis[realdim] = i1;
            basis_Position[i1] = realdim;
            realdim++;
        }
    }
    return realdim;

}

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

__host__ void ConstructSparseMatrix(const int how_many, int** Bond, d_hamiltonian*& hamil_lancz, parameters* data, int*& count_array, int device)
{
    cudaFree(0);
    cudaSetDevice(device);

    int* num_Elem = (int*)malloc(how_many*sizeof(int));
    f_hamiltonian* d_H = (f_hamiltonian*)malloc(how_many*sizeof(f_hamiltonian));

    int stride[how_many];

    int** basis_Position = (int**)malloc(how_many*sizeof(int*));
    int** basis = (int**)malloc(how_many*sizeof(int*));

    int** d_basis_Position = (int**)malloc(how_many*sizeof(int*));
    int** d_basis = (int**)malloc(how_many*sizeof(int*));

    int** d_Bond = (int**)malloc(how_many*sizeof(int*));

    int padded_dim[how_many];
    int raw_size[how_many];

    dim3* bpg = (dim3*)malloc(how_many*sizeof(dim3));
    dim3* tpb = (dim3*)malloc(how_many*sizeof(dim3));

    int* offset = (int*)malloc(how_many*sizeof(int));

    cudaStream_t stream[how_many];

    cudaError_t status[how_many];

    for(int i = 0; i<how_many; i++)
    {

        switch(data[i].model_type)
        {
        case 0: //2D spin 1/2 Heisenberg
        case 1: //2D spin 1/2 XY
            num_Elem[i] = 0;
            stride[i] = 4*data[i].nsite + 1;

            d_H[i].fulldim = 2;
            for (int ch=1; ch<data[i].nsite; ch++) d_H[i].fulldim *= 2;
            break;
        case 2: //1D Transverse Field Ising Model
            num_Elem[i] = 0;
            stride[i] = 2*data[i].nsite + 1;

            d_H[i].fulldim = 2;
            for (int ch=1; ch<data[i].nsite; ch++) d_H[i].fulldim *= 2;
            break;
        }

        basis_Position[i] = (int*)malloc(d_H[i].fulldim*sizeof(int));
        basis[i] = (int*)malloc(d_H[i].fulldim*sizeof(int));

        d_H[i].sectordim = GetBasis(d_H[i].fulldim, data[i], basis_Position[i], basis[i]);
        
        status[i] = cudaMalloc((void**)&d_basis_Position[i], d_H[i].fulldim*sizeof(int));
        
        if (status[i] != cudaSuccess)
        {
            cout<<"Error allocating "<<i<<"th d_basis_Position array: "<<cudaGetErrorString(status[i])<<endl;
        }
        
        status[i] = cudaMalloc((void**)&d_basis[i], d_H[i].sectordim*sizeof(int));

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

    for(int i = 0; i<how_many; i++)
    {
        //-------Copy basis information to the GPU------------------------
        
        status[i] = cudaMemcpyAsync(d_basis_Position[i], basis_Position[i], d_H[i].fulldim*sizeof(int), cudaMemcpyHostToDevice, stream[i]);

        if (status[i] != cudaSuccess)
        {
            cout<<"Error copying "<<i<<"th basis_Position: "<<cudaGetErrorString(status[i])<<endl;
        }

        status[i] = cudaMemcpyAsync(d_basis[i], basis[i], d_H[i].sectordim*sizeof(int), cudaMemcpyHostToDevice, stream[i]);

        if (status[i] != cudaSuccess)
        {
            cout<<"Error copying "<<i<<"th basis: "<<cudaGetErrorString(status[i])<<endl;
        }

        //---------Determine the size of arrays which will be needed for Hamiltonian storage --------------

        if ( d_H[i].sectordim % 512 )
        {
            padded_dim[i] = (d_H[i].sectordim/512 + 1)*512;
        }
        else
        {
            padded_dim[i] = d_H[i].sectordim;
        }

        raw_size[i] = padded_dim[i] + ((stride[i] - 1)*d_H[i].sectordim);

        if (raw_size[i] % 2048 )
        {
            raw_size[i] = (raw_size[i]/2048 + 1)*2048;
        }

        //-------Allocate space on the GPU to store the Hamiltonian---------

        status[i] = cudaMalloc((void**)&d_H[i].rows, raw_size[i]*sizeof(int));
        
        if (status[i] != cudaSuccess)
        {
            cout<<"Error creating "<<i<<"th rows array: "<<cudaGetErrorString(status[i])<<endl;
        }
        
        status[i] = cudaMalloc((void**)&d_H[i].cols, raw_size[i]*sizeof(int));
        
        if (status[i] != cudaSuccess)
        {
            cout<<"Error creating "<<i<<"th cols array: "<<cudaGetErrorString(status[i])<<endl;
        }
        
        status[i] = cudaMalloc((void**)&d_H[i].vals, raw_size[i]*sizeof(float));
        
        if (status[i] != cudaSuccess)
        {
            cout<<"Error creating "<<i<<"th values array: "<<cudaGetErrorString(status[i])<<endl;
        }

        status[i] = cudaMalloc((void**)&d_H[i].set, raw_size[i]*sizeof(int));
        
        if (status[i] != cudaSuccess)
        {
            cout<<"Error creating "<<i<<"th flag array: "<<cudaGetErrorString(status[i])<<endl;
        }

        status[i] = cudaMemset(d_H[i].set, 0, raw_size[i]*sizeof(int));

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

        if ( ( ( stride[i] - 1 ) * d_H[i].sectordim ) % tpb[i].x == 1 )
        {
            bpg[i].x = ( ( ( stride[i] - 1 ) * d_H[i].sectordim ) / tpb[i].x ) + 1;
        }

        else
        {
            bpg[i].x = ( ( stride[i] - 1 ) * d_H[i].sectordim ) / tpb[i].x;
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

        switch(data[i].model_type)
        {
        case 0:

            FillDiagonalsHeisenberg<<<d_H[i].sectordim/512 + 1, 512, device, stream[i]>>>(d_basis[i], d_H[i], d_Bond[i], data[i]);
            break;
        
        case 1:
        
            FillDiagonalsXY<<<d_H[i].sectordim/512 + 1, 512, device, stream[i]>>>(d_basis[i], d_H[i], d_Bond[i], data[i]);
            break;
        
        case 2:
        
            FillDiagonalsTFI<<<d_H[i].sectordim/512 + 1, 512, device, stream[i]>>>(d_basis[i], d_H[i], d_Bond[i], data[i]);
            break;
        }
    }

    for(int i = 0; i < how_many; i++)
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

        switch(data[i].model_type)
        {

        case 0:

            while ( bpg[i].x > (1<<16 - 1) )
            {
                
                FillSparseHeisenberg<<< (1 << 16 - 1), tpb[i].x, device, stream[i]>>>(d_basis_Position[i], d_basis[i], d_H[i], d_Bond[i], data[i], 0);
                
                if ( bpg[i].x - offset[i] > (1<<16 - 1))
                {
                    FillSparseHeisenberg<<< (1<<16 - 1), tpb[i].x, device, stream[i]>>>(d_basis_Position[i], d_basis[i], d_H[i], d_Bond[i], data[i], offset[i]);
                
                    bpg[i].x -= (1<<16 - 1);
                    offset[i] += (1<< 16 - 1);
                }

                else
                {
                    FillSparseHeisenberg<<< bpg[i].x - offset[i], tpb[i].x, device, stream[i]>>>(d_basis_Position[i], d_basis[i], d_H[i], d_Bond[i], data[i], offset[i]);
                    bpg[i].x -= offset[i];
                }
            }
            
            FillSparseHeisenberg<<< bpg[i].x, tpb[i].x, device, stream[i]>>>(d_basis_Position[i], d_basis[i], d_H[i], d_Bond[i], data[i], 0);
            
            break;

        case 1:

            while ( bpg[i].x > (1<<16 - 1) )
            {
            
                FillSparseXY<<< (1 << 16) - 1, tpb[i].x, device, stream[i]>>>(d_basis_Position[i], d_basis[i], d_H[i], d_Bond[i], data[i], offset[i]);
                //if ( bpg[i].x -  > (1<<16 - 1))
                //{

                   // FillSparseXY<<< (1<<16 - 1), tpb[i].x, device, stream[i]>>>(d_basis_Position[i], d_basis[i], d_H[i], d_Bond[i], data[i], offset[i]);
                    bpg[i].x -= (1<<16 - 1);
                    offset[i] += (1 << 16 - 1);
                //}
                //else
                //{
                    //FillSparseXY<<< (1 << 16 - 1), tpb[i].x, device, stream[i]>>>(d_basis_Position[i], d_basis[i], d_H[i], d_Bond[i], data[i], offset[i]);
                 //   bpg[i].x -= offset[i];
                //}
            }
            
            FillSparseXY<<< bpg[i].x, tpb[i].x, device, stream[i]>>>(d_basis_Position[i], d_basis[i], d_H[i], d_Bond[i], data[i], 0);
            break;

        case 2:

            while ( bpg[i].x > (1<<16 - 1) )
            {
                FillSparseTFI<<< (1 << 16 - 1), tpb[i].x, device, stream[i]>>>(d_basis_Position[i], d_basis[i], d_H[i], d_Bond[i], data[i], 0);
            
                if ( bpg[i].x - offset[i] > (1<<16 - 1))
                {

                    FillSparseTFI<<< (1<<16 - 1), tpb[i].x, device, stream[i]>>>(d_basis_Position[i], d_basis[i], d_H[i], d_Bond[i], data[i], offset[i]);
            
                    bpg[i].x -= (1<<16 - 1);
                    offset[i] += (1 << 16 - 1);
                }
            
                else
                {
                    FillSparseTFI<<< bpg[i].x - offset[i], tpb[i].x, device, stream[i]>>>(d_basis_Position[i], d_basis[i], d_H[i], d_Bond[i], data[i], offset[i]);
                    bpg[i].x -= offset[i];
                }
            }
            
            FillSparseTFI<<< bpg[i].x, tpb[i].x, device, stream[i]>>>(d_basis_Position[i], d_basis[i], d_H[i], d_Bond[i], data[i], 0);
            break;
        }

        status[i] = cudaPeekAtLastError();
        
        if (status[i] != cudaSuccess)
        {
            cout<<"Error in "<<i<<"th stream: "<<cudaGetErrorString(status[i])<<endl;
        }

    }

    cudaThreadSynchronize();
    
    for(int i = 0; i < how_many; i++)
    {
        thrust::device_ptr<int> red_ptr(d_H[i].set);
        num_Elem[i] = thrust::reduce(red_ptr, red_ptr + raw_size[i]);
    }

    //----Free GPU storage for basis and bond information which is not needed------

    for(int i = 0; i < how_many; i++)
    {

        status[i] = cudaFree(d_basis[i]);
    
        if ( status[i] != cudaSuccess)
        {
            cout<<"Error freeing "<<i<<"th basis array: "<<cudaGetErrorString(status[i])<<endl;
        }
    
        status[i] = cudaFree(d_basis_Position[i]);
    
        if (status[i] != cudaSuccess)
        {
            cout<<"Error freeing "<<i<<"th basis_Position array: "<<cudaGetErrorString(status[i])<<endl;
        }
    
        status[i] = cudaFree(d_Bond[i]); // we don't need these later on
    
        if (status[i] != cudaSuccess)
        {

            cout<<"Error freeing "<<i<<"th Bond array: "<<cudaGetErrorString(status[i])<<endl;
        }
    
        free(basis[i]);
        free(basis_Position[i]);
    }
    //----------------Sorting Hamiltonian--------------------------//

    float** vals_buffer = (float**)malloc(how_many*sizeof(float*));
    int sortnumber[how_many];

    //-------Row-sort Hamiltonian to eliminate zero-valued elements---------

    for(int i = 0; i<how_many; i++)
    {

        sortEngine_t engine;
    
        sortStatus_t sortstatus = sortCreateEngine("sort/sort/src/cubin64/", &engine);

        MgpuSortData sortdata;

        sortdata.AttachKey((unsigned int*)d_H[i].rows);
        //sortdata.AttachKey((unsigned int*)d_H[i].index);
        //sortdata.AttachVal(0, (unsigned int*)d_H[i].rows);
        sortdata.AttachVal(0, (unsigned int*)d_H[i].cols);
        sortdata.AttachVal(1, (unsigned int*)d_H[i].vals);

        sortnumber[i] = raw_size[i];

        sortdata.Alloc(engine, sortnumber[i], 2);

        sortdata.firstBit = 0;
        sortdata.endBit = 31; //2*lattice_Size[i];

        sortArray(engine, &sortdata);

        //-----Allocate final Hamiltonian storage and copy data to it-------

        status[i] = cudaMalloc((void**)&hamil_lancz[i].vals, num_Elem[i]*sizeof(cuDoubleComplex));

        if (status[i] != cudaSuccess)
        {
            cout<<"Error allocating "<<i<<"th lancz values array: "<<cudaGetErrorString(status[i])<<endl;
        }

        status[i] = cudaMalloc((void**)&hamil_lancz[i].rows, num_Elem[i]*sizeof(int));

        if (status[i] != cudaSuccess)
        {
            cout<<"Error allocating "<<i<<"th lancz rows array: "<<cudaGetErrorString(status[i])<<endl;
        }

        status[i] = cudaMalloc((void**)&hamil_lancz[i].cols, num_Elem[i]*sizeof(int));

        if (status[i] != cudaSuccess)
        {
            cout<<"Error allocating "<<i<<"th lancz cols array: "<<cudaGetErrorString(status[i]);
        }

        //half of these are commented out to get check indexed sort vs nonindexed
        cudaMemcpy(hamil_lancz[i].rows, (int*)sortdata.keys[0], num_Elem[i]*sizeof(int), cudaMemcpyDeviceToDevice);

        //cudaMemcpy(hamil_lancz[i].rows, (int*)sortdata.values1[0], num_Elem[i]*sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(hamil_lancz[i].cols, (int*)sortdata.values1[0], num_Elem[i]*sizeof(int), cudaMemcpyDeviceToDevice);

        //cudaMemcpy(hamil_lancz[i].cols, (int*)sortdata.values2[0], num_Elem[i]*sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMalloc((void**)&vals_buffer[i], num_Elem[i]*sizeof(float));

        cudaMemcpy(vals_buffer[i], (float*)sortdata.values2[0], num_Elem[i]*sizeof(float), cudaMemcpyDeviceToDevice);

        //cudaMemcpy(vals_buffer[i], (float*)sortdata.values3[0], num_Elem[i]*sizeof(float), cudaMemcpyDeviceToDevice);

        FullToCOO<<<num_Elem[i]/1024 + 1, 1024>>>(num_Elem[i], vals_buffer[i], hamil_lancz[i].vals, d_H[i].sectordim);

        //int* h_index = (int*)malloc(num_Elem[i]*sizeof(int));
        // status[i] = cudaMemcpy(h_index, d_H[i].index, num_Elem[i]*sizeof(int), cudaMemcpyDeviceToHost);

        sortReleaseEngine(engine);
    
        cudaFree(d_H[i].rows);
        cudaFree(d_H[i].cols);
        cudaFree(d_H[i].vals);
        cudaFree(d_H[i].set);

        hamil_lancz[i].fulldim = d_H[i].fulldim;
        hamil_lancz[i].sectordim = d_H[i].sectordim;
        count_array[i] = num_Elem[i];

        //----This code dumps the Hamiltonian to a file-------------

        double* h_vals = (double*)malloc(num_Elem[i]*sizeof(double));
        int* h_rows = (int*)malloc(num_Elem[i]*sizeof(int));
        int* h_cols = (int*)malloc(num_Elem[i]*sizeof(int));

        status[i] = cudaMemcpy(h_vals, hamil_lancz[i].vals, num_Elem[i]*sizeof(double), cudaMemcpyDeviceToHost);

        if (status[i] != cudaSuccess)
        {
            cout<<"Error copying to h_vals: "<<cudaGetErrorString(status[i])<<endl;
        }

        status[i] = cudaMemcpy(h_rows, hamil_lancz[i].rows, num_Elem[i]*sizeof(int), cudaMemcpyDeviceToHost);

        if (status[i] != cudaSuccess)
        {
            cout<<"Error copying to h_rows: "<<cudaGetErrorString(status[i])<<endl;
        }

        status[i] = cudaMemcpy(h_cols, hamil_lancz[i].cols, num_Elem[i]*sizeof(int), cudaMemcpyDeviceToHost);

        if (status[i] != cudaSuccess)
        {
            cout<<"Error copying to h_cols: "<<cudaGetErrorString(status[i])<<endl;
        }


        if(i == 0)
        {
            ofstream fout;
            fout.open("hamiltonian.log");
            for(int j = 0; j < num_Elem[i]; j++)
            {
                //fout<<h_index[j]<<" - ";
                fout<<h_rows[j]<<" "<<h_cols[j];
                fout<<" "<<h_vals[j]<<std::endl;

            }
            fout.close();
        }

        free(h_rows);
        free(h_cols);
        free(h_vals);

        cudaStreamSynchronize(stream[i]);
        cudaFree(vals_buffer[i]);
        free(Bond[i]);
    }
    cudaDeviceSynchronize();

    //----Free all the array storage to avoid memory leaks---------

    free(d_basis_Position);
    free(d_Bond);
    free(d_basis);
    free(basis);
    free(basis_Position);
    free(d_H);
    free(bpg);
    free(tpb);
    free(vals_buffer);
    memcpy(count_array, num_Elem, how_many);
    //cout<<num_Elem[0]<<endl;
    free(num_Elem);
    //cudaFree(d_num_Elem);
    //return num_Elem;
}

/*Function: FullToCOO - takes a full sparse matrix and transforms it into COO format
Inputs - num_Elem - the total number of nonzero elements
H_vals - the Hamiltonian values
H_pos - the Hamiltonian positions
hamil_Values - a 1D array that will store the values for the COO form

*/
__global__ void FullToCOO(int num_Elem, float* H_vals, double* hamil_Values, int dim)
{

    int i = threadIdx.x + blockDim.x*blockIdx.x;

    if (i < num_Elem)
    {

        hamil_Values[i] = H_vals[i];


    }
}
;
