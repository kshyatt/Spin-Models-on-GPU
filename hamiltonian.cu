#include"hamiltonian.h"

/* NOTE: this function uses FORTRAN style matrices, where the values and positions are stored in a ONE dimensional array! Don't forget this! */


/*int main()
{


    int** Bond;

    int how_many = 30;
    Bond = (int**)malloc(how_many*sizeof(int*));
    d_hamiltonian* hamil_lancz = (d_hamiltonian*)malloc(how_many*sizeof(d_hamiltonian));
    int* nsite = (int*)malloc(how_many*sizeof(int));
    int* Sz = (int*)malloc(how_many*sizeof(int));
    float* JJ = (float*)malloc(how_many*sizeof(float));
    int* model_type = (int*)malloc(how_many*sizeof(int));

    for(int i = 0; i < how_many; i++)
    {

        nsite[i] = 16;
        Bond[i] = (int*)malloc(3*nsite[i]*sizeof(int));
        Fill_Bonds_16B(Bond[i]);


        Sz[i] = 0;
        JJ[i] = 1.f;
        model_type[i] = 0;
    }


    int dim;

    int* num_Elem = ConstructSparseMatrix(how_many, model_type, nsite, Bond, hamil_lancz, JJ, Sz );

    return 0;
}*/

__host__ __device__ int idx(int i, int j, int lda)
{

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
__host__ int GetBasis(int dim, int lattice_Size, int Sz, int basis_Position[], int basis[])
{
    unsigned int temp = 0;
    int realdim = 0;

    for (unsigned int i1=0; i1<dim; i1++)
    {
        temp = 0;
        basis_Position[i1] = -1;
        for (int sp =0; sp<lattice_Size; sp++)
        {
            temp += (i1>>sp)&1;
        } //unpack bra
        if (temp==(lattice_Size/2+Sz) ){
        basis[realdim] = i1;
        basis_Position[i1] = realdim;
        realdim++;
        //cout<<basis[realdim]<<" "<<basis_Position[i1]<<endl;
        }
    }

    return realdim;

}

/* Function HOffBondX
Inputs: si - the spin operator in the x direction
bra - the state
JJ - the coupling constant
Outputs: valH - the value of the Hamiltonian

*/

__device__ float HOffBondX(const int si, const int bra, const float JJ)
{

    float valH;
    //int S0, S1;
    //int T0, T1;

    valH = JJ*0.5; //contribution from the J part of the Hamiltonian

    return valH;

}

__device__ float HOffBondY(const int si, const int bra, const float JJ)
{

    float valH;
    //int S0, S1;
    //int T0, T1;

    valH = JJ*0.5; //contribution from the J part of the Hamiltonian

    return valH;


}

__device__ float HDiagPart(const int bra, int lattice_Size, int3* d_Bond, const float JJ)
{

    int S0b,S1b ; //spins (bra
    int T0,T1; //site
    //int P0, P1, P2, P3; //sites for plaquette (Q)
    //int s0p, s1p, s2p, s3p;
    float valH = 0.f;

    for (int Ti=0; Ti<lattice_Size; Ti++)
    {
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

__host__ void ConstructSparseMatrix(const int how_many, int* model_Type, int* lattice_Size, int** Bond, d_hamiltonian*& hamil_lancz, float* JJ, int* Sz, int*& count_array, int device)
{
    cudaFree(0);
    cudaSetDevice(device);

    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    //float time;

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

    cudaStream_t stream[how_many];

    cudaError_t status[how_many];

    int* d_num_Elem;
    //cudaEventRecord(start,0);
    cudaMalloc(&d_num_Elem, how_many*sizeof(int));
    //cudaEventRecord(stop,0);
    //cudaEventSynchronize(stop);
    //cudaEventElapsedTime(&time, start, stop);
    //std::cout<<"Time to maloc d_num_Elem: "<<time<<std::endl;

    for(int i = 0; i<how_many; i++)
    {
        num_Elem[i] = 0;
        stride[i] = 4*lattice_Size[i] + 1;

        d_H[i].fulldim = 2;
        for (int ch=1; ch<lattice_Size[i]; ch++) d_H[i].fulldim *= 2;

        basis_Position[i] = (int*)malloc(d_H[i].fulldim*sizeof(int));
        basis[i] = (int*)malloc(d_H[i].fulldim*sizeof(int));

        d_H[i].sectordim = GetBasis(d_H[i].fulldim, lattice_Size[i], Sz[i], basis_Position[i], basis[i]);
        //cudaEventRecord(start, 0);
        status[i] = cudaMalloc(&d_basis_Position[i], d_H[i].fulldim*sizeof(int));
        if (status[i] != cudaSuccess)
        {
            cout<<"Error allocating "<<i<<"th d_basis_Position array: "<<cudaGetErrorString(status[i])<<endl;
        }
        //cudaEventRecord(stop,0);
        //cudaEventSynchronize(stop);
        //cudaEventElapsedTime(&time, start, stop);
        //std::cout<<"Time to malloc d_basis_Position: "<<time<<std::endl;
        //cudaEventRecord(start,0);
        status[i] = cudaMalloc(&d_basis[i], d_H[i].sectordim*sizeof(int));

        if (status[i] != cudaSuccess)
        {
            cout<<"Error allocating "<<i<<"th d_basis array: "<<cudaGetErrorString(status[i])<<endl;
        }

        //cudaEventRecord(stop,0);
        //cudaEventSynchronize(stop);
        //cudaEventElapsedTime(&time, start, stop);
        //std::cout<<"Time to malloc d_basis: "<<time<<std::endl;
        status[i] = cudaStreamCreate(&stream[i]);

        if (status[i] != cudaSuccess)
        {
            cout<<"Error creating "<<i<<"th stream: "<<cudaGetErrorString(status[i])<<endl;
        }

        num_Elem[i] = d_H[i].sectordim;
	//cudaEventRecord(start,0);
        status[i] = cudaMemcpy(d_num_Elem, num_Elem, how_many*sizeof(int), cudaMemcpyHostToDevice);
        if (status[i] != cudaSuccess)
        {
            cout<<"Error copying num_Elem array to device in "<<i<<"th stream: "<<cudaGetErrorString(status[i])<<endl;
        }
	//cudaEventRecord(stop,0);
        //cudaEventSynchronize(stop);
        //cudaEventElapsedTime(&time, start, stop);
        //std::cout<<"Time to copy num_Elem: "<<time<<std::endl;

    } // can insert more code in here to handle model type later

    for(int i = 0; i<how_many; i++)
    {
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

        padded_dim[i] = (bool)(d_H[i].sectordim%512) ? (d_H[i].sectordim/512 + 1)*512 : d_H[i].sectordim;
        raw_size[i] = padded_dim[i] + ((stride[i] - 1)*d_H[i].sectordim);
        raw_size[i] = (bool)(raw_size[i]%2048) ? (raw_size[i]/2048 + 1)*2048 : raw_size[i];

        //cudaEventRecord(start,0);
        status[i] = cudaMalloc(&d_H[i].rows, raw_size[i]*sizeof(int));
        if (status[i] != cudaSuccess)
        {
            cout<<"Error creating "<<i<<"th rows array: "<<cudaGetErrorString(status[i])<<endl;
        }
        //cudaEventRecord(stop,0);
        //cudaEventSynchronize(stop);
        //cudaEventElapsedTime(&time, start, stop);
        //std::cout<<"Time to malloc d_H.rows: "<<time<<std::endl;
        //cudaEventRecord(start,0);
        status[i] = cudaMalloc(&d_H[i].cols, raw_size[i]*sizeof(int));
        if (status[i] != cudaSuccess)
        {
            cout<<"Error creating "<<i<<"th cols array: "<<cudaGetErrorString(status[i])<<endl;
        }
        //cudaEventRecord(stop,0);
        //cudaEventSynchronize(stop);
        //cudaEventElapsedTime(&time, start, stop);
        //std::cout<<"Time to malloc d_H.cols: "<<time<<std::endl;
        //cudaEventRecord(start,0);
        status[i] = cudaMalloc(&d_H[i].vals, raw_size[i]*sizeof(float));
        if (status[i] != cudaSuccess)
        {
            cout<<"Error creating "<<i<<"th values array: "<<cudaGetErrorString(status[i])<<endl;
        }
        //cudaEventRecord(stop,0);
        //cudaEventSynchronize(stop);
        //cudaEventElapsedTime(&time, start, stop);
        //std::cout<<"Time to malloc d_H.vals: "<<time<<std::endl;

        /*status[i] = cudaMalloc(&d_H[i].index, raw_size[i]*sizeof(int));
        if (status[i] != cudaSuccess)
        {
            cout<<"Error creating "<<i<<"th index array: "<<cudaGetErrorString(status[i])<<endl;
        }*/
        status[i] = cudaMalloc(&d_Bond[i], 3*lattice_Size[i]*sizeof(int));
        if (status[i] != cudaSuccess)
        {
            cout<<"Error creating "<<i<<"th bonds array: "<<cudaGetErrorString(status[i])<<endl;
        }

        status[i] = cudaMemcpyAsync(d_Bond[i], Bond[i], 3*lattice_Size[i]*sizeof(int), cudaMemcpyHostToDevice, stream[i]);

        if (status[i] != cudaSuccess)
        {
            cout<<"Error copying "<<i<<"th bonds array: "<<cudaGetErrorString(status[i])<<endl;
        }

        tpb[i].x = lattice_Size[i];
        do
        {
            tpb[i].x *= 2;
        }
        while(tpb[i].x < 512);


        bpg[i].x = (bool)((stride[i]-1)*d_H[i].sectordim)%tpb[i].x ? ((((stride[i]-1)*d_H[i].sectordim)/tpb[i].x) + 1) : ((stride[i]-1)*d_H[i].sectordim)/tpb[i].x;

        cout<<bpg[i].x<<endl;

        status[i] = cudaStreamSynchronize(stream[i]);

        if (status[i] != cudaSuccess)
        {
            cout<<"Error synchronizing "<<i<<"th stream: "<<cudaGetErrorString(status[i])<<endl;
        }

        FillDiagonals<<<d_H[i].sectordim/512 + 1, 512, device, stream[i]>>>(d_basis[i], d_H[i].sectordim, d_H[i].index, d_H[i].rows, d_H[i].cols, d_H[i].vals, d_Bond[i], lattice_Size[i], JJ[i]);
    }
    for(int i = 0; i < how_many; i++){
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


        FillSparse<<<bpg[i].x, tpb[i].x, device, stream[i]>>>(d_basis_Position[i], d_basis[i], d_H[i].sectordim, d_H[i].index, d_H[i].rows, d_H[i].cols, d_H[i].vals, d_Bond[i], lattice_Size[i], JJ[i], d_num_Elem, i);

        status[i] = cudaPeekAtLastError();
        if (status[i] != cudaSuccess)
        {
            cout<<"Error in "<<i<<"th stream: "<<cudaGetErrorString(status[i])<<endl;
        }

    }

    /*hamstruct* d_H_sort;
    status2 = cudaMalloc(&d_H_sort, *vdim*stride*sizeof(hamstruct));

    if (status2 != cudaSuccess){
    	std::cout<<"Allocating d_H_sort failed! Error: ";
    	std::cout<<cudaGetErrorString( status1 )<<std::endl;
    	return 1;
    }*/


    cudaThreadSynchronize();

    //int* num_ptr;
    //cudaGetSymbolAddress((void**)&num_ptr, (const char*)"d_num_Elem");

    cudaMemcpy(num_Elem, d_num_Elem, how_many*sizeof(int), cudaMemcpyDeviceToHost);
    //std::cout<<num_Elem<<std::endl;
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

    for(int i = 0; i<how_many; i++)
    {

        sortEngine_t engine;
        sortStatus_t sortstatus = sortCreateEngine("sort/sort/src/cubin64/", &engine);

        MgpuSortData sortdata;

        sortdata.AttachKey((uint*)d_H[i].rows);
        //sortdata.AttachKey((uint*)d_H[i].index);
        //sortdata.AttachVal(0, (uint*)d_H[i].rows);
        sortdata.AttachVal(0, (uint*)d_H[i].cols);
        sortdata.AttachVal(1, (uint*)d_H[i].vals);

        sortnumber[i] = raw_size[i];

        sortdata.Alloc(engine, sortnumber[i], 2);

        sortdata.firstBit = 0;
        sortdata.endBit = 2*lattice_Size[i];

        sortArray(engine, &sortdata);

        /*thrust::device_ptr<int> sort_key_ptr(d_H_rows);
        thrust::device_ptr<int> sort_val_ptr(d_H_cols);

        thrust::sort_by_key(sort_key_ptr, sort_key_ptr + *vdim*stride, sort_val_ptr);*/

        status[i] = cudaMalloc(&hamil_lancz[i].vals, num_Elem[i]*sizeof(cuDoubleComplex));
        if (status[i] != cudaSuccess)
        {
            cout<<"Error allocating "<<i<<"th lancz values array: "<<cudaGetErrorString(status[i])<<endl;
        }
        status[i] = cudaMalloc(&hamil_lancz[i].rows, num_Elem[i]*sizeof(int));
        if (status[i] != cudaSuccess)
        {
            cout<<"Error allocating "<<i<<"th lancz rows array: "<<cudaGetErrorString(status[i])<<endl;
        }
        status[i] = cudaMalloc(&hamil_lancz[i].cols, num_Elem[i]*sizeof(int));
        if (status[i] != cudaSuccess)
        {
            cout<<"Error allocating "<<i<<"th lancz cols array: "<<cudaGetErrorString(status[i]);
        }
        //half of these are commented out to get check indexed sort vs nonindexed
        cudaMemcpy(hamil_lancz[i].rows, (int*)sortdata.keys[0], num_Elem[i]*sizeof(int), cudaMemcpyDeviceToDevice);

        //cudaMemcpy(hamil_lancz[i].rows, (int*)sortdata.values1[0], num_Elem[i]*sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(hamil_lancz[i].cols, (int*)sortdata.values1[0], num_Elem[i]*sizeof(int), cudaMemcpyDeviceToDevice);

        //cudaMemcpy(hamil_lancz[i].cols, (int*)sortdata.values2[0], num_Elem[i]*sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMalloc(&vals_buffer[i], num_Elem[i]*sizeof(float));

        cudaMemcpy(vals_buffer[i], (float*)sortdata.values2[0], num_Elem[i]*sizeof(float), cudaMemcpyDeviceToDevice);
        
        //cudaMemcpy(vals_buffer[i], (float*)sortdata.values3[0], num_Elem[i]*sizeof(float), cudaMemcpyDeviceToDevice);
        
        FullToCOO<<<num_Elem[i]/1024 + 1, 1024>>>(num_Elem[i], vals_buffer[i], hamil_lancz[i].vals, d_H[i].sectordim);
         
        //int* h_index = (int*)malloc(num_Elem[i]*sizeof(int));
       // status[i] = cudaMemcpy(h_index, d_H[i].index, num_Elem[i]*sizeof(int), cudaMemcpyDeviceToHost);

        sortReleaseEngine(engine);
        cudaFree(d_H[i].rows);
        cudaFree(d_H[i].cols);
        cudaFree(d_H[i].vals);
        //cudaFree(d_H[i].index);

        hamil_lancz[i].fulldim = d_H[i].fulldim;
        hamil_lancz[i].sectordim = d_H[i].sectordim;
        count_array[i] = num_Elem[i];
        /*cuDoubleComplex* h_vals = (cuDoubleComplex*)malloc(num_Elem[i]*sizeof(cuDoubleComplex));
        int* h_rows = (int*)malloc(num_Elem[i]*sizeof(int));
        int* h_cols = (int*)malloc(num_Elem[i]*sizeof(int));

        status[i] = cudaMemcpy(h_vals, hamil_lancz[i].vals, num_Elem[i]*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

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
                fout<<" "<<h_vals[j].x<<std::endl;

            }
            fout.close();
        }*/
        cudaStreamSynchronize(stream[i]);
        cudaFree(vals_buffer[i]);
        free(Bond[i]);
    }
    cudaDeviceSynchronize();
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
    cudaFree(d_num_Elem);
    //return num_Elem;
}

__global__ void FillDiagonals(int* d_basis, int dim, int* H_index, int* H_rows, int* H_cols, float* H_vals, int* d_Bond, int lattice_Size, float JJ)
{

    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int site = threadIdx.x%(lattice_Size);

    unsigned int tempi;

    __shared__ int3 tempbond[18];
    //int3 tempbond[16];

    if (row < dim)
    {
        tempi = d_basis[row];
        (tempbond[site]).x = d_Bond[site];
        (tempbond[site]).y = d_Bond[lattice_Size + site];
        (tempbond[site]).z = d_Bond[2*lattice_Size + site];

        H_vals[row] = HDiagPart(tempi, lattice_Size, tempbond, JJ);
        H_rows[row] = row;
        H_cols[row] = row;
        //H_index[row] = row + row*dim;

    }

    else
    {
        H_rows[row] = 2*dim;
        H_cols[row] = 2*dim;
        //H_index[row] = 2*dim + 4*dim*dim;
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

__global__ void FillSparse(int* d_basis_Position, int* d_basis, int dim, int* H_index, int* H_rows, int* H_cols, float* H_vals, int* d_Bond, const int lattice_Size, const float JJ, int* num_Elem, int index)
{

    int ii = (blockDim.x/(2*lattice_Size))*blockIdx.x + threadIdx.x/(2*lattice_Size);
    int T0 = threadIdx.x%(2*lattice_Size);

#if __CUDA_ARCH__ < 200
    const int array_size = 512;
#elif __CUDA_ARCH__ >= 200
    const int array_size = 1024;
#else
#error Could not detect GPU architecture
#endif

    __shared__ int3 tempbond[18];
    int count;
    __shared__ int temppos[array_size];
    __shared__ float tempval[array_size];
    //__shared__ uint tempi[array_size];
    uint tempi;
    __shared__ uint tempod[array_size];

    int stride = 4*lattice_Size;
    //int tempcount;
    int site = T0%(lattice_Size);
    count = 0;
    int rowtemp;

    int start = (bool)(dim%array_size) ? (dim/array_size + 1)*array_size : dim/array_size;

    int s;
    //int si, sj;//sk,sl; //spin operators
    //unsigned int tempi;// tempod; //tempj;
    //cuDoubleComplex tempD;

    tempi = d_basis[ii];

    __syncthreads();

    bool compare;

    if( ii < dim )
    {
        if (T0 < 2*lattice_Size)
        {
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
            s = (tempbond[site]).x;
            tempod[threadIdx.x] = tempi;
            tempod[threadIdx.x] ^= (1<<s);
            s = (tempbond[site]).y;
            tempod[threadIdx.x] ^= (1<<s);

            //tempod[threadIdx.x] ^= (1<<si); //toggle bit
            //tempod[threadIdx.x] ^= (1<<sj); //toggle bit

            compare = (d_basis_Position[tempod[threadIdx.x]] > ii);
            temppos[threadIdx.x] = (compare) ? d_basis_Position[tempod[threadIdx.x]] : dim;
            tempval[threadIdx.x] = HOffBondX(site, tempi, JJ);

            count += (int)compare;
            //tempcount = (T0/lattice_Size);
            rowtemp = (T0/lattice_Size) ? ii : temppos[threadIdx.x];
            rowtemp = (compare) ? rowtemp : 2*dim;

            H_vals[ idx(ii, 4*site + (T0/lattice_Size)+ start, stride) ] = tempval[threadIdx.x]; //(T0/lattice_Size) ? tempval[threadIdx.x] : cuConj(tempval[threadIdx.x]);
            H_cols[ idx(ii, 4*site + (T0/lattice_Size) + start, stride) ] = (T0/lattice_Size) ? temppos[threadIdx.x] : ii;
            H_rows[ idx(ii, 4*site + (T0/lattice_Size) + start, stride) ] = rowtemp;

            //H_index[ idx(ii, 4*site + (T0/lattice_Size) + start, stride) ] = rowtemp*dim + H_cols[ idx(ii, 4*site + (T0/lattice_Size) + start, stride) ];

//Vertical bond -----------------
            s = (tempbond[site]).x;
            tempod[threadIdx.x] = tempi;
            tempod[threadIdx.x] ^= (1<<s);
            s = (tempbond[site]).z;
            tempod[threadIdx.x] ^= (1<<s);

            //tempod[threadIdx.x] ^= (1<<si); //toggle bit
            //tempod[threadIdx.x] ^= (1<<sj); //toggle bit

            compare = (d_basis_Position[tempod[threadIdx.x]] > ii);
            temppos[threadIdx.x] =  (compare) ? d_basis_Position[tempod[threadIdx.x]] : dim;
            tempval[threadIdx.x] = HOffBondY(site,tempi, JJ);

            count += (int)compare;
            //tempcount = (T0/lattice_Size);
            rowtemp = (T0/lattice_Size) ? ii : temppos[threadIdx.x];
            rowtemp = (compare) ? rowtemp : 2*dim;

            H_vals[ idx(ii, 4*site + 2 + (T0/lattice_Size) + start, stride) ] =  tempval[threadIdx.x]; // (T0/lattice_Size) ? tempval[threadIdx.x] : cuConj(tempval[threadIdx.x]);
            H_cols[ idx(ii, 4*site + 2 + (T0/lattice_Size) + start, stride) ] = (T0/lattice_Size) ? temppos[threadIdx.x] : ii;
            H_rows[ idx(ii, 4*site + 2 + (T0/lattice_Size) + start, stride) ] = rowtemp;

            //H_index[ idx(ii, 4*site + 2 + (T0/lattice_Size) + start, stride) ] = rowtemp*dim + H_cols[ idx(ii, 4*site + 2 + (T0/lattice_Size) + start, stride) ];
            __syncthreads();

            atomicAdd(&num_Elem[index], count);
        }
    }//end of ii
}//end of FillSparse

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



