/*! 
    \file xy.cu
    \brief Functions to generate Hamiltonians for the XY model
*/

#include "hamiltonian.h"
__device__ float HOffBondXXY(const int si, const int bra, const float JJ)
{

    float valH;
    //int S0, S1;
    //int T0, T1;

    valH = JJ*0.5; //contribution from the J part of the Hamiltonian

    return valH;

}

__device__ float HOffBondYXY(const int si, const int bra, const float JJ)
{

    float valH;
    //int S0, S1;
    //int T0, T1;

    valH = JJ*0.5; //contribution from the J part of the Hamiltonian

    return valH;


}

__device__ float HDiagPartXY(const int bra, int latticeSize, int3* d_Bond, const float JJ)
{

    return 0.f;

}//HdiagPart


__global__ void FillDiagonalsXY(int* d_basis, f_hamiltonian H, int* d_Bond, parameters data)
{

    int row = blockIdx.x*blockDim.x + threadIdx.x;

    H.vals[row] = 0.f;
    H.rows[row] = 2*H.sectorDim;
    H.cols[row] = 2*H.sectorDim;
    H.set[row] = 0;

}

/* Function FillSparse: this function takes the empty Hamiltonian arrays and fills them up. Each thread in x handles one ket |i>, and each thread in y handles one site T0
Inputs: d_basisPosition - position information about the basis
d_basis - other basis infos
d_dim - the number of kets
H_sort - an array that will store the Hamiltonian
d_Bond - the bond information
d_latticeSize - the number of lattice sites
JJ - the coupling parameter

*/

__global__ void FillSparseXY(int* d_basisPosition, int* d_basis, f_hamiltonian H, int* d_Bond, parameters data, int offset)
{

    int dim = H.sectorDim;
    int latticeSize = data.nsite;
    int ii = (blockDim.x/(2*latticeSize))*(blockIdx.x + offset) + threadIdx.x/(2*latticeSize);
    int T0 = threadIdx.x%(2*latticeSize);

#if __CUDA_ARCH__ < 200
    const int arraySize = 512;
#elif __CUDA_ARCH__ >= 200
    const int arraySize = 1024;
#else
#error Could not detect GPU architecture
#endif

    __shared__ int3 tempBond[32];
    int count;
    __shared__ int tempPos[arraySize];
    __shared__ float tempVal[arraySize];
    //__shared__ uint tempi[arraySize];
    unsigned int tempi;
    __shared__ unsigned int tempod[arraySize];

    int stride = 4*latticeSize;
    //int tempcount;
    int site = T0%(latticeSize);
    count = 0;
    int rowTemp;

    int braSector;

    int start = (bool)(dim%arraySize) ? (dim/arraySize + 1)*arraySize : dim/arraySize;

    int s;
    //int si, sj;//sk,sl; //spin operators
    //unsigned int tempi;// tempod; //tempj;
    //cuDoubleComplex tempD;

    __syncthreads();

    bool compare;

    if( ii < dim )
    {
        if (T0 < 2*latticeSize)
        {
            tempi = d_basis[ii];
            //Putting bond info in shared memory
            (tempBond[site]).x = d_Bond[site];
            (tempBond[site]).y = d_Bond[latticeSize + site];
            (tempBond[site]).z = d_Bond[2*latticeSize + site];

            __syncthreads();

            //Horizontal bond ---------------
            s = (tempBond[site]).x;
            tempod[threadIdx.x] = tempi;
            braSector = (tempi & (1 << s)) >> s;
            tempod[threadIdx.x] ^= (1<<s);
            s = (tempBond[site]).y;
            braSector ^= (tempi & (1 << s)) >> s;
            tempod[threadIdx.x] ^= (1<<s);

            compare = (d_basisPosition[tempod[threadIdx.x]] != -1) && braSector;
            compare &= (d_basisPosition[tempod[threadIdx.x]] > ii);
            tempPos[threadIdx.x] = (compare) ? d_basisPosition[tempod[threadIdx.x]] : dim;
            tempVal[threadIdx.x] = HOffBondXXY(site, tempi, data.J1);

            count += (int)compare;
            rowTemp = (T0/latticeSize) ? ii : tempPos[threadIdx.x];
            rowTemp = (compare) ? rowTemp : 2*dim;

            H.vals[ ii*stride + 4*site + (T0/latticeSize)+ start ] = tempVal[threadIdx.x]; //(T0/latticeSize) ? tempVal[threadIdx.x] : cuConj(tempVal[threadIdx.x]);
            H.cols[ ii*stride + 4*site + (T0/latticeSize) + start ] = (T0/latticeSize) ? tempPos[threadIdx.x] : ii;
            H.rows[ ii*stride + 4*site + (T0/latticeSize) + start ] = rowTemp;

            H.set[ ii*stride + 4*site + (T0/latticeSize) + start ] = (int)compare;

            //Vertical bond -----------------
            s = (tempBond[site]).x;
            tempod[threadIdx.x] = tempi;
            braSector = (tempi & (1 << s)) >> s;
            tempod[threadIdx.x] ^= (1<<s);
            s = (tempBond[site]).z;
            braSector ^= (tempi & (1 << s)) >> s;
            tempod[threadIdx.x] ^= (1<<s);

            compare = (d_basisPosition[tempod[threadIdx.x]] != -1) && braSector;
            compare &= (d_basisPosition[tempod[threadIdx.x]] > ii);
            tempPos[threadIdx.x] =  (compare) ? d_basisPosition[tempod[threadIdx.x]] : dim;
            tempVal[threadIdx.x] = HOffBondYXY(site,tempi, data.J1);

            count += (int)compare;
            rowTemp = (T0/latticeSize) ? ii : tempPos[threadIdx.x];
            rowTemp = (compare) ? rowTemp : 2*dim;

            H.vals[ ii*stride + 4*site + 2 + (T0/latticeSize) + start ] =  tempVal[threadIdx.x]; // (T0/latticeSize) ? tempVal[threadIdx.x] : cuConj(tempVal[threadIdx.x]);
            H.cols[ ii*stride + 4*site + 2 + (T0/latticeSize) + start ] = (T0/latticeSize) ? tempPos[threadIdx.x] : ii;
            H.rows[ ii*stride + 4*site + 2 + (T0/latticeSize) + start ] = rowTemp;

            H.set[ ii*stride + 4*site + 2 + (T0/latticeSize) + start ] = (int)compare;
        }
    }//end of ii
}//end of FillSparse
