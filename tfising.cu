/*!
    \file tfising.cu
    \brief Functions to generate a Hamiltonian for the Transverse Field Ising model
*/

#include "hamiltonian.h"
__device__ float HOffBondXTFI(const int si, const int bra, const float JJ)
{

    float valH;
    //int S0, S1;
    //int T0, T1;

    valH = JJ; //contribution from the J part of the Hamiltonian

    return valH;

}

__device__ float HDiagPartTFI1D(const int bra, int latticeSize, int3* d_Bond, const float JJ)
{

    int S0b,S1b ; //spins (bra
    int T0,T1; //site
    //int P0, P1, P2, P3; //sites for plaquette (Q)
    //int s0p, s1p, s2p, s3p;
    float valH = 0.f;

    for (int Ti=0; Ti<latticeSize; Ti++)
    {

        T0 = (d_Bond[Ti]).x; //lower left spin
        S0b = (bra>>T0)&1;
        //if (T0 != Ti) cout<<"Square error 3\n";
        T1 = (d_Bond[Ti]).y; //first bond
        S1b = (bra>>T1)&1; //unpack bra
        valH += JJ*(S0b-0.5)*(S1b-0.5);

    }//T0

    return valH;
}//HdiagPart

__device__ float HDiagPartTFI2D(const int bra, int latticeSize, int3* d_Bond, const float JJ)
{

    int S0b,S1b ; //spins (bra
    int T0,T1; //site
    //int P0, P1, P2, P3; //sites for plaquette (Q)
    //int s0p, s1p, s2p, s3p;
    float valH = 0.f;

    for (int Ti=0; Ti<latticeSize; Ti++)
    {

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

    return valH;
}//HdiagPart
__global__ void FillDiagonalsTFI(int* d_basis, f_hamiltonian H, int* d_Bond, parameters data)
{

    int latticeSize = data.nsite;
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int site = threadIdx.x%(latticeSize);

    unsigned int tempi;

    __shared__ int3 tempBond[18];
    //int3 tempBond[16];

    if (row < H.sectorDim)
    {
        tempi = d_basis[row];
        (tempBond[site]).x = d_Bond[site];
        (tempBond[site]).y = d_Bond[latticeSize + site];

        switch( data.dimension )
        {
            case 1 :
                H.vals[row] = HDiagPartTFI1D(tempi, latticeSize, tempBond, data.J1);
                break;
            case 2 :
                
                (tempBond[site].z) = d_Bond[2*latticeSize + site];
                H.vals[row] = HDiagPartTFI2D(tempi, latticeSize, tempBond, data.J1);
                break;
        }

        H.rows[row] = row;
        H.cols[row] = row;
        H.set[row] = 1;

    }

    else
    {
        H.rows[row] = H.sectorDim + 1;
        H.cols[row] = H.sectorDim + 1;
        H.set[row] = 0;
    }

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

__global__ void FillSparseTFI(int* d_basisPosition, int* d_basis, f_hamiltonian H, int* d_Bond, parameters data, int offset)
{

    int latticeSize = data.nsite;
    int dim = H.sectorDim;
    int ii = ( blockDim.x / ( 2 * latticeSize ) ) * blockIdx.x + threadIdx.x / ( 2 * latticeSize ) + offset;// + blockIdx.y* gridDim.x * blockDim.x / (2 * latticeSize);
    int T0 = threadIdx.x % ( 2 * latticeSize );

#if __CUDA_ARCH__ < 200
    const int array_size = 512;
#elif __CUDA_ARCH__ >= 200
    const int array_size = 1024;
#else
#error Could not detect GPU architecture
#endif

    //__shared__ int2 tempBond[20];
    __shared__ int tempPos[ array_size ];
    __shared__ float tempVal[ array_size ];

    int stride = 2 * latticeSize;
    int site = T0 % ( latticeSize );
    int rowTemp;
    __shared__ unsigned int tempi[array_size];
    __shared__ unsigned int tempod[array_size];

    //int start = (bool)(dim%array_size) ? (dim/array_size + 1)*array_size : dim/array_size;
    int start = ( bool )( dim % 512 ) ? ( dim / 512 + 1 ) * 512 : dim ;
    bool compare;

    if( ii < dim )
    {
        tempi[ threadIdx.x ] = d_basis[ ii ];
        if ( T0 < 2 * latticeSize )
        {

            //Putting bond info in shared memory
            // (tempBond[site]).x = d_Bond[site];
            // (tempBond[site]).y = d_Bond[latticeSize + site];

            // __syncthreads();

            tempod[ threadIdx.x ] = tempi[ threadIdx.x ];

            //-----------------Horizontal bond ---------------
            tempPos[ threadIdx.x ] = ( tempod[ threadIdx.x ] ^ ( 1 << site ) );
            //flip the site-th bit of row - applying the sigma_x operator
            compare = ( tempPos[ threadIdx.x ] > ii ) && ( tempPos[ threadIdx.x ] < dim );
            tempPos[ threadIdx.x ] = compare ? tempPos[ threadIdx.x ] : dim + 1;
            tempVal[ threadIdx.x ] = HOffBondXTFI(site, tempi[ threadIdx.x ], data.J2);

            rowTemp = ( T0 / latticeSize ) ? ii : tempPos[ threadIdx.x ];
            rowTemp = compare ? rowTemp : dim + 1;
            tempPos[ threadIdx.x ] = ( T0 / latticeSize) ? tempPos[ threadIdx.x ] : ii;
            tempPos[ threadIdx.x ] = compare ? tempPos[threadIdx.x] : dim + 1;

            //----Putting everything back into GPU main memory-----------

            H.vals[ ii*stride + 2 * site + ( T0 / latticeSize ) + start ] = tempVal[ threadIdx.x ];
            H.cols[ ii*stride + 2 * site + ( T0 / latticeSize ) + start ] = tempPos[ threadIdx.x ];
            H.rows[ ii*stride + 2 * site + ( T0 / latticeSize ) + start ] = rowTemp;
            H.set[ ii*stride + 2 * site + ( T0 / latticeSize ) + start ] = (int)compare;

        }
    }//end of ii
}//end of FillSparse
