#include "hamiltonian.h"
__device__ float HOffBondXTFI(const int si, const int bra, const float JJ)
{

    float valH;
    //int S0, S1;
    //int T0, T1;

    valH = JJ; //contribution from the J part of the Hamiltonian

    return valH;

}

__device__ float HDiagPartTFI(const int bra, int lattice_Size, int2* d_Bond, const float JJ)
{

    //int S0b,S1b ; //spins (bra
    //int T0,T1; //site
    //int P0, P1, P2, P3; //sites for plaquette (Q)
    //int s0p, s1p, s2p, s3p;
    float valH = 0.f;
    int temp = 0;
    int total = 0;


    for (int Ti=0; Ti<lattice_Size; Ti++)
    {
        temp = (bra>>Ti)&1;
        total += -1 + 2*temp;
    }

    /*for (int Ti=0; Ti<lattice_Size; Ti++)
    {

        T0 = (d_Bond[Ti]).x; //lower left spin
        S0b = (bra>>T0)&1;
        //if (T0 != Ti) cout<<"Square error 3\n";
        T1 = (d_Bond[Ti]).y; //first bond
        S1b = (bra>>T1)&1; //unpack bra
        valH += JJ*(S0b-0.5)*(S1b-0.5);

    }//T0*/

    //cout<<bra<<" "<<valH<<endl;
    valH = JJ*total*0.5;
    return valH;

}//HdiagPart

__global__ void FillDiagonalsTFI(int* d_basis, f_hamiltonian H, int* d_Bond, parameters data)
{

    int lattice_Size = data.nsite;
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int site = threadIdx.x%(lattice_Size);

    unsigned int tempi;

    __shared__ int2 tempbond[18];
    //int3 tempbond[16];

    if (row < H.sectordim)
    {
        tempi = d_basis[row];
        (tempbond[site]).x = d_Bond[site];
        (tempbond[site]).y = d_Bond[lattice_Size + site];

        H.vals[row] = HDiagPartTFI(tempi, lattice_Size, tempbond, data.J1);
        H.rows[row] = row;
        H.cols[row] = row;
        H.set[row] = 1;

    }

    else
    {
        H.rows[row] = H.sectordim + 1;
        H.cols[row] = H.sectordim + 1;
        H.set[row] = 0;
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

__global__ void FillSparseTFI(int* d_basis_Position, int* d_basis, f_hamiltonian H, int* d_Bond, parameters data, int offset)
{

    int lattice_Size = data.nsite;
    int dim = H.sectordim;
    int ii = ( blockDim.x / ( 2 * lattice_Size ) ) * blockIdx.x + threadIdx.x / ( 2 * lattice_Size ) + offset;// + blockIdx.y* gridDim.x * blockDim.x / (2 * lattice_Size);
    int T0 = threadIdx.x % ( 2 * lattice_Size );

#if __CUDA_ARCH__ < 200
    const int array_size = 512;
#elif __CUDA_ARCH__ >= 200
    const int array_size = 1024;
#else
#error Could not detect GPU architecture
#endif

    //__shared__ int2 tempbond[20];
    __shared__ int temppos[ array_size ];
    __shared__ float tempval[ array_size ];

    int stride = 2 * lattice_Size;
    int site = T0 % ( lattice_Size );
    int rowtemp;
    uint tempi;
    __shared__ uint tempod[array_size];

    //int start = (bool)(dim%array_size) ? (dim/array_size + 1)*array_size : dim/array_size;
    int start = ( bool )( dim % 512 ) ? ( dim / 512 + 1 ) * 512 : dim ;
    bool compare;

    if( ii < dim )
    {
        tempi = d_basis[ii];
        if ( T0 < 2 * lattice_Size )
        {

            //Putting bond info in shared memory
            // (tempbond[site]).x = d_Bond[site];
            // (tempbond[site]).y = d_Bond[lattice_Size + site];

            // __syncthreads();

            tempod[threadIdx.x] = tempi;

            //-----------------Horizontal bond ---------------
            temppos[ threadIdx.x ] = ( tempod[threadIdx.x] ^ ( 1 << site ) );// & ( 1 << site ); //flip the site-th bit of row - applying the sigma_x operator
            compare = ( temppos[ threadIdx.x ] > ii ) && ( temppos[ threadIdx.x ] < dim );
            temppos[ threadIdx.x ] = compare ? temppos[ threadIdx.x ] : dim + 1;
            tempval[ threadIdx.x ] = HOffBondXTFI(site, tempi, data.J2);

            rowtemp = ( T0 / lattice_Size ) ? ii : temppos[ threadIdx.x ];
            rowtemp = compare ? rowtemp : dim + 1;
            temppos[ threadIdx.x ] = ( T0 / lattice_Size) ? temppos[ threadIdx.x ] : ii;
            temppos[ threadIdx.x ] = compare ? temppos[threadIdx.x] : dim + 1;

            //----Putting everything back into GPU main memory-----------

            H.vals[ ii*stride + 2 * site + ( T0 / lattice_Size ) + start ] = tempval[ threadIdx.x ];
            H.cols[ ii*stride + 2 * site + ( T0 / lattice_Size ) + start ] = temppos[ threadIdx.x ];
            H.rows[ ii*stride + 2 * site + ( T0 / lattice_Size ) + start ] = rowtemp;
            H.set[ ii*stride + 2 * site + ( T0 / lattice_Size ) + start ] = (int)compare;

        }
    }//end of ii
}//end of FillSparse
