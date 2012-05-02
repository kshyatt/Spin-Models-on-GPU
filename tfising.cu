#include "hamiltonian.h"
__device__ float HOffBondXTFI(const int si, const int bra, const float JJ)
{

    float valH;
    //int S0, S1;
    //int T0, T1;

    valH = JJ*0.5; //contribution from the J part of the Hamiltonian

    return valH;

}

__device__ float HDiagPartTFI(const int bra, int lattice_Size, int2* d_Bond, const float JJ)
{

    int S0b,S1b ; //spins (bra
    int T0,T1; //site
    //int P0, P1, P2, P3; //sites for plaquette (Q)
    //int s0p, s1p, s2p, s3p;
    float valH = 0.f;

    for (int Ti=0; Ti<lattice_Size; Ti++)
    {

        T0 = (d_Bond[Ti]).x; //lower left spin
        S0b = (bra>>T0)&1;
        //if (T0 != Ti) cout<<"Square error 3\n";
        T1 = (d_Bond[Ti]).y; //first bond
        S1b = (bra>>T1)&1; //unpack bra
        valH += JJ*(S0b-0.5)*(S1b-0.5);

    }//T0

    //cout<<bra<<" "<<valH<<endl;

    return valH;

}//HdiagPart

__global__ void FillDiagonalsTFI(int* d_basis, int dim, int* H_set, int* H_rows, int* H_cols, float* H_vals, int* d_Bond, int lattice_Size, float JJ)
{

    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int site = threadIdx.x%(lattice_Size);

    unsigned int tempi;

    __shared__ int2 tempbond[18];
    //int3 tempbond[16];

    if (row < dim)
    {
        tempi = d_basis[row];
        (tempbond[site]).x = d_Bond[site];
        (tempbond[site]).y = d_Bond[lattice_Size + site];

        H_vals[row] = HDiagPartTFI(tempi, lattice_Size, tempbond, JJ);
        H_rows[row] = row;
        H_cols[row] = row;
        H_set[row] = 1;

    }

    else
    {
        H_rows[row] = 2*dim;
        H_cols[row] = 2*dim;
        H_set[row] = 0;
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

__global__ void FillSparseTFI(int* d_basis_Position, int* d_basis, int dim, int* H_rows, int* H_cols, float* H_vals, int* H_set, int* d_Bond, const int lattice_Size, const float JJ, const float h)
{

    int ii = ( blockDim.x / ( 2 * lattice_Size ) ) * blockIdx.x + threadIdx.x / ( 2 * lattice_Size ) + blockIdx.y* gridDim.x * blockDim.x / (2 * lattice_Size);
    int T0 = threadIdx.x % ( 2 * lattice_Size );

//    printf("%d \n", ii);

	#if __CUDA_ARCH__ < 200
    	const int array_size = 512;
	#elif __CUDA_ARCH__ >= 200
    	const int array_size = 1024;
	#else
		#error Could not detect GPU architecture
	#endif

    int count;
	__shared__ int2 tempbond[20];
    __shared__ int temppos[ array_size ];
    __shared__ float tempval[ array_size ];

    int stride = 2 * lattice_Size;
    int site = T0 % ( lattice_Size );
    count = 0;
    int rowtemp;
    uint tempi;
    __shared__ uint tempod[array_size];

	tempi = d_basis[ii];


    //int start = (bool)(dim%array_size) ? (dim/array_size + 1)*array_size : dim/array_size;
    int start = ( bool )( dim % 512 ) ? ( dim / 512 + 1 ) * 512 : dim ; 
    bool compare;

    if( ii < dim )
    {
        if ( T0 < 2 * lattice_Size )
        {

          //Putting bond info in shared memory
            (tempbond[site]).x = d_Bond[site];
            (tempbond[site]).y = d_Bond[lattice_Size + site];

            __syncthreads();

            //-----------------Horizontal bond ---------------
            tempod[threadIdx.x] = tempi;
            tempod[threadIdx.x] ^= ( 1<<(tempbond[site].x) );//toggles both spins on the bond

            compare = (d_basis_Position[tempod[threadIdx.x]] > ii); //don't double count elements

            temppos[threadIdx.x] = (compare) ? d_basis_Position[tempod[threadIdx.x]] : dim;
            tempval[threadIdx.x] = HOffBondXTFI(site, tempi, JJ);

            count += (int)compare;
            rowtemp = (T0/lattice_Size) ? ii : temppos[threadIdx.x];
            rowtemp = (compare) ? rowtemp : dim + 1;

            //H_index[ ii*stride + 2*site + (T0/lattice_Size) + start ] = rowtemp*dim + H_cols[ ii*stride + 2*site + (T0/lattice_Size) + start ];

            //----sigma^x term ------------------------------------------

            /*temppos[ threadIdx.x ] = ( ii ^ ( 1 << site ) );// & ( 1 << site ); //flip the site-th bit of row - applying the sigma_x operator
            compare = ( temppos[ threadIdx.x ] > ii ) && ( temppos[ threadIdx.x ] < dim );
            temppos[ threadIdx.x ] = compare ? temppos[ threadIdx.x ] : dim + 1;
            tempval[ threadIdx.x ] = 0.5 * h;
            
            rowtemp = ( T0 / lattice_Size ) ? ii : temppos[ threadIdx.x ];
            rowtemp = compare ? rowtemp : dim + 1;
            temppos[ threadIdx.x ] = ( T0 / lattice_Size) ? temppos[ threadIdx.x ] : ii;
            temppos[ threadIdx.x ] = compare ? temppos[threadIdx.x] : dim + 1;

            //count = (H_vals[ idx(ii, 2*site + (T0/lattice_Size) + start, stride) ] < 1e-8) ? (int)compare : 0;
            //count = (bool)H_set[ idx(ii, 2*site + (T0/lattice_Size) + start, stride) ] ? 0 : (int)compare; 
            count += ( int )compare;*/
            //----Putting everything back into GPU main memory-----------

            H_vals[ ii*stride + 2 * site + ( T0 / lattice_Size ) + start ] = tempval[ threadIdx.x ]; 
            H_cols[ ii*stride + 2 * site + ( T0 / lattice_Size ) + start ] = temppos[ threadIdx.x ];
            H_rows[ ii*stride + 2 * site + ( T0 / lattice_Size ) + start ] = rowtemp;
            H_set[ ii*stride + 2 * site + ( T0 / lattice_Size ) + start ] = count;
            //atomicExch(&H_set[ idx(ii, 2*site + (T0/lattice_Size) + start, stride) ], 1); 

            //atomicAdd( &num_Elem[ index ], count);
        }
    }//end of ii
}//end of FillSparse

