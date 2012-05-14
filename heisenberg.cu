/*!
    \file heisenberg.cu
    \brief Functions to generate Hamiltonians for the Heisenberg model
*/

#include "hamiltonian.h"

__device__ float HOffBondXHeisenberg(const int si, const int bra, const float JJ)
{

    float valH;
    //int S0, S1;
    //int T0, T1;

    valH = JJ*0.5; //contribution from the J part of the Hamiltonian

    return valH;

}

__device__ float HOffBondYHeisenberg(const int si, const int bra, const float JJ)
{

    float valH;
    //int S0, S1;
    //int T0, T1;

    valH = JJ*0.5; //contribution from the J part of the Hamiltonian

    return valH;


}

__device__ float HDiagPartHeisenberg(const int bra, int lattice_Size, int3* d_Bond, const float JJ)
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

__global__ void FillDiagonalsHeisenberg(int* d_basis, f_hamiltonian H, int* d_Bond, parameters data)
{

    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int lattice_Size = data.nsite;
    int site = threadIdx.x%(lattice_Size);
    int dim = H.sectordim;

    unsigned int tempi;

    __shared__ int3 tempbond[32];

    if (row < dim)
    {
        tempi = d_basis[row];
        (tempbond[site]).x = d_Bond[site];
        (tempbond[site]).y = d_Bond[lattice_Size + site];
        (tempbond[site]).z = d_Bond[2*lattice_Size + site];

        H.vals[row] = HDiagPartHeisenberg(tempi, lattice_Size, tempbond, data.J1);
        H.rows[row] = row;
        H.cols[row] = row;
        H.set[row]  = 1;
    }

    else
    {
        H.rows[row] = 2*dim;
        H.cols[row] = 2*dim;
        H.set[row] = 0;
    }
}

__global__ void FillSparseHeisenberg(int* d_basis_Position, int* d_basis, f_hamiltonian H, int* d_Bond, parameters data, int offset)
{

    int lattice_Size = data.nsite;
    int dim = H.sectordim;
    int ii = (blockDim.x/(2*lattice_Size))*blockIdx.x + threadIdx.x/(2*lattice_Size) + offset*(blockDim.x/(2*lattice_Size));
    int T0 = threadIdx.x%(2*lattice_Size);

#if __CUDA_ARCH__ < 200
    const int array_size = 512;
#elif __CUDA_ARCH__ >= 200
    const int array_size = 1024;
#else
#error Could not detect GPU architecture
#endif

    __shared__ int3 tempbond[32];
    int count;
    __shared__ int temppos[array_size];
    __shared__ float tempval[array_size];
    //__shared__ uint tempi[array_size];
    unsigned int tempi;
    __shared__ unsigned int tempod[array_size];

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

    bool compare;

    if( ii < dim )
    {
        tempi = d_basis[ii];
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
            tempval[threadIdx.x] = HOffBondXHeisenberg(site, tempi, data.J1);

            count += (int)compare;
            //tempcount = (T0/lattice_Size);
            rowtemp = (T0/lattice_Size) ? ii : temppos[threadIdx.x];
            rowtemp = (compare) ? rowtemp : 2*dim;

            H.vals[ ii*stride + 4*site + (T0/lattice_Size) + start ] = tempval[threadIdx.x]; // (T0/lattice_Size) ? tempval[threadIdx.x] : cuConj(tempval[threadIdx.x]);
            H.cols[ ii*stride + 4*site + (T0/lattice_Size) + start ] = (T0/lattice_Size) ? temppos[threadIdx.x] : ii;
            H.rows[ ii*stride + 4*site + (T0/lattice_Size) + start ] = rowtemp;

            H.set[ ii*stride + 4*site + (T0/lattice_Size) + start ] = (int)compare;


//Vertical bond -----------------
            s = (tempbond[site]).x;
            tempod[threadIdx.x] = tempi;
            tempod[threadIdx.x] ^= (1<<s);
            s = (tempbond[site]).z;
            tempod[threadIdx.x] ^= (1<<s);

            //tempod[threadIdx.x] ^= (1<<si); //toggle bit
            //tempod[threadIdx.x] ^= (1<<sj); //toggle bit

            compare = (d_basis_Position[tempod[threadIdx.x]] > ii);
            temppos[threadIdx.x] = (compare) ? d_basis_Position[tempod[threadIdx.x]] : dim;
            tempval[threadIdx.x] = HOffBondYHeisenberg(site,tempi, data.J1);

            count += (int)compare;
            //tempcount = (T0/lattice_Size);
            rowtemp = (T0/lattice_Size) ? ii : temppos[threadIdx.x];
            rowtemp = (compare) ? rowtemp : 2*dim;

            H.vals[ ii*stride + 4*site + 2 + (T0/lattice_Size) + start ] = tempval[threadIdx.x]; // (T0/lattice_Size) ? tempval[threadIdx.x] : cuConj(tempval[threadIdx.x]);
            H.cols[ ii*stride + 4*site + 2 + (T0/lattice_Size) + start ] = (T0/lattice_Size) ? temppos[threadIdx.x] : ii;
            H.rows[ ii*stride + 4*site + 2 + (T0/lattice_Size) + start ] = rowtemp;

            H.set[ ii*stride + 4*site + 2 + (T0/lattice_Size) + start ] = (int)compare;
        }
    }//end of ii
}//end of FillSparse
