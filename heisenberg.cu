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

__device__ float HDiagPartHeisenberg1D(const int bra, int latticeSize, int3* d_Bond, const float JJ)
{

    int S0b,S1b ; //spins (bra
    int T0,T1; //site
    //int P0, P1, P2, P3; //sites for plaquette (Q)
    //int s0p, s1p, s2p, s3p;
    float valH = 0.f;

    for (int Ti=0; Ti<latticeSize; Ti++)
    {
        //***HEISENBERG PART

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


__device__ float HDiagPartHeisenberg2D(const int bra, int latticeSize, int3* d_Bond, const float JJ)
{

    int S0b,S1b ; //spins (bra
    int T0,T1; //site
    //int P0, P1, P2, P3; //sites for plaquette (Q)
    //int s0p, s1p, s2p, s3p;
    float valH = 0.f;

    for (int Ti=0; Ti<latticeSize; Ti++)
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
    int latticeSize = data.nsite;
    int site = threadIdx.x%(latticeSize);
    int dim = H.sectorDim;

    unsigned int tempi;

    __shared__ int3 tempBond[32];

    if (row < dim)
    {
        tempi = d_basis[row];
        (tempBond[site]).x = d_Bond[site];
        (tempBond[site]).y = d_Bond[latticeSize + site];

        switch( data.dimension )
        {
            case 1 :
                H.vals[row] = HDiagPartHeisenberg1D(tempi, latticeSize, tempBond, data.J1);
                break;
            case 2 :
                (tempBond[site]).z = d_Bond[2*latticeSize + site];
                H.vals[row] = HDiagPartHeisenberg2D(tempi, latticeSize, tempBond, data.J1);
                break;
        }

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

__global__ void FillSparseHeisenberg(int* d_basisPosition, int* d_basis, f_hamiltonian H, int* d_Bond, parameters data, int offset)
{

    int latticeSize = data.nsite;
    int dim = H.sectorDim;
    int ii = (blockDim.x/(2*latticeSize))*blockIdx.x + threadIdx.x/(2*latticeSize) + offset*(blockDim.x/(2*latticeSize));
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
    __shared__ uint tempod[arraySize];

    int stride = 2*data.dimension*latticeSize;
    //int tempcount;
    int site = T0%(latticeSize);
    count = 0;
    int rowTemp;

    int start = (bool)(dim%arraySize) ? (dim/arraySize + 1)*arraySize : dim/arraySize;

    int s;
    int braSector;
    //int si, sj;//sk,sl; //spin operators
    //unsigned int tempi;// tempod; //tempj;
    //cuDoubleComplex tempD;

    bool compare;

    if( ii < dim )
    {
        tempi = d_basis[ii];
        if (T0 < 2*latticeSize)
        {
            //Putting bond info in shared memory
            (tempBond[site]).x = d_Bond[site];
            (tempBond[site]).y = d_Bond[latticeSize + site];

            __syncthreads();
            
            //Horizontal bond ---------------
            s = (tempBond[site]).x;
            tempod[threadIdx.x] = tempi;
            braSector = (tempi & (1 << s)) >> s;
            tempod[threadIdx.x] ^= (1<<s);
            s = (tempBond[site]).y;
            braSector ^= (tempi & (1 << s)) >> s;
            tempod[threadIdx.x] ^= (1<<s);

            //tempod[threadIdx.x] ^= (1<<si); //toggle bit
            //tempod[threadIdx.x] ^= (1<<sj); //toggle bit

            compare = (d_basisPosition[tempod[threadIdx.x]] != -1) && braSector;
            compare &= (d_basisPosition[tempod[threadIdx.x]] > ii);
            tempPos[threadIdx.x] = (compare) ? d_basisPosition[tempod[threadIdx.x]] : dim;
            tempVal[threadIdx.x] = HOffBondXHeisenberg(site, tempi, data.J1);

            count += (int)compare;
            //tempcount = (T0/latticeSize);
            rowTemp = (T0/latticeSize) ? ii : tempPos[threadIdx.x];
            rowTemp = (compare) ? rowTemp : 2*dim;

            H.vals[ ii*stride + 4*site + (T0/latticeSize) + start ] = tempVal[threadIdx.x]; // (T0/latticeSize) ? tempVal[threadIdx.x] : cuConj(tempVal[threadIdx.x]);
            H.cols[ ii*stride + 4*site + (T0/latticeSize) + start ] = (T0/latticeSize) ? tempPos[threadIdx.x] : ii;
            H.rows[ ii*stride + 4*site + (T0/latticeSize) + start ] = rowTemp;

            H.set[ ii*stride + 4*site + (T0/latticeSize) + start ] = (int)compare;

            if (data.dimension == 2)
            {
                (tempBond[site]).z = d_Bond[2*latticeSize + site];

                //Vertical bond -----------------
                s = (tempBond[site]).x;
                tempod[threadIdx.x] = tempi;
                braSector = (tempi & (1 << s)) >> s;
                tempod[threadIdx.x] ^= (1<<s);
                s = (tempBond[site]).z;
                braSector ^= (tempi & (1 << s)) >> s;
                tempod[threadIdx.x] ^= (1<<s);

                //tempod[threadIdx.x] ^= (1<<si); //toggle bit
                //tempod[threadIdx.x] ^= (1<<sj); //toggle bit

                compare = (d_basisPosition[tempod[threadIdx.x]] != -1) && braSector;
                compare &= (d_basisPosition[tempod[threadIdx.x]] > ii);
                tempPos[threadIdx.x] = (compare) ? d_basisPosition[tempod[threadIdx.x]] : dim;
                tempVal[threadIdx.x] = HOffBondYHeisenberg(site,tempi, data.J1);

                count += (int)compare;
                //tempcount = (T0/latticeSize);
                rowTemp = (T0/latticeSize) ? ii : tempPos[threadIdx.x];
                rowTemp = (compare) ? rowTemp : 2*dim;

                H.vals[ ii*stride + 4*site + 2 + (T0/latticeSize) + start ] = tempVal[threadIdx.x]; // (T0/latticeSize) ? tempVal[threadIdx.x] : cuConj(tempVal[threadIdx.x]);
                H.cols[ ii*stride + 4*site + 2 + (T0/latticeSize) + start ] = (T0/latticeSize) ? tempPos[threadIdx.x] : ii;
                H.rows[ ii*stride + 4*site + 2 + (T0/latticeSize) + start ] = rowTemp;

                H.set[ ii*stride + 4*site + 2 + (T0/latticeSize) + start ] = (int)compare;
            }
        }
    }//end of ii
}//end of FillSparse
