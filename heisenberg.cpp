#include"hamiltonian.h"
#include"lattice.h"
#include<cstdlib>
#include"cuda.h"
#include<iostream>

int main()
{
    for(int i = 0; i < 1; i++){
    int** Bond;
    //cout<<i<<" "<<endl;
    int how_many = 25;
    Bond = (int**)malloc(how_many*sizeof(int*));
    d_hamiltonian* hamil_lancz = (d_hamiltonian*)malloc(how_many*sizeof(d_hamiltonian));
    int* nsite = (int*)malloc(how_many*sizeof(int));
    int* Sz = (int*)malloc(how_many*sizeof(int));
    float* JJ = (float*)malloc(how_many*sizeof(float));
    int* model_type = (int*)malloc(how_many*sizeof(int));
    int* num_Elem = (int*)malloc(how_many*sizeof(int));

    int device = i%2;

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

    ConstructSparseMatrix(how_many, model_type, nsite, Bond, hamil_lancz, JJ, Sz, num_Elem, device);

    free(nsite);
    free(Sz);
    free(JJ);
    free(model_type);
    free(Bond);
    free(hamil_lancz);
    free(num_Elem);
    }
    return 0;
}
