//#include"hamiltonian.h"
#include"lattice.h"
#include<cstdlib>
#include"cuda.h"
#include<iostream>
#include"lanczos.h"
//#include"hamiltonian.h"
int main()
{
    for(int i = 0; i < 1; i++){
    int** Bond;
    //cout<<i<<" "<<endl;
    int how_many = 1;
    /*if (i == 1)
    {
        how_many = 5;
    }*/
    Bond = (int**)malloc(how_many*sizeof(int*));
    d_hamiltonian* hamil_lancz = (d_hamiltonian*)malloc(how_many*sizeof(d_hamiltonian));
    int* nsite = (int*)malloc(how_many*sizeof(int));
    int* Sz = (int*)malloc(how_many*sizeof(int));
    float* JJ = (float*)malloc(how_many*sizeof(float));
    int* model_type = (int*)malloc(how_many*sizeof(int));
    int* num_Elem = (int*)malloc(how_many*sizeof(int));

    //cudaSetDevice(1);
    int device = 0; //i%2;

    for(int i = 0; i < how_many; i++)
    {
        
        nsite[i] = 12;
        Bond[i] = (int*)malloc(3*nsite[i]*sizeof(int));
        Fill_Bonds_12A(Bond[i]);
        Sz[i] = 0;
        JJ[i] = 1.f;
        model_type[i] = 0;
    }


    int dim;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time;

    cudaEventRecord(start,0);
    ConstructSparseMatrix(how_many, model_type, nsite, Bond, hamil_lancz, JJ, Sz, num_Elem, device);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout<<"Time to construct Hamiltonians: "<<time<<endl;
    cudaEventRecord(start,0);
    lanczos(how_many, num_Elem, hamil_lancz, 200, 3, 1e-12);
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout<<"Time to perform Lanczos: "<<time<<endl;
    for(int j = 0; j<how_many; j++)
    {
        cudaFree(hamil_lancz[j].rows);
        cudaFree(hamil_lancz[j].cols);
        cudaFree(hamil_lancz[j].vals);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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
