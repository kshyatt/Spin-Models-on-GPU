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
    float* J1 = (float*)malloc(how_many*sizeof(float));
    float* J2 = (float*)malloc(how_many*sizeof(float));
    int* model_type = (int*)malloc(how_many*sizeof(int));
    int* num_Elem = (int*)malloc(how_many*sizeof(int));

    //cudaSetDevice(1);
    int device = 0; //i%2;

    for(int i = 0; i < how_many; i++)
    {
        
        nsite[i] = 16;
        Bond[i] = (int*)malloc(3*nsite[i]*sizeof(int));
        Fill_Bonds_16B(Bond[i]);
        Sz[i] = 0;
        J1[i] = 1.f;
        J2[i] = 0.f;
        model_type[i] = 0;
    }


    int dim;
    /*cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);*/
    float time;

    //cudaEventRecord(start,0);
    ConstructSparseMatrix(how_many, model_type, nsite, Bond, hamil_lancz, J1, J2, Sz, num_Elem, device);
    /*cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout<<"Time to construct Hamiltonians: "<<time<<endl;
    cout<<num_Elem[0]<<endl;
    cudaEventRecord(start,0);
    */
    lanczos(how_many, num_Elem, hamil_lancz, 200, 3, 1e-12);
    /*
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout<<"Time to perform Lanczos: "<<time<<endl;*/
    for(int j = 0; j<how_many; j++)
    {
        cudaFree(hamil_lancz[j].rows);
        cudaFree(hamil_lancz[j].cols);
        cudaFree(hamil_lancz[j].vals);
    }
    //cudaEventDestroy(start);
    //cudaEventDestroy(stop);
    free(nsite);
    free(Sz);
    free(J1);
    free(J2);
    free(model_type);
    free(Bond);
    free(hamil_lancz);
    free(num_Elem);
    }
    return 0;
}
