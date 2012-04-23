#include"hamiltonian.h"
//#include"lattice.h"
#include<cstdlib>
#include"cuda.h"
#include<iostream>

int main()
{
    for(int i = 0; i < 1; i++){
    int** Bond;
    //cout<<i<<" "<<endl;
    int how_many = 1;
    /*if (i == 1)
    {
        how_many = 30;
    }*/
    Bond = (int**)malloc(how_many*sizeof(int*));
    if (Bond == NULL)
    {
        cout<<"Error: Bond array is NULL"<<endl;
        return 1;
    }
    d_hamiltonian* hamil_lancz = (d_hamiltonian*)malloc(how_many*sizeof(d_hamiltonian));
    if (hamil_lancz == NULL)
    {
        cout<<"Error: hamil_lancz array is NULL"<<endl;
        return 1;
    }
    int* nsite = (int*)malloc(how_many*sizeof(int));
    if (nsite == NULL)
    {
        cout<<"Error: nsite array is NULL"<<endl;
        return 1;
    }
    int* Sz = (int*)malloc(how_many*sizeof(int));
    if (Sz == NULL)
    {
        cout<<"Error: Sz array is NULL"<<endl;
        return 1;
    }
    float* JJ = (float*)malloc(how_many*sizeof(float));
    if (JJ == NULL)
    {
        cout<<"Error: JJ array is NULL"<<endl;
        return 1;
    }
    
    float* h = (float*)malloc(how_many*sizeof(float));
    if (h == NULL)
    {
        cout<<"Error: JJ array is NULL"<<endl;
        return 1;
    }
    
    int* model_type = (int*)malloc(how_many*sizeof(int));
    if (model_type == NULL)
    {
        cout<<"Error: model_type array is NULL"<<endl;
        return 1;
    }
    int* num_Elem = (int*)malloc(how_many*sizeof(int));
    if (num_Elem == NULL)
    {
        cout<<"Error: num_Elem array is NULL"<<endl;
        return 1;
    }

    int device = 0; //i%2;

    for(int i = 0; i < how_many; i++)
    {
        
        nsite[i] = 20;
        Bond[i] = (int*)malloc(2*nsite[i]*sizeof(int));
        for(int k = 0; k < nsite[i]; k++){
          Bond[i][k] = k;
          Bond[i][k + nsite[i]] = k + 1;
        }
        
        Sz[i] = 0;
        JJ[i] = 1.f;
        h[i] = 3.f;
        model_type[i] = 0;
    }


    int dim;

    ConstructSparseMatrix(how_many, model_type, nsite, Bond, hamil_lancz, JJ, h, Sz, num_Elem, device);
    //lanczos(how_many, num_Elem, hamil_lancz, 200, 3, 1e-12);
    for(int j = 0; j<how_many; j++)
    {
        cudaFree(hamil_lancz[j].rows);
        cudaFree(hamil_lancz[j].cols);
        cudaFree(hamil_lancz[j].vals);
    }
    free(nsite);
    free(Sz);
    free(JJ);
    free(h);
    free(model_type);
    free(Bond);
    free(hamil_lancz);
    free(num_Elem);
    }
    return 0;
}
