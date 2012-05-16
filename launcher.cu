/*!
    \file launcher.cu
    \brief Controller code that launches Hamiltonian generation and Lanczos diagonalization
*/

#include "lattice.h"
#include <cstdlib>
#include "cuda.h"
#include <fstream>
#include <iostream>
#include "lanczos.h"
//#include"hamiltonian.h"
int main()
{
    for(int i = 0; i < 1; i++)
    {
        int** Bond;
        //cout<<i<<" "<<endl;
        int how_many;
        /*if (i == 1)
        {
            how_many = 5;
        }*/

        ifstream fin;
        fin.open("data.dat");
        fin >> how_many ; 

        Bond = (int**)malloc(how_many*sizeof(int*));

        d_hamiltonian* hamil_lancz = (d_hamiltonian*)malloc(how_many*sizeof(d_hamiltonian));

        parameters* data = (parameters*)malloc(how_many*sizeof(parameters));

        double** groundstates = (double**)malloc(how_many*sizeof(double*));

        if (data == NULL)
        {
            cerr<<"Malloc of parameter container failed!"<<endl;
            return 1;
        }

        int* num_Elem = (int*)malloc(how_many*sizeof(int));

        //cudaSetDevice(1);
        int device = 1; //i%2;

        for(int i = 0; i < how_many; i++)
        {

            fin >> data[i].nsite >> data[i].Sz >> data[i].J1 >> data[i].J2 >> data[i].model_type >> data[i].dimension;
            switch (data[i].dimension)
            {
                case 1:
                    Bond[i] = (int*)malloc(2*data[i].nsite*sizeof(int));
                    for( int j = 0; j < data[i].nsite; j++ ){
                        Bond[i][j] = j;
                        Bond[i][j+ data[i].nsite] = (j+1)%data[i].nsite;
                    }
                    break;
                case 2:
                    Bond[i] = (int*)malloc(3*data[i].nsite*sizeof(int));
                    Fill_Bonds_16B(Bond[i]);
                    break;
            }
        }


        /*cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);*/
        //float time;

        //cudaEventRecord(start,0);
        ConstructSparseMatrix(how_many, Bond, hamil_lancz, data, num_Elem, device);
        /*cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        cout<<"Time to construct Hamiltonians: "<<time<<endl;
        cudaEventRecord(start,0);
        */
        lanczos(how_many, num_Elem, hamil_lancz, groundstates, 200, 3, 1e-12);
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
            cudaFree(groundstates[j]);
        }
        //cudaEventDestroy(start);
        //cudaEventDestroy(stop);
        free(data);
        free(Bond);
        free(hamil_lancz);
        free(num_Elem);
        free(groundstates);
    }
    return 0;
}
