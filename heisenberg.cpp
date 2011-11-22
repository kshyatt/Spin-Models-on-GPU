#include<iostream>
#include"lanczos.h"
#include"cuda.h"
#include"cuComplex.h"
#include"lattice.h"

using namespace std;

int main(){

	int** Bond;

	const int how_many = 1;
  	Bond = (int**)malloc(how_many*sizeof(int*));
	d_hamiltonian* hamil_lancz = (d_hamiltonian*)malloc(how_many*sizeof(d_hamiltonian));
	int* nsite = (int*)malloc(how_many*sizeof(int));
	int* Sz = (int*)malloc(how_many*sizeof(int));
	float* JJ = (float*)malloc(how_many*sizeof(float));	
	int* model_type = (int*)malloc(how_many*sizeof(int));

	for(int i = 0; i < how_many; i++){
		
		nsite[i] = 16;
		Bond[i] = (int*)malloc(3*nsite[i]*sizeof(int));
		Fill_Bonds_16B(Bond[i]);

		
		Sz[i] = 0;
		JJ[i] = 1.f;
		model_type[i] = 0;
	}
	

  	int dim;

	int* num_Elem = ConstructSparseMatrix(how_many, model_type, nsite, Bond, hamil_lancz, JJ, Sz );

	free(Bond);


	lanczos(how_many, num_Elem, hamil_lancz, 200, 3, 1e-3);

	cudaFree(hamil_lancz);
	return 0;

}


