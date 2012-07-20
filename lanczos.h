/*!
    \file lanczos.h

    \brief Functions to perform Lanczos diagonalizations on Hamiltonians
*/

#include<iomanip>
#include"hamiltonian.h"
#include"cuda_runtime.h"
#include"cublas_v2.h"
#include"cusparse_v2.h"

#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

using namespace std;

/*!
    \brief A CPU function which interfaces with CUSPARSE and CUBLAS to perform Lanczos diagonalization on many Hamiltonians simultaneously, with reorthogonalization

    \param how_many How many Hamiltonians will be diagonalized this run
    \param num_Elem An array containing the number of non-zero elements per Hamiltonian
    \param Hamiltonian An array of structs containing the Hamiltonian information
    \param groundstates An array which will be filled with the groundstate vector after diagonalization
    \param eigenvalues An array which will be filled with the eigenenergies
    \param max_Iter The starting maximum number of iterations 
    \param num_Eig The number of eigenstates to extract
    \param conv_req The convergence requirement for the eigenvalues
*/

__host__ void lanczos(const int how_many, const int* num_Elem, d_hamiltonian*& Hamiltonian, double**& groundstates, double**& eigenvalues, int max_Iter, const int num_Eig, const double conv_req);

    /*! 
    \brief A CPU function which diagonalizes a symmetric tridiagonal matrix using the QL scheme

    \param d An array containing the diagonal elements of the matrix
    \param e An array containing the off-diagonal elements of the matrix
    \param n The dimension of the matrix
    \param max_Iter The maximum dimension of the eigenvector storage matrix
    \param z An array which stores the eigenvectors of the tridiagonal matrix 

    \return A flag determining whether the function completed successfully or not
*/

__host__ int tqli(double* d, double* e, int n, int max_Iter, double* z);

/*!
    \brief A CPU function which computes c = sqrt(a^2 + b^2)

    \param a The length of the first "side"
    \param b The length of the second "side"
    
    \result The length of the "hypotenuse"
*/
__host__ double pythag(double a, double b);

/*!
    \brief A GPU function which rotates the Lanczos basis to get the groundstate in the S_z basis

    \param groundstates An array which will be filled with the groundstate information
    \param lanczos_store An array which stores all the Lanczos vectors generated during diagonalization
    \param H_eigen An array which stores all the eigenvectors generated from diagonalizing the tridiagonal matrix
    \param mat_dim The dimension of H_eigen (equivalent to the total number of iterations)
    \param vec_dim The dimension of the Hilbert space
*/
__global__ void GetGroundstate(double* groundstates, double** lanczos_store, double* H_eigen, int mat_dim, int vec_dim);
