/*!
    \file hamiltonian.h
    \brief Functions for Hamiltonian generation
*/

#include<cmath>
#include <iostream>
#include <cstdlib>
#include "cuda.h"
#include "cuComplex.h"
#include <fstream>
#include "sort/util/cucpp.h" // MGPU utility classes
#include "sort/inc/mgpusort.hpp"
#include "lattice.h"
#include "thrust/device_ptr.h"
#include "thrust/reduce.h"

using namespace std;

/*! 
    \brief A struct to hold input parameters
    
    Holds information on which model we are considering, the lattice size, the S_z sector in question, and coupling parameters
*/
struct parameters
{
    //! The spin model being simulated - three possibilities. All handled in the S_z basis   
    int model_type;
    
    //! Restricts basis to states which have (spins up) - (spins down) = Sz. Onlly useful when Hamiltonian is S_z preserving. 
    int Sz;
    //! The number of lattice sites
    int nsite;
    //! Describes the strength of the first interaction. In the case that only one interaction is present (Heisenberg and XY), this is the only parameter used. 
    float J1;
    //! Describes the strength of the second interaction. Only used in Transverse Field Ising model. In other cases, set to 0.
    float J2;
};

/*!
    \brief A struct to hold the Hamiltonian using doubles
    
    Holds rows, columns, values, and dimensions of the Hilbert space 
*/
struct d_hamiltonian
{

    //! An array which holds the row indices
    int* rows;
    //! An array which holds the column indices
    int* cols;
    //! An array which holds the values as double precision floats
    double* vals;
    //! The full dimension of the Hilbert space
    int fulldim;
    //! The dimension of the sector considered (can equal fulldim)
    int sectordim;
};

/*!
    \brief A struct to hold the Hamiltonian using floats
    
    Holds rows, columns, values, and dimensions of the Hilbert space
*/
struct f_hamiltonian
{
    //! An array which holds the row indices
    int* rows;
    //! An array which holds the column indices
    int* cols;
    //! An array which holds the values as single precision floats
    float* vals;
    //! This array contains either 0 or 1, determining whether the corresponding Hamiltonian element is zero-valued (0) or not (1). This is used to get the total number of nonzero elements. 
    int* set;
    //! The full dimension of the Hilbert space
    int fulldim;
    //! The dimension of the sector considered (can equal fulldim)
    int sectordim;
};

/*!
    \fn __host__ int GetBasis(int dim, int lattice_Size, int Sz, int basis_Position[], int basis[]);
    
    \brief Extracts the basis elements in the sector considered and determines size of this sector
    
    \param dim The dimension of the full Hilbert space
    \param lattice_Size The number of sites in the lattice
    \param Sz The spin sector we are restricting ourselves to
    \param basis_Position Array which stores the position of the in-sector basis states within basis
    \param basis Array which stores only the in-sector basis states (kets)

    \return The size of the in-sector Hilbert space
*/

__host__ int GetBasis(int dim, int lattice_Size, int Sz, int basis_Position[], int basis[]);

/*! 
    \fn __host__ void ConstructSparseMatrix(const int how_many, int** Bond, d_hamiltonian*& hamil_lancz, parameters* data, int*& count_array, int device);
    
    \brief A CPU function which creates basis information, then calls GPU functions to fill arrays describing the Hamiltonian
    
    \param how_many How many Hamiltonians will be constructed at once on the GPU
    \param Bond Array storing information about which lattice sites are bonded to which
    \param hamil_lancz Array of structs which store the final Hamiltonians for diagonalization once they have been generated
    \param data Array of structs which store the input parameters such as model, S_z sector, and coupling constants
    \param count_array Array which holds the number of nonzero elements for each Hamiltonian
    \param device Which GPU the functions will be launched on
*/
__host__ void ConstructSparseMatrix(const int how_many, int** Bond, d_hamiltonian*& hamil_lancz, parameters* data, int*& count_array, int device);

/*! 
    \fn __global__ void FullToCOO(int num_Elem, float* H_vals, double* hamil_Values, int dim);
    
    \brief A GPU function which switches the single precision values to double precision
    
    \param num_Elem The number of non-zero elements in the Hamiltonian
    \param H_vals The array storing the Hamiltonian values in single precision
    \param hamil_Values The array which will store the Hamiltonian values in double precision
    \param dim The dimension of the Hilbert space
*/

__global__ void FullToCOO(int num_Elem, float* H_vals, double* hamil_Values, int dim);

//--------Declarations of Hamiltonian functions for Heisenberg Model--------------

/*!
    \fn __device__ float HOffBondXHeisenberg(const int si, const int bra, const float JJ);
    \brief A GPU function which finds the value of the spin operator on the bond in the x direction
    
    \param si The site whose bond we are looking at
    \param bra The state resulting from applying the spin operator
    \param JJ The coupling of the spin operator on the bond 
    
    \result The value of <a|H|b> for the x bond
*/

__device__ float HOffBondXHeisenberg(const int si, const int bra, const float JJ);

/*!
    \fn __device__ float HOffBondYHeisenberg(const int si, const int bra, const float JJ);
    \brief A GPU function which finds the value of the spin operator on the bond in the y direction
    
    \param si The site whose bond we are looking at
    \param bra The state resulting from applying the spin operator
    \param JJ The coupling of the spin operator on the bond 

    \result The value of <a|H|b> for the y bond
*/
__device__ float HOffBondYHeisenberg(const int si, const int bra, const float JJ);

/*!
    \fn __device__ float HDiagPartHeisenberg(const int bra, int lattice_Size, int3* d_Bond, const float JJ);
    \brief A GPU function which finds the value of the spin operator on the bond on the diagonal
    
    \param bra The state resulting from applying the spin operator
    \param lattice_Size The number of sites in the lattice
    \param d_Bond Array which stores information about which sites are bonded to which
    \param JJ The coupling of the spin operator on the bond
    
    \result The value of <a|H|b> on the diagonal 
*/
__device__ float HDiagPartHeisenberg(const int bra, int lattice_Size, int3* d_Bond, const float JJ);

/*!
    \brief A GPU function which finds and inserts the values, row indices, and column indices of diagonal elements into Hamiltonian storage arrays
    
    \param d_basis An array storing the kets which are in-sector
    \param H A struct which contains arrays to be filled with information about the Hamiltonian's diagonal elements
    \param d_Bond An array which contains information about which sites are bonded to which
    \param data A struct containing information about the simulation parameters, such as number of lattice sites, coupling constants, and S_z sector
*/

__global__ void FillDiagonalsHeisenberg(int* d_basis, f_hamiltonian H, int* d_Bond, parameters data);

/*!
    \brief A GPU function which finds and inserts the values, row indices, and column indices of off-diagonal elements into Hamiltonian storage arrays 
    
    \param d_basis_Position Array which stores the location of in-sector kets in d_basis
    \param d_basis Array which stores the in-sector kets
    \param H Struct which contains arrays to be filled with information about the Hamiltonian's off-diagonal elements
    \param d_Bond An array which contains information about which sites are bonded to which
    \param data A struct containing information about the simulation parameters, such as number of lattice sites, coupling constants, and S_z sector
    \param offset How many blocks of off-diagonal elements have already been processed by other runs of the kernel
*/

__global__ void FillSparseHeisenberg(int* d_basis_Position, int* d_basis, f_hamiltonian H, int* d_Bond, parameters data, int offset);

//------Declarations of Hamiltonian functions for XY Model -------------------

/*!
    \brief A GPU function which finds the value of the spin operator on the bond in the x direction
    
    \param si The site whose bond we are looking at
    \param bra The state resulting from applying the spin operator
    \param JJ The coupling of the spin operator on the bond 
    
    \result The value of <a|H|b> on the x-bond 
*/
__device__ float HOffBondXXY(const int si, const int bra, const float JJ);

/*!
    \brief A GPU function which finds the value of the spin operator on the bond in the y direction
    
    \param si The site whose bond we are looking at
    \param bra The state resulting from applying the spin operator
    \param JJ The coupling of the spin operator on the bond 
    
    \result The value of <a|H|b> on the y-bond 
*/
__device__ float HOffBondYXY(const int si, const int bra, const float JJ);

/*!
    \brief A GPU function which finds the value of the spin operator on the bond on the diagonal
    
    \param bra The state resulting from applying the spin operator
    \param lattice_Size The number of sites in the lattice
    \param d_Bond Array which stores information about which sites are bonded to which
    \param JJ The coupling of the spin operator on the bond 
*/
__device__ float HDiagPartXY(const int bra, int lattice_Size, int3* d_Bond, const float JJ);

/*!
    \brief A GPU function which finds and inserts the values, row indices, and column indices of diagonal elements into Hamiltonian storage arrays
    
    \param d_basis An array storing the kets which are in-sector
    \param H A struct which contains arrays to be filled with information about the Hamiltonian's diagonal elements
    \param d_Bond An array which contains information about which sites are bonded to which
    \param data A struct containing information about the simulation parameters, such as number of lattice sites, coupling constants, and S_z sector
*/
__global__ void FillDiagonalsXY(int* d_basis, f_hamiltonian H, int* d_Bond, parameters data);

/*!
    \brief A GPU function which finds and inserts the values, row indices, and column indices of off-diagonal elements into Hamiltonian storage arrays 
    
    \param d_basis_Position Array which stores the location of in-sector kets in d_basis
    \param d_basis Array which stores the in-sector kets
    \param H Struct which contains arrays to be filled with information about the Hamiltonian's off-diagonal elements
    \param d_Bond An array which contains information about which sites are bonded to which
    \param data A struct containing information about the simulation parameters, such as number of lattice sites, coupling constants, and S_z sector
    \param offset How many blocks of off-diagonal elements have already been processed by other runs of the kernel
*/
__global__ void FillSparseXY(int* d_basis_Position, int* d_basis, f_hamiltonian H, int* d_Bond, parameters data, int offset);

//--------Declarations of Hamiltonian functions for transverse field Ising Model-------

/*!
    \brief A GPU function which finds the value of the spin operator on the bond in the x direction
    
    \param si The site whose bond we are looking at
    \param bra The state resulting from applying the spin operator
    \param JJ The coupling of the spin operator on the bond 
    
    \result The value of <a|H|b> on the x-bond 
*/
__device__ float HOffBondXTFI(const int si, const int bra, const float JJ);

/*!
    \brief A GPU function which finds the value of the spin operator on the bond in the x direction
    
    \param si The site whose bond we are looking at
    \param bra The state resulting from applying the spin operator
    \param JJ The coupling of the spin operator on the bond 
    
    \result The value of <a|H|b> on the y-bond 
*/
__device__ float HOffBondYTFI(const int si, const int bra, const float JJ);

/*!
    \brief A GPU function which finds the value of the spin operator on the bond on the diagonal
    
    \param bra The state resulting from applying the spin operator
    \param lattice_Size The number of sites in the lattice
    \param d_Bond Array which stores information about which sites are bonded to which
    \param JJ The coupling of the spin operator on the bond 
*/
__device__ float HDiagPartTFI(const int bra, int lattice_Size, int2* d_Bond, const float JJ);

/*!
    \brief A GPU function which finds and inserts the values, row indices, and column indices of diagonal elements into Hamiltonian storage arrays
    
    \param d_basis An array storing the kets which are in-sector
    \param H A struct which contains arrays to be filled with information about the Hamiltonian's diagonal elements
    \param d_Bond An array which contains information about which sites are bonded to which
    \param data A struct containing information about the simulation parameters, such as number of lattice sites, coupling constants, and S_z sector
*/
__global__ void FillDiagonalsTFI(int* d_basis, f_hamiltonian H, int* d_Bond, parameters data);

/*!
    \brief A GPU function which finds and inserts the values, row indices, and column indices of off-diagonal elements into Hamiltonian storage arrays 
    
    \param d_basis_Position Array which stores the location of in-sector kets in d_basis
    \param d_basis Array which stores the in-sector kets
    \param H Struct which contains arrays to be filled with information about the Hamiltonian's off-diagonal elements
    \param d_Bond An array which contains information about which sites are bonded to which
    \param data A struct containing information about the simulation parameters, such as number of lattice sites, coupling constants, and S_z sector
    \param offset How many blocks of off-diagonal elements have already been processed by other runs of the kernel
*/
__global__ void FillSparseTFI(int* d_basis_Position, int* d_basis, f_hamiltonian H, int* d_Bond, parameters data, int offset);

