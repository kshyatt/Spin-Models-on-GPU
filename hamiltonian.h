/*!
    \file hamiltonian.h
    \brief Functions for Hamiltonian generation

    Contains declarations for functions which generate diagonal and off-diagonal elements of the Hamiltonian on the GPU. Also declares handler functions which go through all steps of matrix creation. Defines structs which store relevant information about the Hamiltonian. 

*/

/*!
    \defgroup Heisenberg
    \brief Functions used to generate Hamiltonians for the Heisenberg model
*/

/*!
    \defgroup XY
    \brief Functions used to generate Hamiltonians for the XY model
*/

/*!
    \defgroup TFIM
    \brief Functions used to generate Hamiltonians for the Transverse Field Ising model
*/
#include <cmath>
#include <iostream>
#include <cstdlib>
#include "cuda.h"
#include "cuComplex.h"
#include <fstream>
#include "sort/util/cucpp.h" // MGPU utility classes
#include "sort/inc/mgpusort.hpp"
//#include "lattice.h"
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
    int modelType;
    
    //! Restricts basis to states which have (spins up) - (spins down) = Sz. Onlly useful when Hamiltonian is S_z preserving. 
    int Sz;
    //! The number of lattice sites
    int nsite;
    //! Describes the strength of the first interaction. In the case that only one interaction is present (Heisenberg and XY), this is the only parameter used. 
    float J1;
    //! Describes the strength of the second interaction. Only used in Transverse Field Ising model. In other cases, set to 0.
    float J2;
    //! How many dimensions the system in question has - options are 1, 2, or 3
    int dimension;
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
    int fullDim;
    //! The dimension of the sector considered (can equal fulldim)
    int sectorDim;
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
    int fullDim;
    //! The dimension of the sector considered (can equal fulldim)
    int sectorDim;
};

/*!
    \fn __host__ int GetBasis(int dim, int lattice_Size, int Sz, int basis_Position[], int basis[]);
    
    \brief Extracts the basis elements in the sector considered and determines size of this sector
    
    Sz eigenstates are stored using a base 2 representation: 1 is an up spin, 0 is a down spin. For example:
    - 1111 -> all spins are up
    - 0000 -> all spins are down
    - 1001 -> end spins are up, middle spins are down

    This function looks at the total S_z value of the ket in question and determines from this whether the ket is in the S_z sector specified when the program is launched. If all S_z sectors are allowed (for instance, if the Hamiltonian is not S_z preserving, like the Transverse Field Ising Model) then all kets are allowed. The in-sector kets are stored in basis, and their position in that array is stored in basis_Position. For instance, if we specify that we only want to consider the S_z = 0 sector:

<table>
<tr>
    <th> Ket </td> <th> S_z sector </td> <th> In basis? </td> <th> basis_Position entry </th>
</tr>
<tr>
    <td> 0000 </td> <td> -4 </td> <td> not placed in basis </td> <td> basis_Position[0] = -1 </td>
</tr>
<tr>
    <td> 0001 </td> <td> -2 </td> <td> not placed in basis </td> <td> basis_Position[1] = -1 </td>
</tr>
<tr>
    <td> 0010 </td> <td> -2 </td> <td> not placed in basis </td> <td> basis_Position[2] = -1 </td>
</tr>
<tr>
    <td> 0011 </td> <td> 0 </td> <td> placed in basis[0] </td> <td> basis_Position[3] = 0 </td>
</tr>
<tr> 
    <td> 0100 </td> <td> -2 </td> <td> not placed in basis </td> <td> basis_Position[4] = -1 </td>
</tr>
<tr>
    <td> 0101 </td> <td> 0 </td> <td> placed in basis[1] </td> <td> basis_Position[5] = 1 </td>
</tr>
<tr>
    <td> 0110 </td> <td> 0 </td> <td> placed in basis[2] </td> <td> basis_Position[6] = 2 </td>
</tr>
<tr>
    <td> 0111 </td> <td> 2 </td> <td> not placed in basis </td> <td> basis_Position[7] = -1 </td>
</tr>
<tr>
    <td> 1000 </td> <td> -2 </td> <td> not placed in basis </td> <td> basis_Position[8] = -1 </td>
</tr>
<tr>
    <td> 1001 </td> <td> 0 </td> <td> placed in basis[3] </td> <td> basis_Position[9] = 3 </td>
</tr>
<tr>
    <td> 1010  </td> <td> 0 </td> <td> placed in basis[4] </td> <td> basis_Position[10] = 4 </td>
</tr>
<tr>
    <td> 1011  </td> <td> 2 </td> <td> not placed in basis </td> <td> basis_Position[11] = -1 </td>
</tr>
<tr>
    <td> 1100  </td> <td> 0 </td> <td> placed in basis[5] </td> <td> basis_Position[12] = 5  </td>
</tr>
<tr>
    <td> 1101  </td> <td> 2 </td> <td> not placed in basis </td> <td> basis_Position[13] = -1 </td>
</tr>
<tr>
    <td> 1110  </td> <td> 2 </td> <td> not placed in basis </td> <td> basis_Position[14] = -1 </td>
</tr>
<tr>
    <td> 1111  </td> <td> 4 </td> <td> not placed in basis </td> <td> basis_Position[15] = -1 </td>
</tr>
</table>
Each time an element is placed in basis, a counter is incremented. The final value of this counter gives the dimension of the sector of interest. In the case that no sector is specified, the counter is just the dimension of the full Hilbert space.

    \param dim The dimension of the full Hilbert space
    \param lattice_Size The number of sites in the lattice
    \param Sz The spin sector we are restricting ourselves to
    \param basis_Position Array which stores the position of the in-sector basis states within basis
    \param basis Array which stores only the in-sector basis states (kets)

    \return The size of the in-sector Hilbert space
*/

__host__ int GetBasis(int dim, parameters data, int basis_Position[], int basis[]);

/*! 
    \fn __host__ void ConstructSparseMatrix(const int how_many, int** Bond, d_hamiltonian*& hamil_lancz, parameters* data, int*& count_array, int device);
    
    \brief A CPU function which creates basis information, then calls GPU functions to fill arrays describing the Hamiltonian

    This function launches one or many cudaStreams, depending on the value of `how_many`. Each stream is responsible for generating one Hamiltonian. First, ConstructSparseMatrix determines the structure of the basis by calling GetBasis. Then it allocates storage arrays on the GPU for the Hamiltonian, using f_hamiltonian. Since single precision operations are much faster on GPU at this time, this allows element generation to take much less time than it otherwise might. The function launches an instance of `FillDiagonals()` and `FillSparse()` in each stream, and these functions fill up the previously allocated storage arrays. A parallel reduction is performed on the array containing zero-valued/non-zero-valued flags to determine how many nonzero elements there are in the Hamiltonian. This is used instead of atomic operations because in this case reduction, a parallel summing algorithm, performs much better. At this point the modern GPU sorting library is loaded. This library is used because it allows multiple value arrays to be sorted by one key array, which thrust does not without defining another custom `struct`. The mgpu library sorts the Hamiltonian arrays and they are copied into the container that will be used for diagonalization later, with an intermediate conversion of the values from single to double precision. The Hamiltonian is now ready to be diagonalized or dumped to disk for storage/correctness checks.
    
    \param howMany How many Hamiltonians will be constructed at once on the GPU
    \param Bond Array storing information about which lattice sites are bonded to which
    \param hamilLancz Array of structs which store the final Hamiltonians for diagonalization once they have been generated
    \param data Array of structs which store the input parameters such as model, S_z sector, and coupling constants
    \param countArray Array which holds the number of nonzero elements for each Hamiltonian
    \param device Which GPU the functions will be launched on
*/
__host__ void ConstructSparseMatrix(const int howMany, int** Bond, d_hamiltonian*& hamilLancz, parameters* data, int*& countArray, int device);

/*! 
    \fn __global__ void FullToCOO(int num_Elem, float* H_vals, double* hamil_Values, int dim);
    
    \brief A GPU function which switches the single precision values to double precision
    
    \param numElem The number of non-zero elements in the Hamiltonian
    \param H_vals The array storing the Hamiltonian values in single precision
    \param hamilValues The array which will store the Hamiltonian values in double precision
    \param dim The dimension of the Hilbert space
*/

__global__ void FullToCOO(int numElem, float* H_vals, double* hamilValues, int dim);

//--------Declarations of Hamiltonian functions for Heisenberg Model--------------

/*!
    \fn __device__ float HOffBondXHeisenberg(const int si, const int bra, const float JJ);
    \brief A GPU function which finds the value of the spin operator on the bond in the x direction
    
    \ingroup Heisenberg

    \param si The site whose bond we are looking at
    \param bra The state resulting from applying the spin operator
    \param JJ The coupling of the spin operator on the bond 
    
    \result The value of <a|H|b> for the x bond
*/

__device__ float HOffBondXHeisenberg(const int si, const int bra, const float JJ);

/*!
    \fn __device__ float HOffBondYHeisenberg(const int si, const int bra, const float JJ);
    \brief A GPU function which finds the value of the spin operator on the bond in the y direction
    
    \ingroup Heisenberg

    \param si The site whose bond we are looking at
    \param bra The state resulting from applying the spin operator
    \param JJ The coupling of the spin operator on the bond 

    \result The value of <a|H|b> for the y bond
*/
__device__ float HOffBondYHeisenberg(const int si, const int bra, const float JJ);

/*!
    \fn __device__ float HDiagPartHeisenberg1D(const int bra, int lattice_Size, int3* d_Bond, const float JJ);
    \brief A GPU function which finds the value of the spin operator on the bond on the diagonal in one dimension
   
    This function computes the value of <i>S<sub>z</sub><sup>i</sup>S<sub>z</sub><sup>j</sup></i> over all bonded pairs <i> (i,j)</i>. If <i>i</i> and <i>j</i> have ferromagnetic ordering (they are aligned) then <i>S<sub>z</sub><sup>i</sup>S<sub>z</sub><sup>j</sup></i> = 1/4. If they have antiferrmagnetic ordering (they are anti-aligned) <i>S<sub>z</sub><sup>i</sup>S<sub>z</sub><sup>j</sup></i> = 0. The final value of the diagonal element is the coupling constant multiplied by the sum of all the bond <i>S<sub>z</sub></i> values.
    
    \ingroup Heisenberg
    
    \param bra The state resulting from applying the spin operator
    \param lattice_Size The number of sites in the lattice
    \param d_Bond Array which stores information about which sites are bonded to which
    \param JJ The coupling of the spin operator on the bond
    
    \result The value of <a|H|b> on the diagonal 
*/
__device__ float HDiagPartHeisenberg1D(const int bra, int lattice_Size, int3* d_Bond, const float JJ);

/*!
    \fn __device__ float HDiagPartHeisenberg2D(const int bra, int lattice_Size, int3* d_Bond, const float JJ);
    \brief A GPU function which finds the value of the spin operator on the bond on the diagonal in two dimensions
   
    This function computes the value of <i>S<sub>z</sub><sup>i</sup>S<sub>z</sub><sup>j</sup></i> over all bonded pairs <i> (i,j)</i>. If <i>i</i> and <i>j</i> have ferromagnetic ordering (they are aligned) then <i>S<sub>z</sub><sup>i</sup>S<sub>z</sub><sup>j</sup></i> = 1/4. If they have antiferrmagnetic ordering (they are anti-aligned) <i>S<sub>z</sub><sup>i</sup>S<sub>z</sub><sup>j</sup></i> = 0. The final value of the diagonal element is the coupling constant multiplied by the sum of all the bond <i>S<sub>z</sub></i> values.
    
    \ingroup Heisenberg
    
    \param bra The state resulting from applying the spin operator
    \param lattice_Size The number of sites in the lattice
    \param d_Bond Array which stores information about which sites are bonded to which
    \param JJ The coupling of the spin operator on the bond
    
    \result The value of <a|H|b> on the diagonal 
*/
__device__ float HDiagPartHeisenberg2D(const int bra, int lattice_Size, int3* d_Bond, const float JJ);

/*!
    \brief A GPU function which finds and inserts the values, row indices, and column indices of diagonal elements into Hamiltonian storage arrays
   
    In the Heisenberg case, the only nonzero diagonal operator is <i>S<sub>z</sub></i> over the bonds. Since the diagonal operations are different than those for the off-diagonal elements, they are sequestered in their own kernel to avoid serializing threads in FillSparseHeisenberg() or a more complicated kernel in general. Each thread is assigned a ket, determines whether this ket is in the Hilbert space/sector, and then calls HDiagPartHeisenberg() to find the value of the diagonal element. If the ket is in the Hilbert space or sector, this value and the true row and column indices are written back to global memory. A flag (1) is also written for element counting purposes. If not, the value is still written but the row and column indices are set to be outside the dimension of the Hilbert space/sector. This is for sorting purposes. The counting flag is set to 0.
    
    \ingroup Heisenberg
    
    \param d_basis An array storing the kets which are in-sector
    \param H A struct which contains arrays to be filled with information about the Hamiltonian's diagonal elements
    \param d_Bond An array which contains information about which sites are bonded to which
    \param data A struct containing information about the simulation parameters, such as number of lattice sites, coupling constants, and S_z sector
*/

__global__ void FillDiagonalsHeisenberg(int* d_basis, f_hamiltonian H, int* d_Bond, parameters data);

/*!
    \brief A GPU function which finds and inserts the values, row indices, and column indices of off-diagonal elements into Hamiltonian storage arrays 
    
    This function processes <i>S<sub>x</sub><sup>i</sup>S<sub>x</sub><sup>j</sup></i> and <i>S<sub>y</sub><sup>i</sup>S<sub>y</sub><sup>j</sup></i>. Since these operators can be expressed instead as combinations of <i>S<sub>+</sub></i> and <i>S<sub>-</sub></i>, this language (which is more convenient) is used instead. Each thread is assigned a row index and a site:

\verbatim
int ii = (blockDim.x/(2*lattice_Size))*blockIdx.x + threadIdx.x/(2*lattice_Size) + offset*(blockDim.x/(2*lattice_Size));
int T0 = threadIdx.x%(2*lattice_Size);
int site = T0 / lattice_Size;
\endverbatim    

The thread determines whether the index it's been assigned is within the dimensions of the Hilbert space/sector and whether the site is within the lattice size. If so, it then copies the bond information from global to shared memory since it will be referenced quite often. 

\verbatim
if( ii < dim )
{
    if (T0 < 2*lattice_Size)
    {   
    
        (tempbond[site]).x = d_Bond[site];
        (tempbond[site]).y = d_Bond[lattice_Size + site];
        (tempbond[site]).z = d_Bond[2*lattice_Size + site];

\endverbatim

An in-sector ket is extracted from d_basis and the spins on the thread's site and the site(s) it is bonded to are flipped as well. This generates the only bra that will give a non-zero matrix element. The thread ensures that this bra's position in d_basis is greater than the row index assigned at the beginning. This forces only the upper triangle of the matrix to be generated, which prevents duplicate elements from forming.

\verbatim
tempi = d_basis[ii];
s = (tempbond[site]).x;
tempod[threadIdx.x] = tempi;
tempod[threadIdx.x] ^= (1<<s);
s = (tempbond[site]).y;
tempod[threadIdx.x] ^= (1<<s);

compare = (d_basis_Position[tempod[threadIdx.x]] > ii);
temppos[threadIdx.x] = (compare) ? d_basis_Position[tempod[threadIdx.x]] : dim;
tempval[threadIdx.x] = HOffBondXHeisenberg(site, tempi, data.J1);
\endverbatim

The thread then inserts the element value, the row index, and the column index of the bra into global memory, interchanging the indices if the thread has been assigned to handle the complex conjugates. Since this Hamiltonian is entirely real-valued, it isn't necessary to conjugate the element values. Finally, the array of flags is set depending on whether the element is in-sector, upper-triangular, etc.

\verbatim
count += (int)compare;
rowtemp = (T0/lattice_Size) ? ii : temppos[threadIdx.x];
rowtemp = (compare) ? rowtemp : 2*dim;

H.vals[ ii*stride + 4*site + (T0/lattice_Size) + start ] = tempval[threadIdx.x]; 
H.cols[ ii*stride + 4*site + (T0/lattice_Size) + start ] = (T0/lattice_Size) ? temppos[threadIdx.x] : ii;
H.rows[ ii*stride + 4*site + (T0/lattice_Size) + start ] = rowtemp;
H.set[ ii*stride + 4*site + (T0/lattice_Size) + start ] = (int)compare;
\endverbatim

    \ingroup Heisenberg
    
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
    
    \ingroup XY
    
    \param si The site whose bond we are looking at
    \param bra The state resulting from applying the spin operator
    \param JJ The coupling of the spin operator on the bond 
    
    \result The value of <a|H|b> on the x-bond 
*/
__device__ float HOffBondXXY(const int si, const int bra, const float JJ);

/*!
    \brief A GPU function which finds the value of the spin operator on the bond in the y direction
    
    \ingroup XY
    
    \param si The site whose bond we are looking at
    \param bra The state resulting from applying the spin operator
    \param JJ The coupling of the spin operator on the bond 
    
    \result The value of <a|H|b> on the y-bond 
*/
__device__ float HOffBondYXY(const int si, const int bra, const float JJ);

/*!
    \brief A GPU function which finds the value of the spin operator on the bond on the diagonal
    
    \ingroup XY
    
    \param bra The state resulting from applying the spin operator
    \param lattice_Size The number of sites in the lattice
    \param d_Bond Array which stores information about which sites are bonded to which
    \param JJ The coupling of the spin operator on the bond 
*/
__device__ float HDiagPartXY(const int bra, int lattice_Size, int3* d_Bond, const float JJ);

/*!
    \brief A GPU function which finds and inserts the values, row indices, and column indices of diagonal elements into Hamiltonian storage arrays
   
    Since the XY model has no diagonal terms, this function simply ensures all diagonal elements are zero-valued.
     
    \ingroup XY
    
    \param d_basis An array storing the kets which are in-sector
    \param H A struct which contains arrays to be filled with information about the Hamiltonian's diagonal elements
    \param d_Bond An array which contains information about which sites are bonded to which
    \param data A struct containing information about the simulation parameters, such as number of lattice sites, coupling constants, and S_z sector
*/
__global__ void FillDiagonalsXY(int* d_basis, f_hamiltonian H, int* d_Bond, parameters data);

/*!
    \brief A GPU function which finds and inserts the values, row indices, and column indices of off-diagonal elements into Hamiltonian storage arrays 
    
    This function processes <i>S<sub>x</sub><sup>i</sup>S<sub>x</sub><sup>j</sup></i> and <i>S<sub>y</sub><sup>i</sup>S<sub>y</sub><sup>j</sup></i>. Since these operators can be expressed instead as combinations of <i>S<sub>+</sub></i> and <i>S<sub>-</sub></i>, this language (which is more convenient) is used instead. Each thread is assigned a row index and a site:

\verbatim
int ii = (blockDim.x/(2*lattice_Size))*blockIdx.x + threadIdx.x/(2*lattice_Size) + offset*(blockDim.x/(2*lattice_Size));
int T0 = threadIdx.x%(2*lattice_Size);
int site = T0 / lattice_Size;
\endverbatim    

The thread determines whether the index it's been assigned is within the dimensions of the Hilbert space/sector and whether the site is within the lattice size. If so, it then copies the bond information from global to shared memory since it will be referenced quite often. 

\verbatim
if( ii < dim )
{
    if (T0 < 2*lattice_Size)
    {   
    
        (tempbond[site]).x = d_Bond[site];
        (tempbond[site]).y = d_Bond[lattice_Size + site];
        (tempbond[site]).z = d_Bond[2*lattice_Size + site];

\endverbatim

An in-sector ket is extracted from d_basis and the spins on the thread's site and the site(s) it is bonded to are flipped as well. This generates the only bra that will give a non-zero matrix element. The thread ensures that this bra's position in d_basis is greater than the row index assigned at the beginning. This forces only the upper triangle of the matrix to be generated, which prevents duplicate elements from forming.

\verbatim
tempi = d_basis[ii];
s = (tempbond[site]).x;
tempod[threadIdx.x] = tempi;
tempod[threadIdx.x] ^= (1<<s);
s = (tempbond[site]).y;
tempod[threadIdx.x] ^= (1<<s);

compare = (d_basis_Position[tempod[threadIdx.x]] > ii);
temppos[threadIdx.x] = (compare) ? d_basis_Position[tempod[threadIdx.x]] : dim;
tempval[threadIdx.x] = HOffBondXHeisenberg(site, tempi, data.J1);
\endverbatim

The thread then inserts the element value, the row index, and the column index of the bra into global memory, interchanging the indices if the thread has been assigned to handle the complex conjugates. Since this Hamiltonian is entirely real-valued, it isn't necessary to conjugate the element values. Finally, the array of flags is set depending on whether the element is in-sector, upper-triangular, etc.

\verbatim
count += (int)compare;
rowtemp = (T0/lattice_Size) ? ii : temppos[threadIdx.x];
rowtemp = (compare) ? rowtemp : 2*dim;

H.vals[ ii*stride + 4*site + (T0/lattice_Size) + start ] = tempval[threadIdx.x]; 
H.cols[ ii*stride + 4*site + (T0/lattice_Size) + start ] = (T0/lattice_Size) ? temppos[threadIdx.x] : ii;
H.rows[ ii*stride + 4*site + (T0/lattice_Size) + start ] = rowtemp;
H.set[ ii*stride + 4*site + (T0/lattice_Size) + start ] = (int)compare;
\endverbatim
    
    \ingroup XY
    
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
    
    \ingroup TFIM
    
    \param si The site whose bond we are looking at
    \param bra The state resulting from applying the spin operator
    \param JJ The coupling of the spin operator on the bond 
    
    \result The value of <a|H|b> on the x-bond 
*/
__device__ float HOffBondXTFI(const int si, const int bra, const float JJ);

/*!
    \brief A GPU function which finds the value of the spin operator on the bond in the x direction
    
    \ingroup TFIM
    
    \param si The site whose bond we are looking at
    \param bra The state resulting from applying the spin operator
    \param JJ The coupling of the spin operator on the bond 
    
    \result The value of <a|H|b> on the y-bond 
*/
__device__ float HOffBondYTFI(const int si, const int bra, const float JJ);

/*!
    \brief A GPU function which finds the value of the spin operator on the bond on the diagonal in one dimension
    
    This function computes the value of <i>S<sub>z</sub><sup>i</sup>S<sub>z</sub><sup>j</sup></i> over all bonded pairs <i>(i,j)</i>. If <i>i</i> and <i>j</i> have ferromagnetic ordering (they are aligned) then <i>S<sub>z</sub><sup>i</sup>S<sub>z</sub><sup>j</sup></i> = 1/4. If they have antiferrmagnetic ordering (they are anti-aligned) <i>S<sub>z</sub><sup>i</sup>S<sub>z</sub><sup>j</sup></i> = 0. The final value of the diagonal element is the coupling constant multiplied by the sum of all the bond <i>S<sub>z</sub></i> values.
    
    \ingroup TFIM
    
    \param bra The state resulting from applying the spin operator
    \param lattice_Size The number of sites in the lattice
    \param d_Bond Array which stores information about which sites are bonded to which
    \param JJ The coupling of the spin operator on the bond 
*/
__device__ float HDiagPartTFI1D(const int bra, int lattice_Size, int3* d_Bond, const float JJ);

/*!
    \brief A GPU function which finds the value of the spin operator on the bond on the diagonal in two dimensions
    
    This function computes the value of <i>S<sub>z</sub><sup>i</sup>S<sub>z</sub><sup>j</sup></i> over all bonded pairs <i>(i,j)</i>. If <i>i</i> and <i>j</i> have ferromagnetic ordering (they are aligned) then <i>S<sub>z</sub><sup>i</sup>S<sub>z</sub><sup>j</sup></i> = 1/4. If they have antiferrmagnetic ordering (they are anti-aligned) <i>S<sub>z</sub><sup>i</sup>S<sub>z</sub><sup>j</sup></i> = 0. The final value of the diagonal element is the coupling constant multiplied by the sum of all the bond <i>S<sub>z</sub></i> values.
    
    \ingroup TFIM
    
    \param bra The state resulting from applying the spin operator
    \param lattice_Size The number of sites in the lattice
    \param d_Bond Array which stores information about which sites are bonded to which
    \param JJ The coupling of the spin operator on the bond 
*/
__device__ float HDiagPartTFI2D(const int bra, int lattice_Size, int3* d_Bond, const float JJ);

/*!
    \brief A GPU function which finds and inserts the values, row indices, and column indices of diagonal elements into Hamiltonian storage arrays
    
    In the Transverse Field Ising case, the only nonzero diagonal operator is <i>S<sub>z</sub></i> over the bonds. Since the diagonal operations are different than those for the off-diagonal elements, they are sequestered in their own kernel to avoid serializing threads in FillSparseTFI() or a more complicated kernel in general. Each thread is assigned a ket, determines whether this ket is in the Hilbert space/sector, and then calls HDiagPartTFI() to find the value of the diagonal element. If the ket is in the Hilbert space or sector, this value and the true row and column indices are written back to global memory. A flag (1) is also written for element counting purposes. If not, the value is still written but the row and column indices are set to be outside the dimension of the Hilbert space/sector. This is for sorting purposes. The counting flag is set to 0.
    
    \ingroup TFIM
    
    \param d_basis An array storing the kets which are in-sector
    \param H A struct which contains arrays to be filled with information about the Hamiltonian's diagonal elements
    \param d_Bond An array which contains information about which sites are bonded to which
    \param data A struct containing information about the simulation parameters, such as number of lattice sites, coupling constants, and S_z sector
*/
__global__ void FillDiagonalsTFI(int* d_basis, f_hamiltonian H, int* d_Bond, parameters data);

/*!
    \brief A GPU function which finds and inserts the values, row indices, and column indices of off-diagonal elements into Hamiltonian storage arrays 
    
    This function processes <i>S<sub>x</sub><sup>i</sup></i>. Since this operator can be expressed instead as combinations of <i>S<sub>+</sub></i> and <i>S<sub>-</sub></i>, this language (which is more convenient) is used instead. This operator is single site, so bond information is not needed. Each thread is assigned a row index and a site. 
    
\verbatim
   int ii = ( blockDim.x / ( 2 * lattice_Size ) ) * blockIdx.x + threadIdx.x / ( 2 * lattice_Size ) + offset;
   int T0 = threadIdx.x % ( 2 * lattice_Size );
\endverbatim

The thread determines whether the index it's been assigned is within the dimensions of the Hilbert space/sector and whether the site is within the lattice size. 

\verbatim
if( ii < dim )
    {
        if ( T0 < 2 * lattice_Size )
        {
\endverbatim

A ket is extracted from d_basis and the spin on the thread's site is flipped. This generates the only bra that will give a non-zero matrix element. The thread ensures that this bra's position in d_basis is greater than the row index assigned at the beginning. This forces only the upper triangle of the matrix to be generated, which prevents duplicate elements from forming. 

\verbatim
tempod[threadIdx.x] = tempi;

//-----------------Horizontal bond ---------------
temppos[ threadIdx.x ] = ( tempod[threadIdx.x] ^ ( 1 << site ) );
//flip the site-th bit of row - applying the sigma_x operator
compare = ( temppos[ threadIdx.x ] > ii ) && ( temppos[ threadIdx.x ] < dim );
temppos[ threadIdx.x ] = compare ? temppos[ threadIdx.x ] : dim + 1;
tempval[ threadIdx.x ] = HOffBondXTFI(site, tempi, data.J2);
\endverbatim

The thread then inserts the element value, the row index, and the column index of the bra into global memory, interchanging the indices if the thread has been assigned to handle the complex conjugates. Since this Hamiltonian is entirely real-valued, it isn't necessary to conjugate the element values. Finally, the array of flags is set depending on whether the element is in-sector, upper-triangular, etc.
   
\verbatim
rowtemp = ( T0 / lattice_Size ) ? ii : temppos[ threadIdx.x ];
rowtemp = compare ? rowtemp : dim + 1;
temppos[ threadIdx.x ] = ( T0 / lattice_Size) ? temppos[ threadIdx.x ] : ii;
temppos[ threadIdx.x ] = compare ? temppos[threadIdx.x] : dim + 1;

//----Putting everything back into GPU main memory-----------

H.vals[ ii*stride + 2 * site + ( T0 / lattice_Size ) + start ] = tempval[ threadIdx.x ];
H.cols[ ii*stride + 2 * site + ( T0 / lattice_Size ) + start ] = temppos[ threadIdx.x ];
H.rows[ ii*stride + 2 * site + ( T0 / lattice_Size ) + start ] = rowtemp;
H.set[ ii*stride + 2 * site + ( T0 / lattice_Size ) + start ] = (int)compare;
\endverbatim

    \ingroup TFIM
    
    \param d_basis_Position Array which stores the location of in-sector kets in d_basis
    \param d_basis Array which stores the in-sector kets
    \param H Struct which contains arrays to be filled with information about the Hamiltonian's off-diagonal elements
    \param d_Bond An array which contains information about which sites are bonded to which
    \param data A struct containing information about the simulation parameters, such as number of lattice sites, coupling constants, and S_z sector
    \param offset How many blocks of off-diagonal elements have already been processed by other runs of the kernel
*/
__global__ void FillSparseTFI(int* d_basis_Position, int* d_basis, f_hamiltonian H, int* d_Bond, parameters data, int offset);
