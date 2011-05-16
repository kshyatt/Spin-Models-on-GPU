// Katharine Hyatt
// A set of functions to implement the Lanczos method for a generic Hamiltonian
// Based on the codes Lanczos_07.cpp and Lanczos07.h by Roger Melko
//-------------------------------------------------------------------------------

#include<lanczos.h>
// h_ means this variable is going to be on the host (CPU)
// d_ means this variable is going to be on the device (GPU)
// s_ means this variable is shared between threads on the GPU
// The notation <<x,y>> before a function defined as global tells the GPU how many threads per block to use
// and how many blocks per grid to use
// blah.x means the real part of blah, if blah is a data type from cuComplex.h
// blah.y means the imaginary party of blah, if blah is a data type from cuComplex.h
// threadIdx.x (or block) means the "x"th thread from the left in the block (or grid)
// threadIdx.y (or block) means the "y"th thread from the top in the block (or grid)

//Function vecdiff: calculates the difference between some vectors, two of which is multiplied by a scalar
//Implements w = x - a*y - b*z 
//-------------------------------------------------------------------------------------------------------
//Input: w, a "dummy pointer" to the vector that is changed
//       x, the vector that a*y and b*z are subtracted from 
//       alpha, the scalar the first subtracted vector is multiplied by
//       y, the first subtracted vector
//       beta, the scalar the second subtraced vector is multiplied by
//       z, the second subtracted vector
//       n, the number of elements in all the vectors
//       Note: no control is put in here to make sure all vectors are the same size. 
//------------------------------------------------------------------------------------------------------
//Output: w, the result of the subtractions
//        All other quantities remain unchanged
//------------------------------------------------------------------------------------------------------
__global__ void vecdiff(cuDoubleComplex* w, cuDoubleComplex* x, cuDoubleComplex alpha, cuDoubleComplex* y, cuDoubleComplex beta, cuDoubleComplex* z, int n){

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < n) {
    w[i] = cuCsub(x[i],cuCsub(cuCmul(alpha,y[i]),cuCmul(beta,z[i]))); //this is the dirtiest thing ever
  }
  __syncthreads();
}

//Function assignr: assigns the real parts of an array of double complex numbers the value of some double
//-------------------------------------------------------------------------------------------------------
//Input: a, a vector of double precision complex numbers whose real parts we would like to change
//       b, the real number that will become the real part of the complex numbers in a
//       n, the number of elements in a
//-------------------------------------------------------------------------------------------------------
//Output: a, the vector of complex numbers whose real parts have been changed
//        All other quantities are unchanged
//------------------------------------------------------------------------------------------------------- 
__global__ void assignr(cuDoubleComplex* a, double b, int n){
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < n){
    a[i] = make_cuDoubleComplex(b,0.);
  }
}

//Function complextodoubler: assigns the real parts of complex numbers in an array to doubles in another array
//------------------------------------------------------------------------------------------------------------
//Input: a, the vector of complex numbers whose real parts we are extracting
//       b, the vector of doubles that will hold the real parts
//       n, the number of elements in the vectors
//------------------------------------------------------------------------------------------------------------
//Output: b, the vector of doubles now holding the real parts
//        All other quantities are unchanged
//------------------------------------------------------------------------------------------------------------
__global__ void complextodoubler(cuDoubleComplex* a, double* b, int n){
  int i = blockDim.x*blockIdx.x + threadIdx.x;

  if(i <= n){
    b[i] = a[i].x; 
  }
}

//Same as above, but in this case the parts are shifted by one space in the vector
__global__ void complextodoubler2(cuDoubleComplex* a, double* b, int n){
  int i = blockDim.x*blockIdx.x + threadIdx.x + 1;

  if(i <= n){
    b[i-1] = a[i].x;
  }
  if(i == n+1){
    b[i-1] = 0.;
  }
} 

__global__ void zero(double* a, int m){
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  int j = blockDim.y*blockIdx.y + threadIdx.y;

  if ( i< m && j < m){
    a[i][j] = 0.;
  }
}
// Note: to get the identity matrix, apply the fuction zero above first
__global__ void identity(double* a, int m){
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  
  if (i < m ){

    a[i][i] = 1.
  }
}

__global__ void assign(double* a, double b, int n){
  int i = blockDim.x*blockIdx.x + threadIdx.x;

  if (i < n){
    a[i] = b;
  }
}

__global__ void arraysalloc(cuDoubleComplex** a, int n, int m){
  int i = threadIdx.x + m;
  a[i] = (cuDoubleComplex*)malloc(n*sizeof(cuDoubleComplex));
}

//Function lanczos: takes a hermitian matrix H, tridiagonalizes it, and finds the n smallest eigenvalues - this version only returns eigenvalues, not
// eigenvectors. Doesn't use sparse matrices yet either, derp. Should be a very simple change to make using CUSPARSE, which has functions for operations
// between sparse matrices and dense vectors
//---------------------------------------------------------------------------------------------------------------------------------------------------
// Input: h_H, a Hermitian matrix of complex numbers (not yet sparse)
//        dim, the dimension of the matrix
//        max_Iter, the starting number of iterations we'll try
//        num_Eig, the number of eigenvalues we're interested in seeing
//        conv_req, the convergence we'd like to see
//---------------------------------------------------------------------------------------------------------------------------------------------------
// Output: h_ordered, the array of the num_Eig smallest eigenvalues, ordered from smallest to largest
//---------------------------------------------------------------------------------------------------------------------------------------------------        
void lanczos(const cuDoubleComplex* h_H, const int dim, int max_Iter, const int num_Eig, const double conv_req){

  size_t h_pitch;
  size_t d_pitch = h_pitch;

  cuDoubleComplex* d_H;

  cudaMallocPitch(&d_H, &h_pitch, dim*sizeof(cuDoubleComplex), dim); //allocating memory in the GPU for our matrix

  cudaMemcpy2D(d_H, d_pitch, &H, h_pitch, dim, dim, cudaMemcpyHosttoDevice ); //copying the matrix into the GPU
  // the above memory code could be total lunacy

  //Now that I have the Hamiltonian on the GPU, it's time to start generating eigenvectors

  cuDoubleComplex* d_a; //these are going to store the elements of the tridiagonal matrix
  cuDoubleComplex* d_b; //they have to be cuDoubleComplex because that's the only input type the cublas functions I need will take

  cudaMalloc(&d_a; max_Iter*sizeof(cuDoubleComplex));
  cudaMalloc(&d_b; max_Iter*sizeof(cuDoubleComplex));

  int tpb = 256; //threads per block - a conventional number
  int bpg = (dim + tpb - 1)/tpb; //blocks per grid

  //Making the "random" starting vector

  cuDoubleComplex** d_eigen_Array; //this thing is an array of pointers to the eigenvectors 
  cudaMalloc(&d_eigen_Array; max_Iter*sizeof(cuDoubleComplex*)); // making the pointer array
  arraysalloc<<1, max_Iter>>(d_eigen_Array, dim, 0); //time to make the actual arrays of the eigenvectors

  cuDoubleComplex* d_v_Start; //the three vectors we'll use later
  cuDoubleComplex* d_v_Mid;
  cuDoubleComplex* d_v_End;

  cudaMalloc(&d_v_Start, dim*sizeof(cuDoubleComplex));
  cudaMalloc(&d_v_Mid, dim*sizeof(cuDoubleComplex));
  cudaMalloc(&d_v_End, dim*sizeof(cuDoubleComplex));
  
  assignr<<bpg,tpb<>>(d_eigen_Array[0], 1., dim); //assigning the values of the "random" starting vector

  cuDoubleComplex* alpha;
  cudaMalloc(&alpha, sizeof(cuDoubleComplex));
  cuDoubleComplex* beta;
  cudaMalloc(&beta, sizeof(cuDoubleComplex));
  cuDoubleComplex *alpha = make_cuDoubleComplex(1.,0.);
  cuDoubleComplex *beta = make_cuDoubleComplex(0.,0.); //I heard you like memory management

  Hoperate(d_H, d_eigen_Array[0], d_eigen_Array[1], alpha, beta, dim);
  //*********************************************************************************************************
  // This is just the first steps so I can do the rest  
  d_a[0] = cublasZdotc(dim, c[0], sizeof(cuDoubleComplex), c[1], sizeof(cuDoubleComplex));
  d_b[0] = make_cuDoubleComplex(0.,0.);

  cuDoubleComplex* y;
  cudaMalloc(&y, dim*sizeof(cuDoubleComplex));
  
  assignr<<bpg,tpb>>(y,0., dim); //a dummy vector of 0s that i can stick in my functions

  vecdiff<<bpg,tpb>>(d_eigen_Array[1], d_eigen_Array[1], d_a[0], d_eigen_Array[0], y[0], y);
  d_b[1] = make_cuDoubleComplex(sqrt(cublasDznrm2(dim, d_eigen_Array[1], sizeof(cuDoubleComplex))),0.);
  // this function (above) takes the norm
  
  cuDoubleComplex gamma = make_cuDoubleComplex(1./d_b[1].x,0.); //alpha = 1/beta in v1 = v1 - alpha*v0

  cublasZaxpy(dim, gamma, d_eigen_Array[1], sizeof(cuDoubleComplex), y, sizeof(cuDoubleComplex)); // function performs a*x + y

  //Now we're done the first round!
  //*********************************************************************************************************

  double* d_ordered;
  cudaMalloc(&d_ordered, num_Eig*sizeof(double));

  assign<<bpg,tpb>>(d_ordered, 0., num_Eig);

  double gs_Energy = 1.; //the lowest energy

  int returned;

  int iter = 0;

  // In the original code, we started diagonalizing from iter = 5 and above. I start from iter = 1 to minimize issues of control flow
  double* d_diag;
  double* d_offdia;

  cudaMalloc(&d_diag, num_Eig*sizeof(double));
  cudaMalloc(&d_offdia, num_Eig*sizeof(double));

  while( abs(gs_Energy - d_ordered[num_Eig-1])> conv_req){ //this is a cleaner version than what was in the original - way fewer if statements

    iter++;

    Hoperate(d_H, d_eigen_Array[iter], d_eigen_Array[iter+1], alpha, beta, dim);

    d_a[iter] = cublasZdotc(dim, d_eigen_Array[iter], sizeof(cuDoubleComplex), d_eigen_Array[iter + 1], sizeof(cuDoubleComplex));

    vecdiff<<bpg,tpb>>(d_eigen_Array[iter+1], d_eigen_Array[iter+1], d_a[iter], d_eigen_Array[iter], d_b[iter], c[iter - 1]);

    d_b[iter+1] = make_cuDoubleComplex(sqrt(cublasDznrm2(dim, d_eigen_Array[iter+1], sizeof(cuDoubleComplex))),0.);
    
    gamma = make_cuDoubleComplex(1./d_b[iter+1].x,0.);
    cublasZaxpy(dim, gamma, d_eigen_Array[iter+1], sizeof(cuDoubleComplex), y, sizeof(cuDoubleComplex));
    
    //cublasZcopy(dim, d_v_Mid, sizeof(cuDoubleComplex), d_v_Start, sizeof(cuDoubleComplex)); //switching my vectors around for the next iteration
    //cublasZcopy(dim, d_v_End, sizeof(cuDoubleComplex), d_v_Mid, sizeof(cuDoubleComplex)); unnecesarry now that i'm using the array of pointers

    d_diag[iter] = 0.; //adding another spot in the tridiagonal matrix representation
    d_offdia[iter] = 0.;

    complextodoubler<<bpg,tpb>>(d_diag, d_a, iter);
    complextodoubler2<<bpg,tpb>>(d_offdia, d_b, iter);

    double* d_H_eigen;
    size_t d_eig_pitch;

    cudaMallocPitch(&d_H_eigen, &d_eig_pitch, iter*sizeof(double), iter);
    zero<<bpg,tpb>>(d_H_eigen, iter);
    identity<<bpg,tpb>>(d_H_eigen, iter); //set this matrix to the identity

    returned = tqli(d_diag, d_offdia, iter + 1, d_H_eigen, 0); //tqli is in a separate file   

    assign<<tpb,bpg>>(d_ordered, d_diag[0], num_Eig);
    
    for(int i = 1, i < d_eigen_Array.size(), i++){ //todo: rewrite this as a setup where if you want 
      for(int j = 0, j< d_ordered.size(), j++){// n smallest eigenvalues, you take the first n
        if (d_diag[i]< d_ordered[j]){ //elements, sort them, then add one element at a time 
          d_ordered[j] = d_diag[i]; // and binary search to see if it is smaller than any other
          break;
        }
      }
    }

    gs_Energy = d_ordered[num_Eig - 1];

    if (iter == d_eigen_Array.size() - 2){// have to use this or d_b will overflow
      //this stuff here is used to resize the main arrays in the case that we aren't converging quickly enough
	cuDoubleComplex* temp;
        cudaMalloc(&temp, (2*d_eigen_Array.size() + 1)*sizeof(cuDoubleComplex));
        
        cudaMemcpy(temp, d_a, d_eigen_Array.size()*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        
        cudaFree(d_a);
        cudaMalloc(&d_a, (2*d_eigen_Array.size() + 1)*sizeof(cuDoubleComplex));
        cudaMemcpy(d_a, temp, (2*d_eigen_Array.size() + 1)*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        
        cudaMemcpy(temp, d_b, d_eigen_Array.size()*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);

        cudaFree(d_b);
        cudaMalloc(&d_b, (2*d_eigen_Array.size() + 1)*sizeof(cuDoubleComplex));
        cudaMemcpy(d_b, temp, (2*d_eigen_Array.size() + 1)*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
          
        cudaFree(temp);
        cudaMalloc(&temp, (2*d_eigen_Array.size() + 1)*sizeof(cuDoubleComplex*));
        
        cudaMemcpy(temp, d_eigen_Array, d_eigen_Array.size()*sizeof(cuDoubleComplex*), cudaMemcpyDeviceToDevice);

        int temp2 = d_eigen_Array.size();

        cudaFree(d_eigen_Array);
        cudaMalloc(&d_eigen_Array, (2*temp2 + 1)*sizeof(cuDoubleComplex*));
        cudaMemcpy(d_eigen_Array, temp, (2*temp2 + 1)*sizeof(cuDoubleComplex*), cudaMemcpyDeviceToDevice);

        arraysalloc<<1, temp2 + 1>>(d_eigen_Array, dim, temp2);
                
        cudaFree(temp); 
        //resizing

    }   
  } 

  double* h_ordered;

  cudaHostMalloc(&h_ordered, num_Eig*sizeof(double)); //a place to put the eigenvalues on the CPU

  cudaMemcpy(h_ordered, d_ordered, num_Eig*sizeof(double), cudaMemcpyDevicetoHost); // moving the eigenvalues over

  for(int i = 0; i < num_Eig, i++){
    printf(h_ordered[i]);
  } //write out the eigenenergies

  cudaFree(alpha);
  cudaFree(beta);
  cudaFree(d_a);
  cudaFree(d_b); //dropping stuff off
  // call the expectation values function
  
  // time to copy back all the eigenvectors
  int* sizeptr;
  cudaMemcpy(sizeptr, &d_eigen_Array.size(), sizeof(int), cudaMemcpyDeviceToHost);
  
  max_Iter = *sizeptr;

  cuDoubleComplex** h_eigen_Array;
  cudaMallocHost(&h_eigen_Array, max_Iter*sizeof(cuDoubleComplex*));
  
  for(int i = 0; i < max_Iter; i++){
     cudaMallocHost(h_eigen_Array[i], dim*sizeof(cuDoubleComplex*));
     cudaMemcpy(h_eigen_Array[i], d_eigen_Array[i], dim*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
  } // now the eigenvectors are available on the host CPU

}
// things left to do:
// write a thing (separate file) to call routines to find expectation values, should be faster on GPU 
// make the tqli thing better!

//Function Hoperate: applies H to some vector to give a = H*b
//NOTE: this function CANNOT be called from the CPU.
//Only variables and functions on the GPU may access it.
//------------------------------------------------------------
//Input: H, a matrix of complex numbers
//       v0, the vector H is applied to
//       v1, a pointer to  the output vector
//       alpha, a dummy variable
//       beta, a dummy variable
//       dim, the number of elements in the vectors and the length of one side of H
//------------------------------------------------------------
//Output: v1, the result of H*v0
//        All other quantities are unchanged
//------------------------------------------------------------
__device__ void Hoperate(cuDoubleComplex* H, cuDoubleComplex* v0, cuDoubleComplex v1, cuDoubleComplex alpha, cuDoubleComplex beta, int dim){
  
  v1 = cublasZgemv(N, dim, dim, alpha, &H, m*sizeof(cuDoubleComplex), v0, sizeof(cuDoubleComplex), beta, v1, sizeof(cuDoubleComplex));

}

//fin
