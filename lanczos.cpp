#include<lanczos.h>
// h_ means this variable is going to be on the host (CPU)
// d_ means this variable is going to be on the device (GPU)
// s_ means this variable is shared between threads on the GPU


//Function vecdiff: calculates the difference between some vectors, one of which is multiplied by a scalar
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
    w[i] = x[i] - alpha*y[i] - beta*z[i];
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
  __syncthreads();
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
__global__ void complextodoubler(vector<cuDoubleComplex>* a, vector<double>* b, int n){
  int i = blockDim.x*blockIdx.x + threadIdx.x;

  if(i <= n){
    b[i] = a[i].x;
  }

  __syncthreads();
}

//Same as above, but in this case the parts are shifted by one space in the vector
__global__ void complextodoubler2(vector<cuDoubleComplex>* a, vector<double>* b, int n){
  int i = blockDim.x*blockIdx.x + threadIdx.x + 1;

  if(i <= n){
    b[i-1] = a[i].x;
  }
  
  b[n] = 0.;

  __syncthreads();
} 

__global__ void identity(double* a, int m){
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  int j = blockDim.y*blockIdx.y + threadIdx.y;

  if (i < m && j < m){

    if (i = j){
      a[i][j] = 1.;
    }

    else{
      a[i][j] = 0.;
    }
  }
}

__global__ void assign(double* a, double b, int n){
  int i = blockDim.x*blockIdx.x + threadIdx.x;

  if (i < n){
    a[i] = b;
  }
}

//Function lanczos: takes a hermitian matrix H, tridiagonalizes it, and finds the n smallest eigenvalues - this version only returns eigenvalues, not
// eigenvectors. Doesn't use sparse matrices yet either, derp. Should be a very simple change to make using CUSPARSE, which has functions for operations
// between sparse matrices and dense vectors
//---------------------------------------------------------------------------------------------------------------------------------------------------
// Input: h_H, a Hermitian matrix of complex numbers (not yet sparse)
//        dim, the dimension of the matrix
//        num_Eig, the number of eigenvalues we're interested in seeing
//        conv_req, the convergence we'd like to see
//---------------------------------------------------------------------------------------------------------------------------------------------------
// Output: h_ordered, the array of the num_Eig smallest eigenvalues, ordered from smallest to largest
//---------------------------------------------------------------------------------------------------------------------------------------------------        
void lanczos(const cuDoubleComplex* h_H, const int dim, const int num_Eig, const double conv_req){

  size_t h_pitch;
  size_t d_pitch = h_pitch;\

  cuDoubleComplex* d_H;

  cudaMallocPitch(&d_H, &h_pitch, dim*sizeof(cuDoubleComplex), dim);

  cudaMemcpy2D(d_H, d_pitch, &H, h_pitch, dim, dim, cudaMemcpyHosttoDevice );
  // the above memory code could be total lunacy

  //Now that I have the Hamiltonian on the GPU, it's time to start generating eigenvectors

  vector<cuDoubleComplex> d_a; //these are going to store the elements of the tridiagonal matrix
  vector<cuDoubleComplex> d_b;

  //Making the "random" starting vector

  cuDoubleComplex* d_v_Start;
  cuDoubleComplex* d_v_Mid;
  cuDoubleComplex* d_v_End;

  cudaMalloc(&d_v_Start, dim*sizeof(cuDoubleComplex));
  cudaMalloc(&d_v_Mid, dim*sizeof(cuDoubleComplex));
  cudaMalloc(&d_v_End, dim*sizeof(cuDoubleComplex));
  
  assignr<<tpb,bpg>>(d_v_Start, 1., dim);

  Hoperate(H, d_v_Start, d_v_Mid, dim);
  //*********************************************************************************************************
  // This is just the first steps so I can do the rest  
  d_a.push_back(cublasZdotc(dim, d_v_Start, sizeof(cuDoubleComplex), d_v_Mid, sizeof(cuDoubleComplex)));
  d_b.push_bach(make_cuDoubleComplex(0.,0.));

  cuDoubleComplex* y;
  cudaMalloc(&y, dim*sizeof(cuDoubleComplex));
  
  assignr<<tpb,bpg>>(y,0., dim);

  cuDoubleComplex* beta;
  cudaMalloc(&beta, sizeof(cuDoubleComplex));
  *beta = make_cuDoubleComplex(0.,0.);
  
  int tpb = 256; //threads per block
  int bpg = (dim + tpb - 1)/tpb; //blocks per grid
  vecdiff(d_v_Mid, d_v_Mid, d_a[0], d_v_Start, y[0], y);
  d_b.push_back(make_cuDoubleComplex(sqrt(cublasDznrm2(dim, d_v_Mid, sizeof(cuDoubleComplex))),0.));
  // this function (above) takes the norm
  
  cuDoubleComplex* alpha;
  cudaMalloc(&alpha, sizeof(cuDoubleComplex));
  cuDoubleComplex *alpha = make_cuDoubleComplex(1./d_b[1].x,0.);

  cublasZaxpy(dim, alpha, d_v_Mid, sizeof(cuDoubleComplex), y, sizeof(cuDoubleComplex)); // function performs a*x + y

  //Now we're done the first round!
  //*********************************************************************************************************

  double* d_ordered;
  cudaMalloc(&d_ordered, num_Eig*sizeof(double));

  assign<<tpb,bpg>>(d_ordered, 0., num_Eig);

  double gs_Energy = 1.;

  int returned;

  int iter = 0;

  // In the original code, we started diagonalizing from iter = 5 and above. I start from iter = 1 to minimize issues of control flow
  vector<double> d_diag;
  vector<double> d_offdia;


  while( abs(gs_Energy - d_ordered[num_Eig-1])> conv_req){

    iter++

    Hoperate(H, d_v_Mid, d_v_End, dim);

    d_a.push_back(cublasZdotc(dim, d_v_Mid, sizeof(cuDoubleComplex), d_v_End, sizeof(cuDoubleComplex)));

    vecdiff<<tpb,bpg>>(d_v_End, d_v_End, d_a[iter], d_v_Mid, d_b[iter], d_v_Start);

    d_b.push_back(make_cuDoubleComplex(sqrt(cublasDznrm2(dim, d_v_End, sizeof(cuDoubleComplex))),0.));
    
    alpha = make_cuDoubleComplex(1./d_b[iter+1].x,0.);
    cublasZaxpy(dim, alpha, d_v_End);
    
    cublasZcopy(dim, d_v_Mid, sizeof(cuDoubleComplex), d_v_Start, sizeof(cuDoubleComplex));
    cublasZcopy(dim, d_v_End, sizeof(cuDoubleComplex), d_v_Mid, sizeof(cuDoubleComplex));

    d_diag.push_back(0.);
    d_offdia.push_back(0.);

    complextodoubler<<tpb,bpg>>(d_diag, d_a, iter);
    complextodoubler2<<tpb,bpg>>(d_offida, d_b, iter);

    double* d_H_eigen;
    size_t d_eig_pitch;


    cudaMallocPitch(&d_H_eigen, &d_eig_pitch, iter*sizeof(double), iter);
    identity<<tpb, bpg>>(d_H_eigen, iter);

    returned = tqli(d_diag, d_offdia, iter + 1, d_H_eigen, 0);    

    assign<<tpb,bpg>>(d_ordered, d_diag[0], num_Eig);
    
    for(int i = 1, i < max_Iter, i++){
      for(int j = 0, j< num_Eig, j++){
        if (d_diag[i]< d_ordered[j]){
          d_ordered[j] = d_diag[i];
          break;
        }
      }
    }

    gs_Energy = d_ordered[num_Eig - 1];
  }
}

__device__ void Hoperate(cuDoubleComplex* H, cuDoubleComplex* v0, cuDoubleComplex v1, int dim){
  cuDoubleComplex* alpha;
  cudaMalloc(&alpha, sizeof(cuDoubleComplex));
  cuDoubleComplex* beta;
  cudaMalloc(&beta, sizeof(cuDoubleComplex));
  cuDoubleComplex *alpha = make_cuDoubleComplex(1.,0.);
  cuDoubleComplex *beta = make_cuDoubleComplex(0.,0.); //I heard you like memory management

  v1 = cublasZgemv(N, dim, dim, alpha, &H, m*sizeof(cuDoubleComplex), v0, sizeof(cuDoubleComplex), beta, v1, sizeof(cuDoubleComplex);
  // line above implements v1 = alpha*H*v0 + beta*v1, the other stuff is there for memory purposes
  cudaFree(alpha);
  cudaFree(beta);
}

  
