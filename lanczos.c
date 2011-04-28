
// h_ means this variable is going to be on the host (CPU)
// d_ means this variable is going to be on the device (GPU)
// s_ means this variable is shared between threads on the GPU

__global__ void vecdiff(cuDoubleComplex* w, cuDoubleComplex* x, cuDoubleComplex alpha, cuDoubleComplex* y, cuDoubleComplex beta, cuDoubleComplex* z int n){

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < n) {
    w[i] = x[i] - alpha*y[i] - beta*z[i];
  }
  syncthreads();
}

__global__ void assignr(cuDoubleComplex* a, double b, int n){
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < n){
    a[i] = make_cuDoubleComplex(b,0.);
  }
  syncthreads();
}

__global__ void complextodoubler(cuDoubleComplex* a, double* b, int n){
  int i = blockDim.x*blockIdx.x + threadIdx.x;

  if(i <= n){
    b[i] = a[i].x;
  }

  syncthreads();
}

__global__ void complextodoubler2(cuDoubleComplex* a, double* b, int n){
  int i = blockDim.x*blockIdx.x + threadIdx.x + 1;

  if(i <= n){
    b[i-1] = a[i].x;
  }
  
  b[n] = 0.;
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

void lanczos(const cuDoubleComplex* h_H, const int dim, const int num_Eig, const int max_Iter){

  size_t h_pitch;
  size_t d_pitch = h_pitch;\

  cuDoubleComplex* d_H;

  cudaMallocPitch(&d_H, &h_pitch, dim*sizeof(cuDoubleComplex), dim);

  cudaMemcpy2D(d_H, d_pitch, &H, h_pitch, dim, dim, cudaMemcpyHosttoDevice );
  // the above memory code could be total lunacy

  //Now that I have the Hamiltonian on the GPU, it's time to start generating eigenvectors

  cuDoubleComplex* d_a; //these are going to store the elements of the tridiagonal matrix
  cuDoubleComplex* d_b;

  cudaMalloc(&d_a, dim*sizeof(cuDoubleComplex));
  cudaMalloc(&d_b, dim*sizeof(cuDoubleComplex));

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
  d_a[0] = cublasZdotc(dim, d_v_Start, sizeof(cuDoubleComplex), d_v_Mid, sizeof(cuDoubleComplex));
  d_b[0] = make_cuDoubleComplex(0.,0.);

  cuDoubleComplex* y;
  cudaMalloc(&y, dim*sizeof(cuDoubleComplex));
  
  assignr<<tpb,bpg>>(y,0., dim);

  cuDoubleComplex* beta;
  cudaMalloc(&beta, sizeof(cuDoubleComplex));
  *beta = make_cuDoubleComplex(0.,0.);
  
  int tpb = 256; //threads per block
  int bpg = (dim + tpb - 1)/tpb; //blocks per grid
  vecdiff(d_v_Mid, d_v_Mid, d_a[0], d_v_Start, y[0], y);
  d_b[1].x = sqrt(cublasDznrm2(dim, d_v_Mid, sizeof(cuDoubleComplex)));
  d_b[1].y = 0.;  // this function (above) takes the norm
  
  cuDoubleComplex* alpha;
  cudaMalloc(&alpha, sizeof(cuDoubleComplex));
  cuDoubleComplex *alpha = make_cuDoubleComplex(1./d_b[1].x,0.);

  cublasZaxpy(dim, alpha, d_v_Mid, sizeof(cuDoubleComplex), y, sizeof(cuDoubleComplex)); // function performs a*x + y

  //Now we're done the first round!
  //*********************************************************************************************************

  vector<double> d_ordered;



  double gs_Energy = 1.;

  int returned;

  int iter = 0;

  while( abs(gs_Energy - d_ordered[num_Eig-1])> conv_req){

    iter++

    Hoperate(H, d_v_Mid, d_v_End, dim);

    d_a[iter] = cublasZdotc(dim, d_v_Mid, sizeof(cuDoubleComplex), d_v_End, sizeof(cuDoubleComplex));

    vecdiff<<tpb,bpg>>(d_v_End, d_v_End, d_a[iter], d_v_Mid, d_b[iter], d_v_Start);

    d_b[iter+1].x = sqrt(cublasDznrm2(dim, d_v_End, sizeof(cuDoubleComplex)));
    d_b[iter+1].y = 0.;
    
    alpha = make_cuDoubleComplex(1./d_b[iter+1].x,0.);
    cublasZaxpy(dim, alpha, d_v_End);
    
    cublasZcopy(dim, d_v_Mid, sizeof(cuDoubleComplex), d_v_Start, sizeof(cuDoubleComplex));
    cublasZcopy(dim, d_v_End, sizeof(cuDoubleComplex), d_v_Mid, sizeof(cuDoubleComplex));

    // In the original code, we started diagonalizing from iter = 5 and above. I start from iter = 1 to minimize issues of control flow
    double* d_diag;
    cudaMalloc(&d_diag, max_Iter*sizeof(double));

    double* d_offdia;
    cudaMalloc(&d_diag, max_Iter*sizeof(double));

    complextodoubler<<tpb,bpg>>(d_diag, a, iter, max_Iter);
    complextodoubler2<<tpb,bpg>>(d_offida, b, iter, max_Iter);

    double* d_H_eigen;
    size_t d_eig_pitch;


    cudaMallocPitch(&d_H_eigen, &d_eig_pitch, max_Iter*sizeof(double), max_Iter);
    identity<<tpb, bpg>>(d_H_eigen, max_Iter);

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

  
