
// h_ means this variable is going to be on the host (CPU)
// d_ means this variable is going to be on the device (GPU)
// s_ means this variable is shared between threads on the GPU

__global__ void vecdiff(cuDoubleComplex* w, cuDoubleComplex* x, cuDoubleComplex alpha, cuDoubleComplex* y, cuDoubleComplex beta, cuDoubleComplex* z int n){

  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if (i < n) {
    w[i] = x[i] - alpha*y[i] - beta*z[i];
  }
}

__global__ void set(cuDoubleComplex* a, cuDoubleComplex* b, int n){

  int i = blockDim.x*blockIdx.x + threadIdx.x;

  if(i < n){
    a[i] = b[i];
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
  
  for( int i = 0; i< dim; i++){
    d_v_Start[i] = 1.;
  }

  Hoperate(H, d_v_Start, d_v_Mid, dim);
  //*********************************************************************************************************
  // This is just the first steps so I can do the rest  
  d_a[0] = cublasZdotc(dim, d_v_Start, sizeof(cuDoubleComplex), d_v_Mid, sizeof(cuDoubleComplex));
  d_b[0] = make_cuDoubleComplex(0.,0.);

  cuDoubleComplex* y;
  cudaMalloc(&y, dim*sizeof(cuDoubleComplex));
  for(int j = 0; j<dim; j++){
    y[j] = make_cuDoubleComplex(0.,0.);
  }

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

  int exit = 0; // an exit flag

  double gs_Energy = 1.;

  int iter = 0;

  while( exit != 1){

    iter++

    Hoperate(H, d_v_Mid, d_v_End, dim);

    d_a[iter] = cublasZdotc(dim, d_v_Mid, sizeof(cuDoubleComplex), d_v_End, sizeof(cuDoubleComplex));

    vecdiff(d_v_End, d_v_End, d_a[iter], d_v_Mid, d_b[iter], d_v_Start);

    d_b[iter+1].x = sqrt(cublasDznrm2(dim, d_v_End, sizeof(cuDoubleComplex)));
    d_b[iter+1].y = 0.;
    
    alpha = make_cuDoubleComplex(1./d_b[iter+1].x,0.);
    cublasZaxpy(dim, alpha, d_v_End);
    
    set(d_v_Start, d_v_Mid, dim);
    set(d_v_Mid, d_v_End, dim);
      
}

__device__ Hoperate(const cuDoubleComplex* H, const cuDoubleComplex*  v0, const cuDoubleComplex v1, const int dim){
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

  
