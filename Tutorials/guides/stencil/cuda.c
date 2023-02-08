// Evolve the 'next' field from 'curr' with total grid size of (size + 2)^2,                                          
// 'alpha' is the diffusion constant and 'dt' is the time derivative                                                  
__global__ void evolve(const float* curr, float* next, const int size, const float
                       cell_size, const float alpha, const float dt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
#define CURR(i,j) curr[((i)+1)*(size+2)+(j)+1]
#define NEXT(i,j) next[((i)+1)*(size+2)+(j)+1]
  // Additional variables                                                                                             
  const float cell = cell_size * cell_size;
  const float r = alpha * dt;                                                                                                
  if (0 < i && i < size + 1 && 0 < j && j < size + 1) {
    NEXT(i,j) = CURR(i,j) + r * (
                                 (CURR(i-1,j)+CURR(i+1,j)+
                                  CURR(i,j-1)+CURR(i,j+1)-
                                  4.0*CURR(i,j)) / (cell_size*cell_size)
                                 );
  }
}

