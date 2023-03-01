#include "png_writer.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// Default grid size (GRID_SIZE * GRID_SIZE elements are needed)
static const int DEFAULT_GRID_SIZE = 500;
// Default number of iterations if no command line argument is given
static const int DEFAULT_ITER = 1000;
// Default diffusion constant
static const float DEFAULT_ALPHA = 0.1;
static const float DEFAULT_CELL_SIZE = 0.01;
// Number of blocks to use when launching CUDA kernels
static const int FIXED_BLOCKS = 16;

// Forward declarations
// Initialize the field with a size of (size + 2)^2
__global__ void init_field(float* field, const int size);
// Evolve the 'next' field from 'curr' with total grid size of (size + 2)^2,
// 'alpha' is the diffusion constant and 'dt' is the time derivative

/*__global__ void evolve(const float* curr, float* next, const int size, const float cell_size, const float alpha, const float dt);
 */
__global__ void evolve(const float* curr, float* next, const int size);
// Helper method to save the field to PNG
void save(const float* field, const int size, const int iteration);
// Check the return value of a CUDA function and abort if abnormal behavior
void check_cuda(const cudaError_t err, const char* msg);


// constants array declaration using constant memory

__constant__ float constants[2]; // cell_size, alpha, dt

void setup_constants () {
  float cell_size = DEFAULT_CELL_SIZE;
  float alpha = DEFAULT_ALPHA;
  float dt = pow(cell_size, 4) / (2.0 * alpha * (pow(cell_size, 2) + pow(cell_size, 2)));
  float r = alpha * dt;
  const float host_constants[] = {cell_size, r};
  check_cuda(cudaMemcpyToSymbol(constants, host_constants, 2*sizeof(float)), "Error constant memory" );
}

int main(int argc, char* argv[]) {
  printf("%s", "Cuda with Constant memory\n");
  // grid_size represents the N x N grid to compute over
  int grid_size = DEFAULT_GRID_SIZE;
  // The number of iterations to perform to solve the equation
  int num_iter = DEFAULT_ITER;
  // Diffusion constant
  float alpha = DEFAULT_ALPHA;
  // Size of each grid cell
  float cell_size = DEFAULT_CELL_SIZE;
  // Calculate the time increment to propagate with
  const float dt = pow(cell_size, 4) / (2.0 * alpha * (pow(cell_size, 2) + pow(cell_size, 2)));
  // Save interval
  int save_interval = -1;

  // Command line handling
  if (argc > 1) {
    grid_size = atoi(argv[1]);
  }
  if (argc > 2) {
    num_iter = atoi(argv[2]);
  }
  if (argc > 3) {
    save_interval = atoi(argv[3]);
  }

  // Initialization
  printf("Solving heat equation for grid \033[0;35m%d x %d\033[0m with \033[0;35m%d\033[0m iterations\n",
    grid_size, grid_size, num_iter);
  // Setup CUDA block and grid dimensions to use for kernel launch
  dim3 dim_block;
  dim3 dim_grid;
  if (grid_size + 2 < 32) {
    dim_block = dim3(grid_size + 2, grid_size + 2);
    dim_grid = dim3(1, 1);
  } else {
    dim_block = dim3(FIXED_BLOCKS, FIXED_BLOCKS);
    const int grids = (grid_size + 2 + FIXED_BLOCKS - 1) / FIXED_BLOCKS;
    dim_grid = dim3(grids, grids);
  }
  printf("Launching \033[0;35m(%d, %d)\033[0m grids with \033[0;35m(%d, %d)\033[0m blocks\n",
    dim_grid.x, dim_grid.y, dim_block.x, dim_block.y);
  // Setup grid arrays
  float* grid;
  float* next_grid;
  check_cuda(cudaMallocManaged(&grid, (grid_size + 2) * (grid_size + 2) * sizeof(float)),
    "Could not allocate 'grid'");
  check_cuda(cudaMallocManaged(&next_grid, (grid_size + 2) * (grid_size + 2) * sizeof(float)),
    "Could not allocate 'next_grid'");
  
  init_field<<<dim_grid, dim_block>>>(grid, grid_size);
  check_cuda(cudaGetLastError(), "'init_field' of 'grid' failed");
  init_field<<<dim_grid, dim_block>>>(next_grid, grid_size);
  check_cuda(cudaGetLastError(), "'init_field' of 'next_grid' failed");

  if (save_interval > 0) {
    check_cuda(cudaDeviceSynchronize(), "'init_field' of 'grid' or 'next_grid' failed");
    save(grid, grid_size, 0);
    if (grid_size < 34) {
      for(int i = 0; i < grid_size + 2; i++) {
	for(int j = 0; j < grid_size + 2; j++) {
	  const int index = i * (grid_size + 2) + j;
	  printf(" %2.0f", grid[index]);
	}
	printf("\n");
      }
    }
  }

  // Main calculation
  setup_constants();
  const double start_time = omp_get_wtime();
  for (int i = 1; i <= num_iter; i++) {
    // One iteration of the heat equation
    // evolve<<<dim_grid, dim_block>>>(grid, next_grid, grid_size, cell_size, alpha, dt);
    evolve<<<dim_grid, dim_block>>>(grid, next_grid, grid_size);
    // Wait until the kernel is done running before performing pointer swap
    check_cuda(cudaDeviceSynchronize(), "Waiting for evolve before pointer swap");
    // Exchange old grid with the new updated grid
    float* tmp = grid;
    grid = next_grid;
    next_grid = tmp;

    // Save image if necessary
    if (save_interval > 0 && (i % save_interval) == 0) {
      save(grid, grid_size, i);
    }
  }
  const double total_time = omp_get_wtime() - start_time;
  printf("Used \033[0;35m%.3f\033[0m seconds to evolve field\n", total_time);
  printf("Average time per field update: \033[0;35m%.3f\033[0m ms\n", (total_time * 1e3) / num_iter);

  // Free data and terminate
  cudaFree(grid);
  cudaFree(next_grid);
  check_cuda(cudaDeviceReset(), "Device Reset Failed\n");
  return EXIT_SUCCESS;
}

// Initialize the field with a size of (size + 2)^2
//
// This function will fill the field with an initial condition that we want to
// simulate from
__global__ void init_field(float* field, const int size) {
  // Calculate CUDA index in two dimensions
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;
  // Calculate field index from CUDA indexes
  const int index = row * (size + 2) + col;
  if (index < (size + 2) * (size + 2)) {
    // First create a uniform temperature with a source disk in the middle
    // Radius of source disk
    const float radius = (float) size / 6.0;
    // Distance of the current index to center of the field
    const int dx = row - size / 2 + 1;
    const int dy = col - size / 2 + 1;
    if (dx * dx + dy * dy < radius * radius) {
      field[index] = 5.0;
    } else if (0 < col && col < size + 1 && 0 < row && row < size + 1){
      field[index] = 65.0;
    }

    // The following will be slow and lead to thread divergence, but it isn't
    // that important since this is not a hot loop
    if (row == 0) {
      // Top of the field
      field[index] = 85.0;
    } 
    if (row == size + 1) {
      // Bottom of the field
      field[index] = 5.0;
    }
    if (col == 0) {
      // Left side of the field
      field[index] = 20.0;
    }
    if (col == size + 1) {
      // Right side of the field
      field[index] = 70.0;
    }
  }
}

// Evolve the 'next' field from 'curr' with total grid size of (size + 2)^2,
// 'alpha' is the diffusion constant and 'dt' is the time derivative

/*__global__ void evolve(const float* curr, float* next, const int size, const float
	    cell_size, const float alpha, const float dt) {
*/ 

__global__ void evolve(const float* curr, float* next, const int size) {

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  
#define CURR(i,j) curr[((i)+1)*(size+2)+(j)+1]
#define NEXT(i,j) next[((i)+1)*(size+2)+(j)+1]
  
  // Additional variables
  // const float cell = cell_size * cell_size;
  // const float r = alpha * dt;

  const float cell_size = constants[0];
  const float r = constants[1];

  // When launching this kernel we don't take into account that we don't want
  // it run for the boundary, we solve this by the following if guard, this
  // means that we launch 4 threads more than we actually need, but this is a
  // very low overhead

  if (0 < i && i < size + 1 && 0 < j && j < size + 1) {
    NEXT(i,j) = CURR(i,j) + r * (
                                 (CURR(i-1,j)+CURR(i+1,j)+
                                  CURR(i,j-1)+CURR(i,j+1)-
                                  4.0*CURR(i,j)) / (cell_size*cell_size)
                                 );
  }
  
}

// Helper method to save the field to PNG
void save(const float* field, const int size, const int iteration) {
  char filename[256];
  sprintf(filename, "field_%05d.png", iteration);
  const int write_res = write_field(filename, field, size);
  if (write_res != 0) {
    fprintf(stderr, "\033[0;31mCould not write initial image!\033[0m\n\tError: %d\n",
	    write_res);
    abort();
  }
}

// Check the return value of a CUDA function and abort if abnormal behavior
void check_cuda(const cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "\033[0;31m%s:\033[0m\n", msg);
    fprintf(stderr, "\tError(\033[0;33m%s\033[0m): %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
    abort();
  }
}

