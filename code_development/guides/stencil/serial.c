#include "png_writer.h"
#include "utils/lodepng.h"
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

// Forward declarations
// Initialize the field with a size of (size + 2)^2
void init_field(float* field, const int size);
// Evolve the 'next' field from 'curr' with total grid size of (size + 2)^2,
// 'alpha' is the diffusion constant and 'dt' is the time derivative
void evolve(const float* curr, float* next, const int size, const float cell_size, const float alpha, const float dt);
// Helper method to save the field to PNG
void save(const float* field, const int size, const int iteration);

int main(int argc, char* argv[]) {
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
    grid_size = strtol(argv[1],NULL,10);
  }
  if (argc > 2) {
    num_iter = strtol(argv[2],NULL,10);
  }
  if (argc > 3) {
    save_interval = strtol(argv[3],NULL,10);
  }

  // Initialization
  printf("Solving heat equation for grid \033[0;35m%d x %d\033[0m with \033[0;35m%d\033[0m iterations\n",
	 grid_size, grid_size, num_iter);
  // Setup grid arrays
  float* grid = (float*)calloc((grid_size + 2) * (grid_size + 2), sizeof(float));
  float* next_grid = (float*)calloc((grid_size + 2) * (grid_size + 2), sizeof(float));
  if (grid == NULL || next_grid == NULL) {
    fprintf(stderr, "\033[0;31Could not allocate fields of size %d x %d\033[0m\n", grid_size, grid_size);
    exit(EXIT_FAILURE);
  }
  
  init_field(grid, grid_size);
  init_field(next_grid, grid_size);

  if (save_interval > 0) {
    save(grid, grid_size, 0);
  }

  // Main calculation
  const double start_time = omp_get_wtime();
  for (int i = 1; i <= num_iter; i++) {
    // One iteration of the heat equation
    evolve(grid, next_grid, grid_size, cell_size, alpha, dt);

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
  free(grid);
  free(next_grid);
  return EXIT_SUCCESS;
}

// Initialize the field with a size of (size + 2)^2
//
// This function will fill the field with an initial condition that we want to
// simulate from
void init_field(float* field, const int size) {
  // First create a uniform temperature with a source disk in the middle
  // Radius of source disk
  const float radius = (float) size / 6.0;
  for (int i = 0; i < size + 2; i++) {
    for (int j = 0; j < size + 2; j++) {
      const int index = i * (size + 2) + j;
      // Distance of the current index to center of the field
      const int dx = i - size / 2 + 1;
      const int dy = j - size / 2 + 1;
      if (dx * dx + dy * dy < radius * radius) {
	field[index] = 5.0;
      } else {
	field[index] = 65.0;
      }
    }
  }
  // Setup boundary conditions so that there are sources/sinks around the edges
  for (int i = 0; i < size + 2; i++) {
    field[i] = 85.0; // Top side
    field[i * (size + 2)] = 20.0; // Left side
    field[i * (size + 2) + size + 1] = 70.0; // Right side
    field[(size + 1) * (size + 2) + i] = 5.0; // Bottom side
  }
}

// Evolve the 'next' field from 'curr' with total grid size of (size + 2)^2,
// 'alpha' is the diffusion constant and 'dt' is the time derivative
void evolve(const float* curr, float* next, const int size, const float
	    cell_size, const float alpha, const float dt) {
#define CURR(i,j) curr[((i)+1)*(size+2)+(j)+1]
#define NEXT(i,j) next[((i)+1)*(size+2)+(j)+1]
  const float r = alpha*dt;
  for ( int i=0; i<size; i++ )
    for ( int j=0; j<size; j++ )
      NEXT(i,j) = CURR(i,j) + r * (
				   (CURR(i-1,j)+CURR(i+1,j)+CURR(i,j-1)+CURR(i,j+1)-4.0*CURR(i,j)) / (cell_size*cell_size)
				   ); 
}

// Helper method to save the field to PNG
void save(const float* field, const int size, const int iteration) {
  char filename[256];
  sprintf(filename, "field_%05d.png", iteration);
  const int write_res = write_field(filename, field, size);
  if (write_res != 0) {
    fprintf(stderr, "\033[0;31mCould not write initial image!\033[0m\n\tError: %s\n",
	    lodepng_error_text(write_res));
    exit(EXIT_FAILURE);
  }
}
