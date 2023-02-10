/**
 * Serial implementation of the Jacobi iteration
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Number of rows and columns in our matrix
static const int NUM_ELEMENTS = 2000;
// Maximum number of iterations before quiting
static const int MAX_ITER = 10000;
// Error tolerance for iteration
static const float MAX_ERROR = 0.01;
// Seed for random number generator
static const int SEED = 12345;

int main (int argc, char** argv) {
  // Initialize random number generator
  srand (SEED);
  // Create array to calculate on
  float array[NUM_ELEMENTS][NUM_ELEMENTS];
  // Fill array with data
  for (int i = 0; i < NUM_ELEMENTS; i++) {
    for (int j = 0; j < NUM_ELEMENTS; j++) {
      // The following will create random values between [0, 1]
      array[i][j] = (float) rand () / (float) RAND_MAX;
    }
  }
  // Before starting calculation we will define a few helper variables
  float arr_new[NUM_ELEMENTS][NUM_ELEMENTS];
  float error = __FLT_MAX__;
  int iterations = 0;
  // Perform Jacobi iterations until we either have low enough error or too
  // many iterations
  while (error > MAX_ERROR && iterations < MAX_ITER) {
    error = 0.;
    // For each element take the average of the surrounding elements
    for (int i = 1; i < NUM_ELEMENTS - 1; i++) {
      for (int j = 1; j < NUM_ELEMENTS - 1; j++) {
        arr_new[i][j] = 0.25 * (array[i][j + 1] +
                                array[i][j - 1] +
                                array[i - 1][j] +
                                array[i + 1][j]);
        error = fmaxf (error, fabsf (arr_new[i][j] - array[i][j]));
      }
    }
    // Transfer new array to old
    for (int i = 1; i < NUM_ELEMENTS - 1; i++) {
      for (int j = 1; j < NUM_ELEMENTS - 1; j++) {
        array[i][j] = arr_new[i][j];
      }
    }
    iterations += 1;
  }
  return EXIT_SUCCESS;
}
