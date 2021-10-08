/**
 * Serial implementation of the Jacobi iteration
 */

#include <iostream>
#include <cstring>

// Number of rows and columns in our matrix
static const int NUM_ELEMENTS = 2000;
// Total number of elements in our matrix
static const int TOT_ELEMENTS = NUM_ELEMENTS * NUM_ELEMENTS;
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
  float arr[TOT_ELEMENTS];

  // Fill array with data
  for (int i = 0; i < TOT_ELEMENTS; i++) {
    // The following will create random values between [0, 1]
    arr[i] = (float) rand () / (float) RAND_MAX;
  }

  // Before starting calculation we will define a few helper variables
  float tmp[TOT_ELEMENTS];
  float err = __FLT_MAX__;

  // We copy here to get the boundary elements, which will be copied back and forth unchanged
  std::memcpy(tmp, arr, TOT_ELEMENTS*sizeof(float));

  int iterations = 0;
  // Perform Jacobi iterations until we either have low enough error or too many iterations
  while (err > MAX_ERROR && iterations < MAX_ITER) {
    err = 0.;
    // For each element take the average of the surrounding elements
    for (int i = 1; i < NUM_ELEMENTS - 1; i++) {
      for (int j = 1; j < NUM_ELEMENTS - 1; j++) {
        tmp[i * NUM_ELEMENTS + j] = 0.25 * (arr[i * NUM_ELEMENTS + j+1] +
                                            arr[i * NUM_ELEMENTS + j-1] +
                                            arr[(i-1) * NUM_ELEMENTS + j] +
                                            arr[(i+1) * NUM_ELEMENTS + j]);
        err = std::max(err, std::abs(tmp[i*NUM_ELEMENTS + j] - arr[i*NUM_ELEMENTS + j]));
      }
    }

    // Transfer new array to old (including boundary, which was untouched in the loop)
    std::memcpy(arr, tmp, TOT_ELEMENTS*sizeof(float));
    
    iterations++;
  }

  std::cout << "Iterations : " << iterations << " | Error : " << err << std::endl;

  return EXIT_SUCCESS;
}
