/**
 * SYCL accelerated implementation of the Jacobi iteration
 */

#include <iostream>
#include <cstring>

#include <SYCL/sycl.hpp>

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
  // Create default SYCL queue and print name of device
  auto Q = sycl::queue{sycl::default_selector{}};
  std::cout << "Chosen device: "
            << Q.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  // Initialize random number generator
  srand (SEED);

  // Create *SHARED* array to store the input/output
  float *arr_s = sycl::malloc_shared<float>(TOT_ELEMENTS, Q);

  // Fill *SHARED* array with data
  for (int i = 0; i < TOT_ELEMENTS; i++) {
    // The following will create random values between [0, 1]
    arr_s[i] = (float) rand () / (float) RAND_MAX;
  }

  // Create *SHARED* array to calculate on
  float *tmp_s = sycl::malloc_shared<float>(TOT_ELEMENTS, Q);
  float err = __FLT_MAX__;

  // We copy here to get the boundary elements, which will be copied back and forth unchanged
  std::memcpy(tmp_s, arr_s, TOT_ELEMENTS*sizeof(float));

  int iterations = 0;
  // Perform Jacobi iterations until we either have low enough error or too many iterations
  while (err > MAX_ERROR && iterations < MAX_ITER) {
    err = 0.;
    // Submit work item to the SYCL queue
    Q.submit(
      [&](sycl::handler &h) {
        // Define work kernel as single loop
        h.parallel_for(
          sycl::range{(NUM_ELEMENTS - 2) * (NUM_ELEMENTS - 2)},
          [=](sycl::id<1> idx) {
            // Retain array indices from single loop variable
            int i = (idx[0] / NUM_ELEMENTS) + 1;
            int j = (idx[0] % NUM_ELEMENTS) + 1;
            // For each element take the average of the surrounding elements
            tmp_s[i * NUM_ELEMENTS + j] = 0.25 * (arr_s[i * NUM_ELEMENTS + j+1] +
                                                  arr_s[i * NUM_ELEMENTS + j-1] +
                                                  arr_s[(i-1) * NUM_ELEMENTS + j] +
                                                  arr_s[(i+1) * NUM_ELEMENTS + j]);
          }
        );
      }
    ).wait(); // Wait for completion before moving on
    
    // Find maximum error (cannot be done in the loop kernel above)
    for (int i = 0; i < TOT_ELEMENTS; i++) {
      err = std::max(err, std::abs(tmp_s[i] - arr_s[i]));
    }

    // Transfer new array to old (including boundary, which was untouched in the loop)
    std::memcpy(arr_s, tmp_s, TOT_ELEMENTS*sizeof(float));

    iterations++;
  }

  std::cout << "Iterations : " << iterations << " | Error : " << err << std::endl;

  // Free *SHARED* memory
  sycl::free(arr_s, Q);
  sycl::free(tmp_s, Q);

  return EXIT_SUCCESS;
}
