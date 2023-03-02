/**
* Example program to show how to combine OpenACC and cuBLAS library calls
*/

#include <cublas_v2.h>
#include <math.h>
#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10000

int main() {
  printf("Starting SAXPY + OpenACC program\n");
  // Allocate vectors which we will use for computations
  float* a = (float*) calloc(N, sizeof(float));
  float* b = (float*) calloc(N, sizeof(float));
  float sum = 0.0;
  const float alpha = 2.0;

  if (a == NULL || b == NULL) {
    printf("Could not allocate compute vectors!");
    return EXIT_FAILURE;
  }

  // Initialize input arrays, this is done on CPU host
  printf("  Initializing vectors on CPU\n");
  for (int i = 0; i < N; i++) {
    a[i] = 1.0;
    b[i] = 2.0;
  }

  // Create cuBLAS handle for interacting with cuBLAS routines
  printf("  Creating cuBLAS handle\n");
  cublasHandle_t handle;
  cublasStatus_t status; // Variable to hold return status from cuBLAS routines
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("Could not initialize cuBLAS handle!\n");
    return EXIT_FAILURE;
  }

  // Create OpenACC data region so that our compute vectors are accessible on
  // GPU device for cuBLAS
  printf("  Starting calculation\n");
  #pragma acc data copy(b[0:N]) copyin(a[0:N])
  {
    // To allow cuBLAS to interact with our compute vectors we need to make
    // them available as pointers. NOTE however that these pointers point to
    // areas in the GPU memory so they cannot be dereferenced on the CPU,
    // however, by using the 'host_data' directive we can use the pointers from
    // CPU code passing them to other functions that require pointers to GPU
    // memory
    #pragma acc host_data use_device(a, b)
    {
      status = cublasSaxpy(handle, N, &alpha, a, 1, b, 1);
      if (status != CUBLAS_STATUS_SUCCESS) {
	printf("SAXPY failed!\n");
	// NOTE we cannot exit here since this is within an accelerated region
      }
    }
    // We can now continue to use a and b in OpenACC kernels and parallel loop
    #pragma acc kernels
    for (int i = 0; i < N; i++) {
      sum += b[i];
    }
  }
  // After the above OpenACC region has ended 'a' has not changed, 'b' contains
  // the result of the SAXPY routine and 'sum' contains the sum over 'b'

  // To ensure everything worked we can check that the sum is as we expected
  if (fabs(sum - 4.0 * (float) N) < 0.001) {
    printf("  Calculation produced the correct result of '4 * %d == %.0f'!\n", N, sum);
  } else {
    printf("  Calculation produced _incorrect_ result, expected '4 * %d == %.3f'\n", N, sum);
  }

  // Free cuBLAS handle
  cublasDestroy(handle);
  // Free computation vectors
  free(a);
  free(b);
  // Indicate to caller that everything worked as expected
  printf("Ending SAXPY + OpenACC program\n");
  return EXIT_SUCCESS;
}
