/**
 * OpenACC + MPI implementation of the 1D wave equation
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Default number of points to calculate over, if not given on command line
static const int NUM_POINTS = 400;
// Default number of steps to perform per point, if not given on command line
static const int NUM_STEPS = 4000;
// Default time interval, if not given on command line
static const double DEFAULT_DT = 0.00125;
// Speed of sound used for calculation
static const double SOUND_SPEED = 1.0;

// Define MPI tags for program
static const int lower_tag = 1010; // Send to lower rank
static const int upper_tag = 2020; // Send to higher rank
static const int scatter_tag = 3030; // Gather / Scatter data
static const int gather_tag = 4040; // Gather / Scatter data
// MPI Error codes
static const int ALLOC_WAVE_FAIL = 1001;
static const int ALLOC_WAVES_FAIL = 1002;
static const int INITIAL_DIST_RECV = 1003;
static const int LAST_DIST_RECV = 1004;

// Helper macro to check an MPI call and print error if it failed
#define check_mpi(code, err) \
if (code != MPI_SUCCESS) { \
  printf("\033[0;31m%s\033[0m\n", err); \
  printf("\tError code: \033[0;31m%d\033[0m\n", code); \
  MPI_Abort(MPI_COMM_WORLD, 1337); \
  return EXIT_FAILURE; \
}

/**
 * Helper method to calculate the exact solution at 'x' with time step 't' and
 * speed of sound 'c'
 */
#pragma acc routine seq
double exact (const double x, const double t, const double c) {
  return sin (2. * M_PI * (x - c * t));
}

/**
 * Helper function to calculate the partial derivative du/dt
 */
#pragma acc routine seq
double dudt (const double x, const double t, const double c) {
  return -2. * M_PI * c * cos (2. * M_PI * (x - c * t));
}

int main (int argc, char** argv) {
  // Define variables to use in calculation, initialized to default values
  int points = NUM_POINTS;
  int steps = NUM_STEPS;
  double dt = DEFAULT_DT;

  /************************** Command line handling ***************************/
  if (argc > 1) {
    if (strncmp (argv[1], "-h", 3) == 0 || strncmp (argv[1], "--help", 7) == 0) {
      printf("Usage: \033[0;32m%s\033[0m <number of points> <number of steps> <timer interval>\n", argv[0]);
      return EXIT_SUCCESS;
    }
    points = atoi (argv[1]);
    if (points < 1) {
      printf("\033[0;31mThe number of points must be a positive number larger than '1'!\033[0m\n");
      return EXIT_FAILURE;
    }
  }
  if (argc > 2) {
    steps = atoi (argv[2]);
    if (steps < 0) {
      printf("\033[0;31mThe number of steps must be a positive number!\033[0m\n");
      return EXIT_FAILURE;
    }
  }
  if (argc > 3) {
    dt = atof (argv[3]);
    if (dt <= 0.) {
      printf("\033[0;31mTime interval must be larger than '0.0'!\033[0m\n");
      return EXIT_FAILURE;
    }
  }

  /*************************** MPI work sharing *******************************/
  // Initialize MPI
  check_mpi (MPI_Init(&argc, &argv), "Could not initialize MPI!");
  // Extract MPI size and current rank
  int num_processes = 1;
  int rank = 0;
  check_mpi (MPI_Comm_size(MPI_COMM_WORLD, &num_processes), "Could not fetch COMM_WORLD size");
  check_mpi (MPI_Comm_rank(MPI_COMM_WORLD, &rank), "Could not fetch COMM_WORLD rank");
  if (points % num_processes != 0) {
    if (rank == 0) {
      printf("\033[0;31m%d points can't be split into %d processes!\033[0m\n", points, num_processes);
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }
  const int equal_share = points / num_processes;
  // The first and last rank calculates one additional element, while all other
  // ranks calculates two additional points
  const int local_points = (rank == 0 || rank == num_processes - 1) ? equal_share + 1 : equal_share + 2;
  const int local_start = (rank == 0) ? 0 : equal_share * rank - 1;

  /*************************** Implementation *********************************/
  // Define pointer to global result so that we can compile, this variable is
  // only allocated on the root rank
  double* wave = NULL;
  if (rank == 0) {
    printf("Calculating 1D wave equation with \033[0;35m%d\033[0m points over \033[0;35m%d\033[0m steps with \033[0;35m%f\033[0m time step\n",
           points, steps, dt);
    printf("\t...split over \033[0;35m%d\033[0m processes, processing \033[0;35m%d\033[0m points each\n",
           num_processes, local_points);
    // On the root rank we allocate enough space for the full wave,
    // it is used as the full result
    wave = calloc (points, sizeof (double));
    if (wave == NULL) {
      printf("\033[0;31mCould not allocate %d points for wave results\033[0m\n", points);
      // No need to check output, we will shortly exit anyway
      MPI_Abort(MPI_COMM_WORLD, ALLOC_WAVE_FAIL);
      return EXIT_FAILURE;
    }
  }
  // Allocate memory for local work arrays
  double* wave0 = calloc (local_points, sizeof (double));
  double* wave1 = calloc (local_points, sizeof (double));
  double* wave2 = calloc (local_points, sizeof (double));
  if (wave0 == NULL || wave1 == NULL || wave2 == NULL) {
    printf("\033[0;31mRank %d could not allocate enough space for arrays!\033[0m\n", rank);
    MPI_Abort(MPI_COMM_WORLD, ALLOC_WAVES_FAIL);
    return EXIT_FAILURE;
  }
  const double dx = 1. / ((double) points - 1);
  const double alpha = SOUND_SPEED * dt / dx;
  const double alpha2 = alpha * alpha;
  if (rank == 0) {
    if (fabs (alpha) >= 1.) {
      printf("\033[0;33mComputation will be unstable with the given parameters\033[0m\n");
      printf("\tdt = %f\n", dt);
      printf("\tdx = %f (1. / %d)\n", dx, points);
      printf("\t|alpha| = %f\n", fabs (alpha));
    }
    // Initialize the wave only on the root rank
    #pragma acc parallel loop copyout(wave[:points])
    for (int i = 0; i < points; i++) {
      const double x = (double) i / (double) (points - 1);
      wave[i] = exact (x, 0., SOUND_SPEED);
    }
    // Distribute computation to all other ranks
    for (int r = 1; r < num_processes; r++) {
      const int index = r * equal_share - 1;
      const int num_points = (r < num_processes - 1) ? equal_share + 2 : equal_share + 1;
      check_mpi (MPI_Send(&wave[index], num_points, MPI_DOUBLE, r, scatter_tag, MPI_COMM_WORLD),
                 "Could not distribute data");
    }
    // Distribute data to root rank also
    for (int i = 0; i < local_points; i++) {
      wave0[i] = wave[i];
      wave1[i] = wave0[i];
    }
  } else {
    MPI_Status out;
    check_mpi (MPI_Recv(wave0, local_points, MPI_DOUBLE, 0, scatter_tag, MPI_COMM_WORLD, &out),
               "Could not receive data");
    if (out.MPI_ERROR != MPI_SUCCESS) {
      printf("\033[0;31mMPI Recv error!\033[0m count: %ld, cancelled: %d, error: %d\n",
             out._ucount / sizeof (double), out._cancelled, out.MPI_ERROR);
      MPI_Abort(MPI_COMM_WORLD, INITIAL_DIST_RECV);
      return EXIT_FAILURE;
    }
    for (int i = 0; i < local_points; i++) {
      wave1[i] = wave0[i];
    }
  }
  // Subsequent steps utilize the existing arrays for computation
  #pragma acc data copy(wave1[:local_points]) copyin(wave0[:local_points]) \
    create(wave2[:local_points])
  for (int s = 1; s < steps + 1; s++) {
    const double t = (double) s * dt;
    if (s == 1) {
      // First time step we use the initial derivative information to calculate
      // the solution
      #pragma acc parallel loop
      for (int i = 1; i < local_points - 1; i++) {
        const double x = (double) (i + local_start) / (double) (points - 1);
        wave2[i] = (1. - alpha2) * wave1[i]
                   + 0.5 * alpha2 * (wave1[i - 1] + wave1[i + 1])
                   + dt * dudt (x, t, SOUND_SPEED);
      }
    } else {
      // After first step we use previous calculations for future values
      #pragma acc parallel loop
      for (int i = 1; i < local_points - 1; i++) {
        wave2[i] = 2. * (1. - alpha2) * wave1[i]
                    + alpha2 * (wave1[i - 1] + wave1[i + 1])
                   - wave0[i];
      }
    }
    // Copy data from GPU to CPU to prepare for MPI sharing
    #pragma acc update self(wave2[1:1])
    #pragma acc update self(wave2[local_points - 2:1])
    // Share data with neighboors
    if (rank > 0) {
      MPI_Send(&wave2[1], 1, MPI_DOUBLE, rank - 1, lower_tag, MPI_COMM_WORLD);
      MPI_Status out;
      MPI_Recv(&wave2[0], 1, MPI_DOUBLE, rank - 1, upper_tag, MPI_COMM_WORLD, &out);
    } else {
      wave2[0] = exact (0., t, SOUND_SPEED);
    }
    if (rank < num_processes - 1) {
      MPI_Status out;
      MPI_Recv(&wave2[local_points - 1], 1, MPI_DOUBLE, rank + 1, lower_tag, MPI_COMM_WORLD, &out);
      MPI_Send(&wave2[local_points - 2], 1, MPI_DOUBLE, rank + 1, upper_tag, MPI_COMM_WORLD);
    } else {
      wave2[local_points - 1] = exact (1., t, SOUND_SPEED);
    }
    // Copy data we got from MPI neighbors back to GPU
    #pragma acc update device(wave2[0:1])
    #pragma acc update device(wave2[local_points - 1:1])
    // Shift data
    #pragma acc parallel loop
    for (int i = 0; i < local_points; i++) {
      wave0[i] = wave1[i];
      wave1[i] = wave2[i];
    }
  }
  // Synchronize data back to root rank
  if (rank == 0) {
    printf("Synchronizing results\033[0;33m...\033[0m ");
    // Copy root rank data back into result array
    for (int i = 0; i < local_points; i++) {
      wave[i] = wave1[i];
    }
    // Receive data from all other ranks
    for (int r = 1; r < num_processes; r++) {
      const int index = r * equal_share - 1;
      const int num_points = (r < num_processes - 1) ? equal_share + 2 : equal_share + 1;
      MPI_Status out;
      check_mpi (MPI_Recv(&wave[index], num_points, MPI_DOUBLE, r, gather_tag, MPI_COMM_WORLD, &out),
                 "Could not receive data when gathering result");
      if (out.MPI_ERROR != MPI_SUCCESS) {
        printf("\033[0;31mMPI Recv error!\033[0m count: %ld, cancelled: %d, error: %d\n",
               out._ucount / sizeof (double), out._cancelled, out.MPI_ERROR);
        MPI_Abort(MPI_COMM_WORLD, LAST_DIST_RECV);
        return EXIT_FAILURE;
      }
    }
    printf("\033[0;32mcompleted\033[0m!\n");
    printf("Calculation ended \033[0;32msuccesfully\033[0m!\n");
  } else {
    check_mpi (MPI_Send(wave1, local_points, MPI_DOUBLE, 0, gather_tag, MPI_COMM_WORLD),
               "Could not send data back to root when gathering results");
  }
  // Free data before exit
  free(wave0);
  free(wave1);
  free(wave2);
  if (rank == 0) {
    free(wave);
  }
  MPI_Finalize();
  return EXIT_SUCCESS;
}
