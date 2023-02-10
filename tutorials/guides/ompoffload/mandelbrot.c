/**
 * Mandelbrot implementation for accelerators (e.g. GPUs)
 */

#include "utils/lodepng.h"
#include "utils/palette.h"
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Default width and height for image if not given
static const int WIDTH = 1280;
static const int HEIGHT = 720;
//static const int WIDTH = 12;
//static const int HEIGHT = 7;
// Default output name if not given
static const char* OUTPUT_NAME = "mandelbrot.png";
// Maximum iteration count before exiting mandelbrot function
//static const uint32_t MAX_ITER = 1000;
static const uint32_t MAX_ITER = 1000;
// Helper function to scale 'num' to the range '[min, max]'
#pragma omp declare target
float scale(float num, const float min, const float max) {
  const float scale = max - min;
  return num * scale + min;
}
#pragma omp end declare target

/**
 * Mandelbrot function, calculates the value of the mandelbrot set at pixel 'px/py'
 */
#pragma omp declare target
uint32_t mandelbrot(const int px, const int py, const int width, const int height,
                    const int max_iter) {
  const float x0 = scale((float) px / (float) width, -2.5, 1.);
  const float y0 = scale((float) py / (float) height, -1., 1.);
  float x = 0.;
  float y = 0.;
  float x2 = 0.;
  float y2 = 0.;
  int iters = 0;
 
  while (x2 + y2 < 4. && iters < max_iter) {
    y = 2. * x * y + y0;
    x = x2 - y2 + x0;
    x2 = x * x;
    y2 = y * y;
    iters += 1;
  }
  return (uint32_t) iters;
}
#pragma omp end declare target

int main (int argc, char** argv) {
  int width = WIDTH;
  int height = HEIGHT;
  char output_name[128];
  int max_iter = MAX_ITER;
  strncpy (output_name, OUTPUT_NAME, strnlen (OUTPUT_NAME, 127) + 1);
  // Assume the first argument is the width and height of the image
  if (argc > 1) {
    if (strncmp (argv[1], "-h", 2) == 0 || strncmp (argv[1], "--help", 6) == 0) {
      printf("Usage: %s <width>x<height> <max iterations> <output filename>\n", argv[0]);
      printf("\tImage size can also be one of {8k, 4k, 3k, 1080p, 720p}\n");
      return EXIT_SUCCESS;
    }
    // First we check image size is one of the predefined sizes
    if (strncmp (argv[1], "8k", 2) == 0) {
      width = 7680;
      height = 4320;
    } else if (strncmp (argv[1], "4k", 2) == 0) {
      width = 3840;
      height = 2160;
    } else if (strncmp (argv[1], "3k", 2) == 0) {
      width = 3000;
      height = 2000;
    } else if (strncmp (argv[1], "1080p", 5) == 0) {
      width = 1920;
      height = 1080;
    } else if (strncmp (argv[1], "720p", 4) == 0) {
      width = 1280;
      height = 720;
    } else {
      // Assume user has supplied <width>x<height>
      // Try to find 'x' in argument
      char* token;
      token = strtok (argv[1], "x");
      if (token != NULL) {
        width = atoi (token);
      } else {
        printf("\033[0;31mInvalid width/height definition:\033[0m '%s'\n", argv[1]);
        printf("\tShould be '<width>x<height>'\n");
        return EXIT_FAILURE;
      }
      token = strtok (NULL, "x");
      if (token != NULL) {
        height = atoi (token);
      } else {
        printf("\033[0;31mInvalid width/height definition:\033[0m '%s'\n", argv[1]);
        printf("\tShould be '<width>x<height>'\n");
        return EXIT_FAILURE;
      }
    }
  }
  // Second argument is the maximum iteration count
  if (argc > 2) {
    max_iter = atoi (argv[2]);
  }
  // Third argument is the output filename to write PNG file to
  if (argc > 3) {
    if (strlen (argv[3]) > 127) {
      printf("\033[0;31mOutput filename to large!\033[0m");
      return EXIT_FAILURE;
    }
    strncpy (output_name, argv[3], strnlen (argv[3], 127) + 1);
  }
  // Allocate storage for image
  uint32_t* image = calloc (width * height, sizeof (uint32_t));
  if (image == NULL) {
    printf("\033[0;31mCould not allocate memory for image!\033[0m\n");
    return EXIT_FAILURE;
  }
  printf("Generating \033[0;35m%dx%d\033[0m image with max \033[0;35m%d\033[0m iterations\n",
         width, height,
         max_iter);
  /****************************************************************************/
  /***************************   Main computation   ***************************/
  /****************************************************************************/
  const double start_time = omp_get_wtime ();
  // For each pixel of our image calculate the value of the mandelbrot set  
  
  /*!
    OMP on GPUs
  */

#pragma omp target data map(to:palette[0:palette_size]), map(from:image[0:width*height])  
  { 
    //#pragma omp target map(tofrom:image[0:width*height], palette[0:palette_size])
    //#pragma omp parallel for
    //#pragma omp target teams distribute parallel for collapse(2) 
#pragma omp target teams distribute parallel for collapse(2) schedule(static, 1)
    for (int y = 0; y < height; y++) {
      //#pragma omp parallel for schedule(static, 1)
      for (int x = 0; x < width; x++) {
	const uint32_t iters = mandelbrot (x, y, width, height, max_iter);
	image[y * width + x] = palette[iters % palette_size];
      }
    }

  }

  const double end_time = omp_get_wtime ();
  printf("Used \033[0;35m%.3f\033[0m ms for computation\n",
	 (end_time - start_time) * 1000.0);
  /****************************************************************************/
  
  // copy the data back only when it's needed
  // #pragma omp target map(from:image[0:width*height])
  // {
  // }
 
  // Write image to file
  const unsigned char png_error = lodepng_encode32_file(output_name,
							(const unsigned char*) image,
							width, height);

  // Free image storage
  free (image);
  if (png_error) {
    printf("\033[0;31mAn error occurred while writing to PNG:\033[0m %s\n",
	   lodepng_error_text (png_error));
    return EXIT_FAILURE;
  }
  printf("Wrote Mandelbrot result to \033[0;35m%s\033[0m\n", output_name);

  return EXIT_SUCCESS;
}
