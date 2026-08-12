# assumed to be sequential.R
library(foreach)


# this function approximates pi by throwing random points into a square
# it is used here to demonstrate a function that takes a bit of time
approximate_pi <- function() {
  # number of points to use
  n <- 2000000

  # generate n random points in the square
  x <- runif(n, -1.0, 1.0)
  y <- runif(n, -1.0, 1.0)

  # count the number of points that are inside the circle
  n_in <- sum(x^2 + y^2 < 1.0)

  4 * n_in / n
}


foreach (i=1:100, .combine=c) %do% {
  approximate_pi()
}
