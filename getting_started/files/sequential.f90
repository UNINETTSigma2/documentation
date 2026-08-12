! assumed to be sequential.f90
program sequential
  implicit none
  integer, parameter :: n = 2000000
  integer :: i, j, n_in
  real(8) :: x, y, pi

  call random_seed()
  do j = 1, 100
    n_in = 0
    do i = 1, n
      call random_number(x)
      call random_number(y)
      x = x * 2.0d0 - 1.0d0
      y = y * 2.0d0 - 1.0d0
      if (x * x + y * y < 1.0d0) n_in = n_in + 1
    end do
    pi = 4.0d0 * n_in / n
  end do
end program sequential
