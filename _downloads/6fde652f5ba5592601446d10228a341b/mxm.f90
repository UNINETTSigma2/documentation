program mxm
  integer, parameter :: r8  = selected_real_kind(p=15,r=307)
  parameter(N=4000)
  real(r8) a(N,N), b(N,N) , c(N,N), temp
  integer i, j, l, c1, c2

  call random_number(a)
  call random_number(b)

  call system_clock(count=c1)
  
!$acc kernels                                                                  
  do j = 1,N
     do l = 1,N
       do i = 1,N
         c(i,j) = c(i,j) + a(i,l)*b(l,j)
       enddo
     enddo
  enddo
!$acc end kernels                                                               
  call system_clock(count=c2)

  write(*,*) "Calc time : ",(c2-c1)/1e6," secs"
  write(*,*) c(1,1), c(N,N), sum(c)
end program mxm
