module parameter_kind
        implicit none
        public
        integer, parameter :: FFTW_FORWARD=-1,FFTW_BACKWARD=+1
        integer, parameter :: FFTW_MEASURE=0
        integer, parameter :: sp = selected_real_kind(6, 37) !Single precision
        integer, parameter :: dp = selected_real_kind(15, 307) !Double precision
        integer, parameter :: fp = dp
        real(fp), parameter :: pi = 4.0_fp*atan(1.0_fp),dt=0.25_fp
      end module parameter_kind

      program fftw_serial

       use parameter_kind

       implicit none

       !include "fftw3.f"

       integer, parameter   :: nt=512
       integer              :: i,ierr
       integer*8            :: plan_forward,plan_backward
       complex(fp), allocatable  :: in(:),out(:),f(:)
       real(fp), allocatable     :: t(:),w(:)

       allocate(t(nt),w(nt)); allocate(f(nt))

       call grid_1d(nt,t,w)

!Example of sine function
       do i=1,nt
          f(i) = cmplx(sin(2.0_fp*t(i)),0.0_fp)
       enddo

       print*,"--sum before FFT", sum(real(f(1:nt/2)))

!Creating 1D plans
       allocate(in(nt),out(nt))
       call dfftw_plan_dft_1d(plan_forward,nt,in,out,FFTW_FORWARD,FFTW_MEASURE)
       call dfftw_plan_dft_1d(plan_backward,nt,in,out,FFTW_BACKWARD,FFTW_MEASURE)

!Forward FFT
       in(:) = f(:)
       call dfftw_execute_dft(plan_forward, in, out)
       f(:) = out(:)

!Backward FFT
       call dfftw_execute_dft(plan_backward, out, in)
!The data on the backforward are unnormalized, so they should be divided by N.        
       in(:) = in(:)/real(nt)

!Destroying plans
       call dfftw_destroy_plan(plan_forward)
       call dfftw_destroy_plan(plan_backward)

       print*,"--sum iFFT", sum(real(in(1:nt/2)))

!Printing the FFT of sin(2t)
        do i=1,nt/2
           write(204,*)w(i),dsqrt(cdabs(f(i))**2)
        enddo
        deallocate(in); deallocate(out); deallocate(f)
      end

       subroutine grid_1d(nt,t,w)
        use parameter_kind

        implicit none
        integer             :: i,nt
        real(fp)            :: t(nt),w(nt)

!Defining a uniform temporal grid
       do i=1,nt
          t(i) = (-dble(nt-1)/2.0_fp + (i-1))*dt
       enddo

!Defining a uniform frequency grid
       do i=0,nt/2-1
          w(i+1) = 2.0_fp*pi*dble(i)/(nt*dt)
       enddo
       do i=nt/2,nt-1
          w(i+1) = 2.0_fp*pi*dble(i-nt)/(nt*dt)
       enddo
       end subroutine grid_1d
