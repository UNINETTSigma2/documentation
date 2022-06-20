module parameter_kind
        implicit none
        public
        integer, parameter :: sp = selected_real_kind(6, 37)   !Single precision
        integer, parameter :: dp = selected_real_kind(15, 307) !Double precision
        integer, parameter :: fp = dp
        real(fp), parameter :: pi = 4.0_fp*atan(1.0_fp),dt=0.25_fp
      end module parameter_kind

      program cufft_acc

       use parameter_kind
       use cufft
       use openacc

       implicit none

       integer, parameter   :: nt=512
       integer              :: i,ierr,plan
       complex(fp), allocatable  :: in(:),out(:)
       real(fp), allocatable     :: t(:),w(:)

       allocate(t(nt),w(nt)); allocate(in(nt),out(nt))

       call grid_1d(nt,t,w)

!Example of a sinus function
       do i=1,nt
          in(i) = cmplx(sin(2.0_fp*t(i)),0.0_fp)
       enddo

       print*,"--sum before FFT", sum(real(in(1:nt/2)))
!cufftExecZ2Z executes a double precision complex-to-complex transform plan
       ierr = cufftPlan1D(plan,nt,CUFFT_Z2Z,1)
!acc_get_cuda_stream: tells the openACC runtime to dientify the CUDA
!stream used by CUDA
       ierr = ierr + cufftSetStream(plan,acc_get_cuda_stream(acc_async_sync))

!$acc data copy(in) copyout(out)
!$acc host_data use_device(in,out)
        ierr = ierr + cufftExecZ2Z(plan, in, out, CUFFT_FORWARD)
        ierr = ierr + cufftExecZ2Z(plan, out, in, CUFFT_INVERSE)
!$acc end host_data 

!$acc kernels
       out(:) = out(:)/nt
       in(:) = in(:)/nt
!$acc end kernels
!$acc end data

       ierr =  ierr + cufftDestroy(plan)
       
       print*,""
       if(ierr.eq.0) then
         print*,"--Yep it works :)"
       else
         print*,"Nop it fails, I stop :("
       endif
       print*,""
       print*,"--sum iFFT", sum(real(in(1:nt/2)))

!printing the fft of sinus
        do i=1,nt/2
           write(204,*)w(i),sqrt(cabs(out(i))**2)
        enddo
        deallocate(in); deallocate(out)
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
