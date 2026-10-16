      program laplace_acc

      use openacc

      implicit none
       integer :: i,j,k,ii
       integer :: iter,count_rate, count_max,count,nenv_var
       integer :: t_start,t_final
       integer, parameter :: nx=8192,ny=nx
       integer, parameter :: max_iter=525
       double precision, parameter    :: pi=4d0*datan(1d0)
       real, parameter    :: error=0.05
       double precision               :: max_err,time_s,&
                                         d2fx,d2fy,max_err_part
       double precision, allocatable :: f(:,:),f_k(:,:)

       call system_clock(count_max=count_max, count_rate=count_rate)

       call system_clock(t_start)

     allocate(f(0:nx+1,0:ny+1)); allocate(f_k(1:nx,1:ny))

     f=0d0; f_k=0d0
    
!Generate the Initial Conditions (ICs)       
     CALL RANDOM_NUMBER(f)

       iter = 0

       print*,""
       print*, "--Start iterations",iter
       print*,""

!Structed data locality       
!$acc data copyin(f) copyout(f_k)

       do while (max_err.gt.error.and.iter.le.max_iter)

!$acc parallel loop present(f,f_k) collapse(2)
        do j=1,ny
            do i=1,nx
               d2fx = f(i+1,j) + f(i-1,j)
               d2fy = f(i,j+1) + f(i,j-1)
               f_k(i,j) = 0.25*(d2fx + d2fy)
             enddo
          enddo
!$acc end parallel loop

          max_err=0.

!$acc parallel loop present(f,f_k) collapse(2) & 
!$acc reduction(max:max_err)
          do j=1,ny
            do i=1,nx
               max_err = max(dabs(f_k(i,j) - f(i,j)),max_err)
               f(i,j) = f_k(i,j)
            enddo
          enddo
!$acc end parallel loop

!max_err is copied back to the CPU-host by default
          
         if(mod(iter,50).eq.0 )write(*,'(i5,f10.6)')iter,max_err

         iter = iter + 1

        enddo
!$acc end data 

       deallocate(f)

       write(*,'(i5,f10.6)') iter,max_err

       call system_clock(t_final)

       time_s = real(t_final - t_start)/real(count_rate)

       print*, '--Time it takes (s)', time_s

       print*, '--Job is completed successfully--'
       print*,''

!to check the result

        do j=1,ny
           write(111,*)j,sum(f_k(:,j))
        enddo
        print*,"--Sum",sum(f_k(:,:))/nx/2
        print*,"--END :)"      
      

       deallocate(f_k)

       end
