      program laplace_mpi

      use mpi

      implicit none
       integer status(MPI_STATUS_SIZE)
       integer :: i,j,k,ii
       integer :: iter,count_rate, count_max,count,nenv_var
       integer :: myid,ierr,nproc,nxp,nyp,tag,tag1,tag2,nsend
       integer, parameter :: nx=8192,ny=nx
       integer, parameter :: max_iter=525
       double precision, parameter    :: pi=4d0*datan(1d0)
       real, parameter    :: error=0.05
       double precision               :: max_err,time_s,&
                                         d2fx,d2fy,max_err_part
       real               :: t_start,t_final
       double precision, allocatable :: f(:,:),f_k(:,:)
       double precision, allocatable :: f_send(:,:),f_full(:,:)

       !MPI starts
        ! Initialise OpenMPI communication.
        call MPI_INIT(ierr)
        ! Get number of active processes (from 0 to nproc-1).
        call MPI_COMM_SIZE(MPI_COMM_WORLD, nproc, ierr )
        ! Identify the ID rank (process).
        call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr )

        t_start = MPI_WTIME()

        if (mod(nx,nproc).ne.0) then
           if (myid.eq.0) write(*,*) 'nproc has to divide nx'
           stop
        else
           nxp = nx/nproc
        endif
        if (mod(ny,nproc).ne.0) then
           if (myid.eq.0) write(*,*) 'nproc has to divide ny'
           stop
        else
           nyp = ny/nproc
        endif

        if(myid.eq.0) then
          print*,'--nbr of proc', nproc
          write(*,*)'--nbr of points nx,ny',nx,ny
          write(*,*)'--nbr of elmts on each proc, nyp=ny/nproc', nyp
        endif

!Generate the Initial Conditions (ICs)
!Distribute the ICs over all processes using the operation MPI_Scatter
     allocate(f(0:nx+1,0:nyp+1))

     f=0d0; tag1=2020; tag2=2021

     if(myid.eq.0) then
       allocate(f_send(1:nx,1:ny))
        CALL RANDOM_NUMBER(f_send)
      endif

      call MPI_Scatter(f_send,nx*nyp,MPI_DOUBLE_PRECISION,&
                      f(1:nx,1:nyp), nx*nyp,MPI_DOUBLE_PRECISION,&
                      0,MPI_COMM_WORLD, ierr)

      call MPI_Barrier(MPI_COMM_WORLD, ierr)

      if(myid.eq.0) deallocate(f_send)

       allocate(f_k(1:nx,1:nyp))

       iter = 0

       if(myid.eq.0) then
         print*,""
         print*, "--Start iterations",iter
         print*,""
       endif

       do while (max_err.gt.error.and.iter.le.max_iter)

!transfer the data at the boundaries to the neighbouring MPI-process
!send f(:,nyp) from myid-1 to be stored in f(:,0) in myid+1
         if(myid.lt.nproc-1) then
          call MPI_Send(f(:,nyp),(nx+2)*1,MPI_DOUBLE_PRECISION,myid+1,tag1,&
                       MPI_COMM_WORLD, ierr)
         endif

!receive f(:,0) from myid-1
         if(myid.gt.0) then
          call MPI_Recv(f(:,0),(nx+2)*1,MPI_DOUBLE_PRECISION,myid-1, &
                      tag1,MPI_COMM_WORLD, status,ierr)
         endif

!send f(:,1) from myid+1 to be stored in f(:,nyp+1) in myid-1
         if(myid.gt.0) then
          call MPI_Send(f(:,1),(nx+2)*1,MPI_DOUBLE_PRECISION,myid-1,tag2,&
                       MPI_COMM_WORLD, ierr)
         endif

!receive f(:,npy+1) from myid-1
        if(myid.lt.nproc-1) then
         call MPI_Recv(f(:,nyp+1),(nx+2)*1,MPI_DOUBLE_PRECISION,myid+1,&
                      tag2,MPI_COMM_WORLD, status,ierr)
        endif

        do j=1,nyp
            do i=1,nx
               d2fx = f(i+1,j) + f(i-1,j)
               d2fy = f(i,j+1) + f(i,j-1)
               f_k(i,j) = 0.25*(d2fx + d2fy)
             enddo
          enddo

          max_err=0.

          do j=1,nyp
            do i=1,nx
               max_err = max(dabs(f_k(i,j) - f(i,j)),max_err)
               f(i,j) = f_k(i,j)
            enddo
          enddo

         call MPI_ALLREDUCE(MPI_IN_PLACE,max_err,1,&
              MPI_DOUBLE_PRECISION,MPI_MAX, MPI_COMM_WORLD,ierr )

          if(myid.eq.0) then
            if(mod(iter,50).eq.0 )write(*,'(i5,f10.6)')iter,max_err
          endif

          iter = iter + 1

        enddo

       deallocate(f)

       if(myid.eq.0) write(*,'(i5,f10.6)') iter,max_err

        call MPI_Barrier(MPI_COMM_WORLD, ierr)

        t_final = MPI_WTIME()
        time_s = t_final - t_start

       if(myid.eq.0)print*, '--Time it takes (s)', time_s

       if(myid.eq.0) then
         print*, '--Job is completed successfully--'
         print*,''
       endif

!to check the result
       allocate(f_full(nx,ny))
       call MPI_Gather(f_k, nx*nyp, MPI_DOUBLE_PRECISION, &
                      f_full, nx*nyp, MPI_DOUBLE_PRECISION, 0, &
                      MPI_COMM_WORLD, ierr)

       if(myid.eq.0) then
        do j=1,ny
           write(111,*)j,sum(f_full(:,j))
        enddo
        print*,"--Sum",sum(f_full(:,:))/nx/2
        print*,"--END :)"      
       endif

       deallocate(f_full,f_k)

       call MPI_FINALIZE( ierr )

       end
