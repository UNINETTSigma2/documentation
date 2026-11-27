      program laplace_mpiomp_aware

      use mpi
      use omp_lib

      implicit none
       integer status(MPI_STATUS_SIZE)
       integer :: i,j,k,ii
       integer :: iter,count_rate, count_max,count,nenv_var
       integer :: myid,ierr,nproc,nxp,nyp,tag,tag1,tag2,nsend
       integer, parameter :: nx=20000,ny=nx
       integer, parameter :: max_iter=525
       double precision, parameter    :: pi=4d0*datan(1d0)
       real, parameter    :: error=0.05
       double precision               :: max_err,time_s,&
                                         d2fx,d2fy,max_err_part
       real               :: t_start,t_final
       double precision, allocatable :: f(:,:),f_k(:,:)
       double precision, allocatable :: f_send(:,:),f_full(:,:)
       character(len=300) :: env_var

       integer :: deviceType,myDevice,numDevice,host_rank,host_comm

       !MPI starts
        ! Initialise OpenMPI communication.
        call MPI_INIT(ierr)
        ! Get number of active processes (from 0 to nproc-1).
        call MPI_COMM_SIZE(MPI_COMM_WORLD, nproc, ierr )
        ! Identify the ID rank (process).
        call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr )

!check if GPU-aware support is enabled
       if(myid.eq.0) then  
         print*, ''      
         call getenv("MPICH_GPU_SUPPORT_ENABLED", env_var)
         read(env_var, '(i10)' ) nenv_var
         if (nenv_var.eq. 1) then
           print *, '--MPICH_GPU_SUPPORT_ENABLED is enabled!'
           print*, ''
         elseif (nenv_var.ne. 1) then
           print *, '--MPICH_GPU_SUPPORT_ENABLED is NOT enabled!'
           print *, ''
         endif
       endif

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
     allocate(f(0:nx+1,0:nyp+1));

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

!Set a device: Determine which processes are on each node
!such that each process will be connected to a GPU

!!Split the world communicator into subgroups of commu, each of which
!contains processes that run on the same node, and which can create a
!shared
!memory region (via the type MPI_COMM_TYPE_SHARED).
!The call returns a new communicator "host_comm", which is created by
!each subgroup.

      call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,&
                               MPI_INFO_NULL, host_comm,ierr)
      call MPI_COMM_RANK(host_comm, host_rank,ierr)
      
      myDevice = host_rank

!returns the device number of the device on which the calling thread is
!executing
     deviceType = omp_get_device_num()
!returns the number of devices available for offloading.
     numDevice = omp_get_num_devices()
!sets the device number to use in device constructs by setting the
!initial value of the default-device-var 

     call omp_set_default_device(myDevice)

       if(myid.eq.0)print*, "--Number of devices per node:", numDevice
       if(myid.eq.0)print*,""

       print*, "--MPI rank", myid, "is connected to GPU", myDevice

       allocate(f_k(1:nx,1:nyp))

       iter = 0

       if(myid.eq.0) then
         print*,""
         print*, "--Start iterations",iter
         print*,""
       endif

!Unstructed data locality 
!$omp target enter data device(myDevice) map(to:f) map(alloc:f_k)

       do while (max_err.gt.error.and.iter.le.max_iter)

!Performing MPI_send and MPI_Recv between GPUs without passing through the host       
!$omp target data use_device_ptr(f)

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

!$omp end target data        

!$omp target teams distribute parallel do collapse(2) schedule(static,1)         
        do j=1,nyp
            do i=1,nx
               d2fx = f(i+1,j) + f(i-1,j)
               d2fy = f(i,j+1) + f(i,j-1)
               f_k(i,j) = 0.25*(d2fx + d2fy)
             enddo
          enddo
!$omp end target teams distribute parallel do

          max_err=0.

!$omp target teams distribute parallel do reduction(max:max_err) &
!$omp collapse(2) schedule(static,1) 
          do j=1,nyp
            do i=1,nx
               max_err = max(dabs(f_k(i,j) - f(i,j)),max_err)
               f(i,j) = f_k(i,j)
            enddo
          enddo
!$omp end target teams distribute parallel do

!max_err is copied back to the CPU-host by default

!$omp target enter data device(myDevice) map(to:max_err)
!Performing MPI_Allreduce between GPUs without passing through the host          
!$omp target data use_device_ptr(max_err)          
         call MPI_ALLREDUCE(MPI_IN_PLACE,max_err,1,&
              MPI_DOUBLE_PRECISION,MPI_MAX, MPI_COMM_WORLD,ierr )
!$omp end target data      
!$omp target exit data map(from:max_err)
      
          if(myid.eq.0) then
            if(mod(iter,50).eq.0 )write(*,'(i5,f10.6)')iter,max_err
          endif

          iter = iter + 1

        enddo
!$omp target exit data map(from:f_k) map(delete:f)

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
