(short-videos)=

# Short Instructions Video Archives

 Here we share the link to our archive of short instruction videos, made available on the [NRIS YouTube channel](https://www.youtube.com/channel/UCG6fTXEY_SQYohtpU6aZwPw):

## Parallel Computing

[In this video playlist](https://www.youtube.com/playlist?list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O), we will cover parallel computing on Linux clusters. Parallel computing involves performing multiple calculations or processes at the same time to break down a large task into smaller, manageable sub-tasks. This approach typically boosts performance and efficiency.

We will start by using Fortran for parallel computing to perform matrix multiplication. Fortran is one of the oldest programming languages still widely used today, especially in high-performance computing. As we continue with these videos, we will also explore parallel computing using Python and C.

In these videos, we emphasize the use of Python scripts. While creating these scripts takes time, they can be reused, enhancing efficiency and serving as useful documentation of your processes. Initially, typing Linux commands directly might seem quicker, but using scripts becomes a time-saver as you advance. Regularly entering the same Linux commands without scripts is inefficient.

In these videos, we will be using the Emacs editor, but please feel free to use any editor that you are comfortable with. We hope these videos will enhance your parallel computing skills. Enjoy your computing journey!

If anything was unclear or you think something should have been explained in more detail, please let us know in the video comments on Youtube. We appreciate your feedback and it will help us improving these videos.

### 1.The Basics on the Saga Linux cluster
[In this video](https://www.youtube.com/watch?v=LSoRhTMPeWk&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=1), we explore the Saga Linux cluster in detail. We demonstrate basic Linux commands and how to determine your location within the cluster. Additionally, we review the various types of compute nodes available on the Saga Linux cluster.

### 2.Interactive Jobs on the Saga compute nodes
[In this video](https://www.youtube.com/watch?v=tEpbr6fKjIs&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=2), we demonstrate how to launch an interactive job on the compute nodes. We walk you through a Python setup script that employs the Slurm-allocate command, explaining different options to move from the login nodes to one of the compute nodes.

### 3.CPU billing on the Saga compute nodes.
[In this video](https://www.youtube.com/watch?v=rtALtvMsPoM&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=3), we explain how to determine the billing factor for various jobs on the compute nodes. We show you how to create a Linux alias for the CPU billing factor and demonstrate its usage. Additionally, we discuss how CPU time varies based on the job's memory and CPU core allocations.

### 4.Fortran = Formula Translation
[In this video](https://www.youtube.com/watch?v=SWwiegpzVXw&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=4), we cover key aspects of the Fortran programming language, highlighting its use in matrix multiplication. We outline a Fortran program tailored for this task and introduce a Python testing script that aids in compiling and evaluating the program. 
The use of a testing script is vital for maintaining consistent performance benchmarks and provides a documented method of how testing was executed.

### 5.Fortran Compiling using the Intel Fortran compiler
[In this video](https://www.youtube.com/watch?v=0r54WVnB-KU&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=5), we explore essential features of the Intel Fortran compiler, emphasizing optimization options for the Intel Sky-lake AVX-512 architecture.
We examine different techniques for calculating matrix multiplication through accumulative summation. We also guide you on how to measure the memory usage of your Fortran programs.

### 6.Fortran Optimization
[In this video](https://www.youtube.com/watch?v=GnVD2eZIvjs&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=6), we investigate the optimization capabilities of the Intel Fortran compiler, focusing on matrix multiplication performance in a parallel computing setting on a single CPU core. 
We'll review the compiler's optimization reports, explore loop vectorization, and discuss various strategies to enhance matrix multiplication efficiency. Furthermore, we'll provide an estimation of the maximum theoretical performance on the Sky-lake architecture.

## Other topics