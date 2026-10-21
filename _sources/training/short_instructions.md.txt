(short-videos)=

# Short Instructions Video Archives

 Here we share the link to our archive of short instruction videos, made available on the [NRIS YouTube channel](https://www.youtube.com/channel/UCG6fTXEY_SQYohtpU6aZwPw):

## Parallel Computing

[In this video playlist](https://www.youtube.com/playlist?list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O), we will cover parallel computing on Linux clusters. Parallel computing involves performing multiple calculations or processes at the same time to break down a large task into smaller, manageable sub-tasks. This approach typically boosts performance and efficiency.

We will start by using Fortran for parallel computing to perform matrix multiplication. Fortran is one of the oldest programming languages still widely used today, especially in high-performance computing. As we continue with these videos, we will also explore parallel computing using Python and C.

In these videos, we emphasize the use of Python scripts. While creating these scripts takes time, they can be reused, enhancing efficiency and serving as useful documentation of your processes. Initially, typing Linux commands directly might seem quicker, but using scripts becomes a time-saver as you advance. Regularly entering the same Linux commands without scripts is inefficient.

In these videos, we will be using the Emacs editor, but please feel free to use any editor that you are comfortable with. We hope these videos will enhance your parallel computing skills. Enjoy your computing journey!

If anything was unclear or you think something should have been explained in more detail, please let us know in the video comments on Youtube. We appreciate your feedback and it will help us improving these videos.

To use the scripts and files from the videos as templates for your work, copy them from this folder on the Saga Linux Cluster: /cluster/work/support/ParallelComputingYoutube

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

### 7.GPU Offloading on the Saga Linux cluster
[In this video](https://www.youtube.com/watch?v=_EgU49Mbm90&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=7), we explore GPU offloading for a Fortran program designed for matrix multiplication. We employ a Fortran program that utilizes the do concurrent method for GPU offloading. 
Additionally, we introduce a Python testing script specifically for GPU offloading scenarios. We'll guide you through the necessary compilation options for offloading and show you how to monitor GPU usage.

### 8.GPU Offloading testing on the Saga Linux cluster
[In this video](https://www.youtube.com/watch?v=vuHVjlZu5Hg&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=8), we examine GPU offloading techniques using a Fortran program tailored for matrix multiplication. We demonstrate the use of the do concurrent method in a Fortran program to enable GPU offloading. 
Alongside this, we employ a Python testing script designed for GPU offloading scenarios. We assess the performance differences caused by various loop orderings in the matrix multiplication loop. Our testing includes performance evaluations on both the Nvidia P-100 and A-100 GPUs.
We also summarize the performance testing of the Fortran program for matrix multiplication that we've conducted in our recent videos, focusing on both a single CPU core and a single GPU.

### 9.OpenACC offloading on the P-100 Nvidia GPUs
[In this video](https://www.youtube.com/watch?v=FLQM-9vUzPA&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=9), we evaluate the performance of a single GPU on the P-100 nodes, specifically focusing on its efficiency in matrix multiplication, measured in Giga-Flops per second. 
We utilize OpenACC directives to facilitate this process. We'll employ an OpenACC environmental variable to monitor GPU performance and use the optimization report feature of the Nvidia Fortran compiler to examine the compilation optimizations. Additionally, we explore the effects of different loop orderings in the matrix multiplication loop.

### 10.OpenACC offloading on the A-100 Nvidia GPUs
[In this video](https://www.youtube.com/watch?v=mdlWUqp1SwU&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=10), we assess the performance of a single GPU on the A-100 nodes, with a particular focus on matrix multiplication efficiency. We use OpenACC directives to streamline this process and conduct the testing as a batch job. 
We'll utilize an OpenACC environmental variable to track GPU performance and explore the optimization report feature of the Nvidia Fortran compiler. Additionally, we investigate the impact of varying the size of loop gang vectors in the matrix multiplication loop.

### 11.OpenMP (Open Multi-Processing) and the Fram Linux cluster
[In this video](https://www.youtube.com/watch?v=5xlvfsGV6-M&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=11), we explore parallel computing on CPU cores, specifically focusing on OpenMP. 
We begin by assessing the performance of a Fortran program designed for matrix multiplication on a single CPU core on the Fram Linux cluster. 
We also provide an overview of a Fortran program and a Python testing script that we use for parallel computing with OpenMP on the Fram Linux cluster.

### 12.OpenMP on the Fram Linux cluster
[In this video](https://www.youtube.com/watch?v=vp33zlzBnWA&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=12), we focus on parallel computing using OpenMP (Open Multi-Processing) on the Fram Linux cluster. We analyze the performance by experimenting with different numbers of threads. 
To clearly demonstrate the impact of thread count on performance, we will present graphs of both wall time and CPU time, highlighting their dependency on the number of threads used.

### 13.OpenMP on the Betzy Linux cluster
[In this video](https://www.youtube.com/watch?v=tlUw7vQm0X0&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=13), we explore parallel computing with OpenMP (Open Multi-Processing) on the Betzy Linux cluster. We experiment with varying numbers of threads to analyze performance differences on the Betzy Linux cluster. 
To visually demonstrate how thread count affects performance, we will show graphs of both wall time and CPU time, emphasizing their correlation with the number of threads.

### 14.OpenMP affinity on the Fram Linux cluster
[In this video](https://www.youtube.com/watch?v=GFFCYmbwiH4&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=14), we investigate the role of thread affinity in enhancing OpenMP performance on the Fram Linux cluster. 
We experiment with different numbers of threads and affinity configurations to analyze their effects on performance. 
To clearly demonstrate these effects, we will show graphs of both wall time and CPU time, emphasizing how performance varies with changes in thread count.

### 15.OpenMP affinity on the Betzy Linux cluster
[In this video](https://www.youtube.com/watch?v=fGFkVRC1Yx4&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=15), we explore the impact of thread affinity on OpenMP performance on the Betzy Linux cluster. We conduct experiments using various numbers of threads and different affinity settings to assess their influence on performance. 
To clearly illustrate these impacts, we will present graphs of both wall time and CPU time, highlighting the variations in performance as the thread count changes.

### 16.OpenMP Summary
[In this video](https://www.youtube.com/watch?v=MZ1LPnSRuKk&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=16), we summarize the performance outcomes of using OpenMP (Open Multi-Processing) for parallel computing on the Betzy and Fram Linux clusters. 
We'll compare wall and CPU times for various methods showcased in our previous videos. Additionally, we'll discuss why using a large number of threads can lead to bottlenecks in parallel computing with multi-threading, and how transitioning from multi-threading to multi-tasking can address this issue.

### 17.Typing versus Scripting.
[In this video](https://www.youtube.com/watch?v=OWj0Vbvsbc0&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=17), we'll show you how to speed up typing Linux commands with the autocomplete feature. We'll also discuss the advantages of using scripts over manual typing. 
While scripts take time to create, they can be reused multiple times. Scripts are also serving as a detailed record of your procedures. This is especially useful in performance testing, where consistent methods are essential for accurate comparison of different parallel computing strategies. 
Beyond performance testing, Python is excellent for automating various routine tasks. Initially, manual typing may appear quicker, but as you'll experience, scripting ultimately saves time and reduces repetitive work.

### 18.Open-MP offloading on the A-100 Nvidia GPUs
[In this video](https://www.youtube.com/watch?v=omrx1IJesVc&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=18), we evaluate the performance of a single GPU on the A-100 nodes by utilizing Open-MP directives. We perform our tests as batch jobs from the work folder on the Saga Linux cluster. 
We present a Fortran program designed for GPU offloading with Open-MP. Additionally, we examine how different loop orders in the matrix multiplication process affect performance.

### 19.Multi-GPU offloading on the A-100 Nvidia GPUs
[In this video](https://www.youtube.com/watch?v=tGm-CtR_TiY&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=19), we assess the performance of multi-GPU parallel computing on A-100 nodes using Open-MP directives. We conduct our tests as batch jobs from the work folder on the Saga Linux cluster. We also introduce a Fortran program tailored for multi-GPU offloading with Open-MP.

### 20.Multi-GPU parallel computing on the A-100 Nvidia GPUs - Error testing
[In this video](https://www.youtube.com/watch?v=waQKXpt6WCw&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=20), we offer a detailed demonstration of using a Fortran program to verify the accuracy of our multi-GPU parallel computing approach for matrix multiplication. We'll walk you through a process to identify any errors in this method. 
Additionally, we show how to use allocatable arrays in Fortran, which are dynamically allocated at runtime instead of being statically allocated at compile time.

### 21.More multi-GPU parallel computing on the A-100 Nvidia GPUs
In our previous video, we identified an issue with our multi-GPU parallel computing method. [In this video](https://www.youtube.com/watch?v=ZDaBgBNQcUk&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=21), we will tackle this problem and introduce a new approach for multi-GPU parallel computing. We'll also present a new Fortran program that utilizes Open-MP for multi-GPU computing. 

### 22.Multi-GPU parallel computing on the A-100 Nvidia GPUs
[In this video](https://www.youtube.com/watch?v=kJkW2LxsX10&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=22), we'll demonstrate how to test our new multi-GPU parallel computing method. We will use up to 4 GPUs on a single node and conduct these tests as batch jobs from the work folder on the Saga Linux cluster. Additionally, we'll guide you through a process to detect any errors in this new multi-GPU parallel computing approach.

### 23.Open-MP with Open-ACC offloading for multi-GPU parallel computing
[In this video](https://www.youtube.com/watch?v=xSxpfPKFcEY&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=23), we explore multi-GPU parallel computing by combining Open-MP with Open-ACC. We introduce a Fortran program designed for multi-GPU parallel computing that utilizes both Open-MP and Open-ACC for offloading. 
We will use two GPUs on a single node and conduct our tests as batch jobs from the work folder on the Saga Linux cluster.

### 24.Open-MP with Open-ACC offloading for multi-GPU parallel computing
[In this video](https://www.youtube.com/watch?v=3Lsn7-hsd0s&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=24), we will address the issue highlighted in our previous video. To resolve it, we need to use the data directive when offloading with Open-ACC, especially important in multi-GPU parallel computing where the compiler needs extra guidance. Simply using the kernels directive, as we did with a single GPU, is not enough for multi-GPU setups. 
We will use up to 4 GPUs on a single node and conduct these tests as batch jobs from the work folder on the Saga Linux cluster.

### 25.CUDA Fortran
[In this video](https://www.youtube.com/watch?v=zuKWhh0mSZE&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=25), we'll focus on GPU offloading through CUDA Fortran. CUDA, created by NVIDIA, is a parallel computing platform and API that allows for programming directly on GPUs. 
CUDA Fortran is an enhancement to the Fortran programming language, enabling it to utilize the features of CUDA directly.
Previously, we mainly used directive-based GPU offloading with Open-ACC and Open-MP. 
Unlike these approaches, CUDA is a kernel-based offloading, that involves explicitly writing kernels to manage offloading.

### 26.GPU Summary
[In this video](https://www.youtube.com/watch?v=Su2LJB6Yl0s&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=26), we'll summarize the GPU offloading methods we've covered so far. 
In our first video with GPU offloading, we explored language-based offloading using the do concurrent construct available in Fortran. 
Following that, we primarily focused on directive-based GPU offloading with OpenACC and OpenMP. 
In our most recent video, we used kernel-based offloading with CUDA Fortran, which involves explicitly writing kernels to handle offloading.

### 27.Scheduling Jobs from the Home folder
In earlier videos, we have used job scripts for parallel computing. [In this video](https://www.youtube.com/watch?v=6ExMu8ee3sE&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=27), we'll examine more details of scheduling jobs on the Fram Linux cluster with the Slurm management system. 
Please note that we will revisit some topics we've covered before, so there will be some repetition.
The purpose of Slurm (Simple Linux Utility for Resource Management) is to provide a robust, scalable, and efficient way to manage and schedule jobs on clusters in high-performance computing (HPC) environments.

### 28.Scheduling Jobs from the Work folder
[In this video](https://www.youtube.com/watch?v=cjqgQXTOhmI&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=28), we'll continue to explore details about scheduling jobs on the Fram Linux cluster with the slurm management system. More specifically, we will provide more details about submitting jobs from the work folder.

### 29.Jobs on Fram
[In this video](https://www.youtube.com/watch?v=MjgDxKUxWsY&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=29), we will provide more details on executing jobs on Fram Linux cluster. 
In our previous videos, we concentrated on performance testing for different parallel computing methods for matrix multiplication using a Fortran program. During these tests, we utilized some Python testing scripts. 
Once the testing phase is complete, you will schedule your computation code as a production job. We will offer some suggestions on how to structure a job script for this purpose.

### 30.More Jobs on Fram
[In this video](https://www.youtube.com/watch?v=FpAfNC9fjks&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=30), we will provide more details on executing jobs on Fram Linux cluster. 
We will attempt to increase the matrix dimension to 65536 and execute this matrix multiplication process as a parallel computation on the Fram Linux cluster using OpenMP.
When we used a matrix dimension of 32768, the memory usage was about 24 gigabytes. Therefore, increasing the dimension to 65536 would result in a memory usage of approximately 100 gigabytes.
Thus, we need to use the big memory compute nodes on the Fram Linux cluster. 

### 31.MKL on Fram
[In this video](https://www.youtube.com/watch?v=AZyMgduUt54&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=31), we will explain how to schedule jobs on the Fram Linux cluster using the Math Kernel Library (MKL) to perform matrix multiplication.
MKL is an optimized library by Intel offering a variety of mathematical functions for scientific computing. It includes: Linear Algebra, Fast Fourier Transforms, Vector Math, Statistics and Data Fitting. 
MKL is designed to maximize the performance of Intel processors through parallelism and vectorization, making it ideal for high-performance computing.

### 32.Job Efficiency on Fram
[In this video](https://www.youtube.com/watch?v=0afG3u0Odg8&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=32), we will examine the efficiency of jobs on the Fram Linux cluster.
We will look at jobs scheduled from 22th of February 2023, to 21th of September 2024, covering a period of about 17 months. During this time, 900 000 jobs were scheduled on the Fram Linux cluster.
If you've made an effort to perform parallel computing but are experiencing low CPU efficiency, you can request [extended user support (EUS)](https://documentation.sigma2.no/getting_help/extended_support/eus.html). 
These projects do not require any funding or in-kind contributions from the project or user. The total work effort should not exceed 70-80 hours over a maximum of 3-4 weeks. 

### 33.Jobs on Betzy
[In this video](https://www.youtube.com/watch?v=UDLqdJVOtjg&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=33), we will provide more details on executing jobs on Betzy Linux cluster. 
In our previous videos, we concentrated on performance testing for different parallel computing methods for matrix multiplication using a Fortran program. During these tests, we utilized some Python testing scripts. 
Once the testing phase is complete, you will schedule your computation code as a production job. We will offer some suggestions on how to structure a job script for this purpose.

### 34.MKL on Betzy
[In this video](https://www.youtube.com/watch?v=EBk_4fgjaK8&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=34), we will explain how to schedule jobs on the Betzy Linux cluster using the Math Kernel Library (MKL) to perform matrix multiplication.
MKL is an optimized library by Intel offering a variety of mathematical functions for scientific computing. It includes: Linear Algebra, Fast Fourier Transforms, Vector Math, Statistics and Data Fitting. 
MKL is designed to maximize the performance of Intel processors through parallelism and vectorization, making it ideal for high-performance computing.

### 35.Job Efficiency on Betzy
[In this video](https://www.youtube.com/watch?v=ZF_zwyv7Qmw&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=35), we will examine the efficiency of jobs on the Betzy Linux cluster.
We will look at jobs scheduled from 25th of February 2023, to 10th of August 2024, covering a period of about 16 months. During this time, 400 000 jobs were scheduled on the Betzy Linux cluster.
If you've made an effort to perform parallel computing but are experiencing low CPU efficiency, you can request [extended user support (EUS)](https://documentation.sigma2.no/getting_help/extended_support/eus.html). 
These projects do not require any funding or in-kind contributions from the project or user. The total work effort should not exceed 70-80 hours over a maximum of 3-4 weeks. 

### 36.Fortran and Python
[In this video](https://www.youtube.com/watch?v=zJbayn6HVvo&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=36), we will perform matrix multiplication using a Python program and demonstrate integrating Python with Fortran. 
We will use a NumPy routine for matrix multiplication, which efficiently multiplies matrices according to linear algebra rules. This routine is about ten thousand times faster than a triple nested loop in Python.
Fortran to Python, a tool within NumPy, connects Python and Fortran by generating Fortran modules, enabling Python to call Fortran codes. This integration is beneficial for leveraging Fortran's performance in numerical computations. 
We will execute a Python program using a Fortran module for initialisation, which is about 100 times faster than without the module.

### 37.Python on Fram
[In this video](https://www.youtube.com/watch?v=wwuGphSlEFA&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=37), we will perform matrix multiplication using Python and provide insights into executing production jobs on the Fram Linux cluster. 
The matrix multiplication finished in about 88 seconds, confirming that NumPy routines can leverage OpenMP for parallel processing. 
This was roughly 20 seconds slower than both the Fortran (video 29) and Fortran MKL (video 31) programs on the Fram Linux cluster.

### 38.MKL with Python on Fram
[In this video](https://www.youtube.com/watch?v=w-Zh7e640dI&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=38), we will use a Python program with the Fortran MKL library on the Fram Linux cluster. 
The matrix multiplication took about 92 seconds with MKL, compared to 88 seconds with NumPy (video 37), showing similar performance. 
This confirms that Fortran MKL routines can be effectively used with Python. However, it is advisable to first check if your calculation can be performed with NumPy before considering Fortran MKL.

### 39.Python on Betzy
[In this video](https://www.youtube.com/watch?v=P8temHvvcqk&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=39), we will perform matrix multiplication using Python and provide details on executing production jobs on the Betzy Linux cluster. 
The matrix multiplication finished in about 35 seconds, confirming that NumPy routines can leverage OpenMP for parallel processing. 
Compared to previous videos, the NumPy routine's performance is between the 39 seconds with the Fortran program (video 33) and the 31 seconds with the Fortran MKL routine (video 34).

### 40.MKL with Python on Betzy
[In this video](https://www.youtube.com/watch?v=jVZdyW1kH6I&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=40), we will use a Python program with the Fortran MKL library on the Betzy Linux cluster. 
The matrix multiplication took about 82 seconds with the Fortran MKL routine, whereas the NumPy routine completed it in 35 and 37 seconds in our last video. This shows that NumPy offers better performance on Betzy. 
It is wise to check if your calculation can be done with NumPy before considering Fortran MKL. Both libraries generally outperform independent efforts, so exploring high-performance libraries like NumPy and Fortran MKL can be very beneficial. 

### 41.Cray OpenMP on Olivia
[In this video](https://www.youtube.com/watch?v=TR8M2Mzk780&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=41), we will explore matrix multiplication using a triple nested loop with OpenMP for parallel computing, utilizing the Cray Fortran compiler loaded through the Cray module system on the Olivia Linux cluster.
We will also demonstrate how to submit jobs to the Slurm queue on Olivia and show how to use the Cray module system on Olivia, both in your home environment and within your Slurm job scripts.

### 42.More Cray OpenMP on Olivia
[In this video](https://www.youtube.com/watch?v=CVasTSdD-CQ&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=42), we will perform matrix multiplication using a triple nested loop with OpenMP for parallel computing, utilizing the Cray Fortran compiler loaded through the Cray module system on the Olivia Linux cluster.
Additionally, we will demonstrate how to use allocatable matrices in a Fortran program and how modifying the loop ordering of a Fortran do-loop can enhance the program's performance.

### 43.Cray MKL on Olivia
[In this video](https://www.youtube.com/watch?v=N_9ecZw9paw&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=43), we will use the MKL routine for matrix multiplication with OpenMP for parallel computing, utilizing the Fortran compiler loaded through the Cray module system. 
We will demonstrate that using MKL results in a significant performance improvement compared to the results achieved in video 42. 
The Math Kernel Library delivers outstanding performance on the Olivia Linux cluster when used with the Cray module system, particularly when compared to its performance on Betzy and Fram.

### 44.Cray MKL Scaling on Olivia
[In this video](https://www.youtube.com/watch?v=afF6ujKZIQ0&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=44), we will use the MKL routine for matrix multiplication with OpenMP for parallel computing, utilizing the Fortran compiler loaded through the Cray module system. 
Additionally, we will conduct a scaling analysis of the Fortran program using MKL with OpenMP on the Olivia Linux cluster.
This involves measuring computation times with varying numbers of threads, starting from a single thread and scaling up to 256 threads.
Scaling analysis in parallel computing is a method used to evaluate and understand how efficiently a parallel system performs as the number of processing units increases. 
It helps determine whether adding more computational resources improves performance and by how much. Scaling analysis is crucial for optimizing parallel systems and ensuring that resources are used effectively.

### 45.NRIS Intel OpenMP Scaling on Olivia
[In this video](https://www.youtube.com/watch?v=BdmUxeXKtFc&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=46), we will perform matrix multiplication using a triple nested loop with OpenMP for parallel computing, utilizing the Intel Fortran compiler loaded through the NRIS module system. 
Additionally, we will conduct a scaling analysis of the Fortran program using the triple nested loop with OpenMP on the Olivia Linux cluster.

### 46.NRIS Intel MKL Scaling on Olivia
[In this video](https://www.youtube.com/watch?v=uHqbu8PwzjA&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=47), we will use the MKL routine for matrix multiplication with OpenMP for parallel computing, utilizing the Intel Fortran compiler loaded through the NRIS module system. 
Additionally, we will conduct a scaling analysis of the Fortran program using MKL with OpenMP on the Olivia Linux cluster.

### 47.Cray Intel OpenMP Scaling on Olivia
[In this video](https://www.youtube.com/watch?v=TKjph4M7R08&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=45), we will perform matrix multiplication using a triple nested loop with OpenMP for parallel computing, utilizing the Intel Fortran compiler loaded through the Cray module system. 
Additionally, we will conduct a scaling analysis of the Fortran program using the triple nested loop with OpenMP on the Olivia Linux cluster.

### 48.Cray Intel MKL Scaling on Olivia
[In this video](https://www.youtube.com/watch?v=CuVhnlaXAuw&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=48), we will use the MKL routine for matrix multiplication with OpenMP for parallel computing, utilizing the Intel Fortran compiler loaded through the Cray module system. 
Additionally, we will conduct a scaling analysis of the Fortran program using MKL with OpenMP on the Olivia Linux cluster.

### 49.Summary of Scaling on Olivia
[In this video](https://www.youtube.com/watch?v=e7PxXJYkoac&list=PLoR6m-sar9AibHwGSFUZQ9QNxa5dSLl7O&index=49), we provide a summary of the scaling analysis conducted on Olivia, as detailed in videos 44 through 48. We also compare these results with the scaling analysis performed on Fram and Betzy. 
The findings reveal that the new Olivia Linux cluster offers significantly better performance than both Fram and Betzy.






## Other topics