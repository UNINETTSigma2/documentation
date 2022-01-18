# Mapping OpenACC to OpenMP offloading 

# Summary

This documentation is designed for beginners in GPU-programming and who want to get familiar with OpenACC and OpenMP offloading models. Here we provide an overview of these two programming models. Specifically, we provide some insights into the functionality of these models and perform experiments involving different directives and discuss their performance. This is achieved through the use of an application based on solving numerically the Laplace equation. Such experiments reveal the benefit of the use of GPU, which in our case manifests by an increase of the performance by almost a factor of 22. We further carry out a comparative study between the OpenACC and OpenMP models in the aim of converting OpenACC to OpenMP. In this context, we present a short overview of the open-source OpenACC compiler Clacc, which is designed based on translating OpenACC to OpenMP in Clang. This documentation ultimately aims at initiating developers'/users' interest in GPU-programming. We therefore expect developers/users, by the end of this documentation, to be able to: 

*	Recognise the benefits of GPU-programming.
*	Acquire some basic knowledge of the GPU-architecture and the functionality of the underlying models.
*	Use appropriate constructs and clauses on either programming model to offload compute regions to a GPU device.
*	Identify and assess differences and similarities between the OpenACC and OpenMP offload features.
*	Get some highlights of available open-source OpenACC compilers.


#### Table of Contents

- [Introduction](#introduction)
- [Computational model](#computational-model)
- [GPU-accelerators](#gpu-accelerators)
- [Experiment on OpenACC offloading](#experiment-on-openacc-offloading)
- [Experiment on OpenMP offloading](#experiment-on-openmp-offloading)
- [Mapping OpenACC to OpenMP](#mapping-openacc-to-openmp)
- [Discussion on porting OpenACC to OpenMP](#discussion-on-porting-openacc-to-openmp)
- [Conclusion](#conclusion)


# Introduction

OpenACC and OpenMP are the most widely used programming models for heterogeneous computing on moderm HPC architectures. OpenACC was developed a decade ago and was designed for parallel programming of heterogenous systems (i.e. CPU host and GPU device). Whereas OpenMP is historically known to be directed to shared-memory multi-core programming, and only recently has provided support for heterogenous systems. OpenACC and OpenMP are directive-based programming models for offloading compute regions from CPU host to GPU devices. These models are referred to as Application Programming Interfaces (APIs), which here enable to communicate between two heterogenous systems and specifically enable offloading to target devices. The offloading process is controlled by a set of compiler directives, library runtime routines as well as environment variables. These three components will be addressed in the following for both models. Furthermore, differences and similarities will be assessed in the aim of converting OpenACC to OpenMP.

*Motivation:* NVIDIA-based Programming models are bounded by some barriers related to the GPU-architecture. 
Such models do not have direct support on different devices such as AMD and Intel accelerators nor by 
the corresponding compilers, i.e. *Clang/Flang* and *Icx/Ifx*. Removing such barriers is one of the bottleneck  
in GPU-programming, which is the case for instance with OpenACC. The latter is one of 
the most popular programming model that requires a special focus in terms of support on available architectures.  

As far as we know, the only open-source compiler that fully supports OpenACC for offloading to NVIDIA and AMD devices is GCC. The GCC's performance, however, suffers from some weaknesses and poses some [challenges](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8639349), which limit its extension. Although the Cray Compilation Environment [(CCE)](https://support.hpe.com/hpesc/public/docDisplay?docId=a00115296en_us&page=OpenACC_Use.html#OpenACC2.6) has full support of OpenACC 2.0 and partial support of OpenACC 2.6, the support is limited only to Fortran, and thus no support for C or C++. This lack of support for OpenACC calls for an alternative that goes beyond the GCC compiler, and which ensures higher performance. On the other hand, the OpenMP offloading is supported on multiple devices by a set of compilers such as *Clang/Flang* and *Cray*, and *Icx/Ifx* which are well-known to provide the highest performance with respect to GCC. Therefore, converting OpenACC to OpenMP becomes a necessity to overcome the lack of stable implementations for all relevant hardware vendors, and to extend the OpenACC implementations to cover various GPU-architectures. In this context, there has been a project funded by [Exascale Computing Project](https://www.exascaleproject.org/highlight/clacc-an-open-source-openacc-compiler-and-source-code-translation-project/) and published [here](https://ieeexplore.ieee.org/document/8639349), which aims at developing an open-source OpenACC compiler. This documentation is thus inspired by this project and is motivated by the need to document how to translate OpenACC to OpenMP offloading.
 
 
 This documentation is organised as follows. In [sec. II](#computational-model), we provide a computational model, which is based on solving the Laplace equation. [Section III](#comparative-study-openacc-versus-openmp) is devoted to 
 the analysis of experiments performed using the OpenACC and OpenMP offload features. [Section IV](#discussion-on-porting-openacc-to-openmp) is directed to a discussion about converting OpenACC to OpenMP in the Clang/Flang compiler. Finally, conclusions are given in [Sec. V](#conclusion).
 

# Computational model

We give a brief description of the numerical model used to solve the Laplace equation &Delta;f=0. For the sake of simplicity, we solve the equation in a two-dimensional (2D) uniform grid according to

```math
\Delta f(x,y)=\frac{\partial^{2} f(x,y)}{\partial x^{2}} + \frac{\partial^{2} f(x,y)}{\partial y^{2}}=0. \ \ \ \ (1)
```
Here we use the finite-difference method to approximate the partial derivative of the form $`\frac{\partial^{2} f(x)}{\partial x^{2}}`$. The spatial discretization in the second-order scheme can be written as 

```math
\frac{\partial^{2} f(x,y)}{\partial^{2} x}=\frac{f(x_{i+1},y) - 2f(x_{i},y) + f(x_{i-1},y)}{\Delta x}. \ \ \ \ (2)
```

Inserting Eq. (2) into Eq. (1) leads to this final expression 

```math
f(x_i,y_j)=\frac{f(x_{i+1},y) + f(x_{i-1},y) + f(x,y_{i+1}) + f(x,y_{i-1})}{4}. \ \ \ \ (3)
```

The Eq. (3) can be solved iteratively by defining some initial conditions that reflect the geometry of the problem at-hand. This can be done either using Gauss–Seidel method or Jacobi method. Here, we apt for the Jacobi algorithm due to its simplicity. The corresponding compute code is written in *Fortran 90* and is given below in a serial form. Note that a *C*-based code can be found [here](https://documentation.sigma2.no/code_development/guides/openacc.html?highlight=openacc).

```bash
       do while (max_err.gt.error.and.iter.le.max_iter)
         do j=2,ny-1
            do i=2,nx-1
               d2fx = f(i+1,j) + f(i-1,j)
               d2fy = f(i,j+1) + f(i,j-1)
               f_k(i,j) = 0.25*(d2fx + d2fy)
             enddo
          enddo

          max_err=0.

          do j=2,ny-1
            do i=2,nx-1
               max_err = max(dabs(f_k(i,j) - f(i,j)),max_err)
               f(i,j) = f_k(i,j)
            enddo
          enddo

          iter = iter +1
        enddo
```

# Comparative study: OpenACC versus OpenMP

In the following we first provide a short description of GPU accelerators and then perform experiments covering both the OpenACC and OpenMP implementations to accelerate the Jacobi algorithm in the aim of conducting a comparative experiment between the two programming models. The experiments are systematically performed with a fixed number of grid points (i.e. 8192 points in both `x` and `y` directions) and a fixed number of iterations that ensure the convergence of the algorithm. This is found to be 240 iterations resulting in an error of 0.001.

## GPU accelerators

We focus in this section on describing the [NVIDIA GPU accelerator](https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf) as it is considered the most powerful accelerator for artificial intelligence (AI) and high-performance computing (HPC). The NVIDIA GPU-device consists of a block of a Streaming Multiprocessor (SM) each of which is organized as a matrix of CUDA cores, as shown in [Fig. 1](#Fig1). As an example, the [NVIDIA P100 GPU-accelerators](https://images.nvidia.com/content/tesla/pdf/nvidia-tesla-p100-PCIe-datasheet.pdf) have 56 SMs and each SM has 64 CUDA cores with a total of 3584 FP32 cores/GPU or 1792 FP64 cores/GPU, while the [NVIDIA V100](https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf) has 80 SMs and each SM has 64 CUDA cores with a total of 5120 FP32/GPU or 2560 FP64/GPU, where FP32 and FP64 correspond to the single-precision Floating Point (FP) (i.e. 32 bit) and to the double precision (64 bit), respectively. 

<div align="center">
![Fig1](figs/fig-hardware.jpg) 

**Fig. 1.** *Schematic representation of a NVIDIA GPU-accelerator.*
</div>

Various NVIDIA [GPU-architectures](https://gpltech.com/wp-content/uploads/2018/11/NVIDIA-Turing-Architecture-Whitepaper.pdf) exist. As an example, we present in [Fig. 2](#Fig2) the characteristic of the NVIDIA V100 Volta architecture. As shown in the figure, the performance of the NVIDIA Volta depends on the specified architecture: V100 PCle, V100 SXMe and V100S PCle, which in trun depends, in particular, on the  Memory Bandwidth. For instance, the double precision performance associated with each architecture is respectively, 7, 7.8 and 8.2 TFLOPS (or TeraFLOPS). Here 1 TFLOPS= $`10^{12}`$ calculations per second, where FLOPS (Floiting-Point of Opertaions Per Second), in general, defines a measure of the speed of a computer to perform arithmetic operations. The peak performance for a single processor can be calculated theoretically based on the following [expression](https://en.wikipedia.org/wiki/FLOPS#cite_note-en.community.dell.com-5)

FLOPS = (Clock speed)$`\times`$(cores)$`\times`$(FLOP/cycle),  

 where FLOP is a way of encoding real numbers (i.e. FP64 or FP32 or ...). We can check the validity of the expression by calculating, for instance, the peak performance for V100 PCle, in which the Clock speed (or GPU Boost Clock) is 1.38 GHz. The totla FLOPS is (1.38 10^9 cycle/second)x5120xFLOP/cycle, which is 7.065 10^12 FLOP per second or also 7.065 TFLOPS.

<div align="center">
<img src="figs/fig-arch-volta.jpg" width="600" height="500"> 

**Fig. 2.** *Specification of the architecture of the NVIDIA Volta GPU taken from [here](https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf).*
</div>



## Experiment on OpenACC offloading

We begin first by illustrating the functionality of the OpenACC model in terms of parallelism, which is specified by the directives **kernels** or **parallel loop**. The concept of parallelism is defined precisely by the generic directives: **gang**, **worker** and **vector** as schematically depicted in [Fig. 3](#Fig3). Here, the compiler initiates the parallelism by generating parallel gangs, in which each gang consists of a set of workers represented by a matrix of threads as depicted in the inset of [Fig. 3](#Fig3). This group of threads within a gang executes the same instruction (SIMT, Single Instruction Multiple Threads) via a vectorization process. In other word, a block of loops is assigned to each gang, which gets vectorized and executed by a group of threads. Specifically, each thread executes the same kernel program but operates on different parts of the offloaded region.  

<div align="center">
![Fig3](figs/fig-software.jpg)

**Fig. 3.** *Schematic representation of the concept of parallelism (see text for more details).*
</div>


In this context, the parallelism is mapped on the GPU-device in the following way: each compute unit is associated to one gang of threads generated by the directive **gang**, in which a block of loops is assigned to. In addition, each block of loops is run on the processing element via the directive **vector**. The description of these directives and other directives implemented in our OpenACC application is summarized in the [Table 1.](#Table1). 

We move now to discuss our OpenACC experiment, in which we evaluate the performance of different compute constructs and clauses and interpret their role. The OpenACC-based code is shown below. In the left-hand side of the code, only the directive **parallel loop** is introduced. Here the construct **parallel** indicates that the compiler will generate a number of parallel gangs to execute the compute region redundantly. When it is combined with the clause **loop**, the compiler will perform the parallelism over all the generated gangs for the offloaded region. In this case the compiler copies the data first to the device in the beginning of the loop and then copies it back to the host at the end of the loop. This process repeats itself at each iteration, which makes it time consuming, thus rending the GPU-acceleration inefficient. This inefficiency is shown in [Fig. 4](#Fig4) and manifests by the increase of the computing time: 111.2 s compared to 101.77 s in the serial case. This low performance is also observed when using the construct **kernels**.   

To overcome this issue, one needs to copy the data to the device only in the beginning of the iteration and copy them back to the host at the end of the iteration, once the result converges. This can be done by introducing the data locality concepts via the directives **data**, **copyin** and **copyout**, as shown in the code application (right-hand side). Here, the clause **copyin** transfers the data to a GPU-device, while the clause **copyout** copies the data back to the host. Implementing this approach shows a vast improvement of the performance: the computing time gets reduced by almost a factor of 53: it decreases from 111.2 s to 2.12 s. One can further tune the process by adding additional control, for instance, by introducing the **collapse** clause. Collapsing two or more loops into a single loop is beneficial for the compiler, as it allows to enhance the parallelism when mapping the compute region into a device. In addition, one can specify the clause **reduction**, which allows to compute the maximum of two elements in a parallel way. These additional clauses affect slightly the computing time: it goes from 2.12 s to 1.95 s.

For completeness, we compare in [Fig. 4](#Fig4) the performance of the compute constructs **kernels** and **parallel loop**. These directives tell the compiler to transfer the control of a compute region to a GPU-device and execute it in a sequence of operations. Although these two constructs have a similar role, they differ in terms of mapping the parallelism into a device. Here, when specifying the **kernels** construct, the compiler performs the partition of the parallelism explicitly by choosing the optimal numbers of gangs, workers and the length of the vectors and also some additional clauses. Whereas, the use of the **parallel loop** construct offers some additional functionality: it allows the programmer to control the execution in a device by specifying additional clauses. At the end, the performance remains roughly the same as shown in [Fig. 4](#Fig4): the computing time is 1.97 s for the **kernels** directive and 1.95 s for the **parallel loop** directive. 
 
```bash
          **OpenACC without data locality**            |              **OpenACC with data locality**
                                                       |  !$acc data copyin(f) copyout(f_k)
   do while (max_err.gt.error.and.iter.le.max_iter)    |     do while (max_err.gt.error.and.iter.le.max_iter)
!$acc parallel loop gang worker vector                 |  !$acc parallel loop gang worker vector collapse(2)  
      do j=2,ny-1                                      |        do j=2,ny-1 
        do i=2,nx-1                                    |          do i=2,nx-1 
           d2fx = f(i+1,j) + f(i-1,j)                  |             d2fx = f(i+1,j) + f(i-1,j)
           d2fy = f(i,j+1) + f(i,j-1)                  |             d2fy = f(i,j+1) + f(i,j-1) 
           f_k(i,j) = 0.25*(d2fx + d2fy)               |             f_k(i,j) = 0.25*(d2fx + d2fy)
        enddo                                          |           enddo
      enddo                                            |         enddo
!$acc end parallel                                     |  !$acc end parallel
                                                       |
       max_err=0.                                      |          max_err=0.
                                                       |
!$acc parallel loop                                    |  !$acc parallel loop collapse(2) reduction(max:max_err)  
      do j=2,ny-1                                      |        do j=2,ny-1
        do i=2,nx-1                                    |          do i=2,nx-1
           max_err = max(dabs(f_k(i,j)-f(i,j)),max_err)|             max_err = max(dabs(f_k(i,j)-f(i,j)),max_err)
           f(i,j) = f_k(i,j)                           |             f(i,j) = f_k(i,j)
        enddo                                          |          enddo 
       enddo                                           |        enddo
!$acc end parallel                                     |  !$acc end parallel 
                                                       |
       iter = iter + 1                                 |        iter = iter + 1 
    enddo                                              |     enddo
                                                       |  !$acc end data
```

<div align="center">
![Fig4](figs/fig-acc.jpeg)

**Fig. 4.** *Performance of different OpenACC directives.*
</div>


```{note} Accoriding to the OpenACC [specification](https://www.openacc.org/sites/default/files/inline-files/OpenACC_Programming_Guide_0_0.pdf) the order of the clauses  **gang**, **worker** and **vector** should be enforced: the **gang** clause must determine the outermost loop and the **vector** clause should define the innermost parallel loop, while the **worker** clause should be in between these two clauses.```

> 
> ```{note} Different gangs operate independently.```
> 
> ```{note} When incorporating the constructs **kernels** or **parallel loop**, the compiler will generate arrays that will be copied back and forth between the host and the device if they are not already present in the device.``` 

### Compiling and running OpenACC-program

We run our OpenACC-program on the NVIDIA-GPU P100. The syntax of the compilation process is
```bash
$ nvfortran -fast -acc -Minfo=accel -o laplace_acc.exe laplace_acc.f90
or
$ nvfortran -ta=tesla:cc60 -Minfo=accel -o laplace_acc.exe laplace_acc.f90
```
where the flags *-acc* and *-⁠ta=[target]* enables OpenACC directives. The option *[target]* reflects the name of the GPU device. The latter is set to be *[tesla:cc60]* for the device name Tesla P100 and *[tesla:cc70]* for the tesla V100 device. This information can be viewed by running the command `pgaccelinfo`. Last, the flag option *-Minfo* enables the compiler to print out the feedback messages on optimizations and transformations.

The generated binary (i.e. `laplace_acc.exe`) can be launched with the use of a Slurm script as follows
```bash
#!/bin/bash
#SBATCH --account=nn1234k 
#SBATCH --job-name=laplace_acc
#SBATCH --partition=accel --gpus=1
#SBATCH --qos=devel
#SBATCH --time=00:01:00
#SBATCH --mem-per-cpu=2G
#SBATCH -o laplace_acc.out

#loading modules
module purge
module load NVHPC/21.2
 
$ srun ./laplace_acc.exe
```
In the script above, the option *--partition=accel* enables the access to the GPU, as shown [here](https://documentation.sigma2.no/code_development/guides/openacc.html?highlight=openacc). One can also use the command `sinfo` to get information about which nodes are connected to the GPUs. 

```{note} The compilation process requires loading a NVHPC module, e.g. `NVHPC/21.2` or a different version.```


## Experiment on OpenMP offloading

In this section, we carry out an experiment on [OpenMP](https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-5-1.pdf) offloading by adopting the same scenario as in the previous [section](#experiment-on-openacc-offloading) but with the use of a different GPU-architecture: AMD Mi100 accelerator. The functionality of OpenMP is similar to the one of OpenACC, although the terminology is different [cf. Fig. 1](#Fig1). In the OpenMP concept, a block of loops is offloaded to a device via the construct **target**. A set of threads is then created on each compute unit (CU) (analogous to streaming multiprocessor in NVIDIA terminology) [cf. Fig. 1](#Fig1) by means of the directive **teams** to execute the offloaded region. Here, the offloaded region (i.e. block of loops) gets assigned to teams via the clause **distribute**, and gets executed on the processing elements (PEs) (analogous to CUDA cores) by means of the directive **parallel do simd**. These directives define the concept of parallelism in OpenMP. 

The concept of parallelism is implemented using the same model described in [Section II](#computational model). The implementation is presented below for two cases: (i) OpenMP without introducing the data directive and (ii) OpenMP with the data directive. This Comparison allows us to identify the benefit of data management during the data-transfer between the host and a device. This in turn provides insights into the performance of the OpenMP features. In the left-hand-side of the OpenMP application, the arrays **f** and **f_k**, which define the main components of the compute region, are copied from the host to a device and back, respectively via the clause **map**. Note that specifying the **map** clause in this case is optional. Once the data are offloaded to a device, the parallelism gets executed according to the scenario described above. This scheme repeats itself at each iteration, which causes a low performance as shown in [Fig. 5](#Fig5). Here the computing time is 119.6 s, which is too high compared to 76.52 s in the serial case. A similar behavior is observed in the OpenACC application.

The OpenMP performance however gets improved when introducing the directive **data** in the beginning of the iteration. This implementation has the advantage of keeping the data in the device during the iteration process and copying them back to the host only at the end of the iteration. By doing so, the performance is improved by almost a factor of 22, as depicted in [Fig. 5](#Fig5): it goes from 119.6 s in the absence of the data directive to 5.4 s when the directive is introduced. As in OpenACC application, the performance can be further tuned by introducing additional clauses, specifically, the clauses **collapse** and **schedule** which are found to reduce the computing time from 5.4 s to 2.15 s. 

The description of the compute constructs and clauses used in our OpenMP application is provided in the [Table. 1](Table1) together with those of OpenACC. For further OpenMP tutorials, we refer to a different scenario implemented in C, which can be found [here](https://documentation.sigma2.no/code_development/guides/ompoffload.html).
         
```bash
          **OpenMP without data directive**            |                 **OpenMP with data directive**
                                                       |  !$omp target data map(to:f) map(from:f_k)
   do while (max_err.gt.error.and.iter.le.max_iter)    |     do while (max_err.gt.error.and.iter.le.max_iter)
!$omp target teams distribute parallel do simd         |  !$omp target teams distribute parallel do simd collapse(2) 
      map(to:f) map(from:f_k)                          |        schedule(static,1) 
      do j=2,ny-1                                      |        do j=2,ny-1 
        do i=2,nx-1                                    |          do i=2,nx-1 
           d2fx = f(i+1,j) + f(i-1,j)                  |             d2fx = f(i+1,j) + f(i-1,j)
           d2fy = f(i,j+1) + f(i,j-1)                  |             d2fy = f(i,j+1) + f(i,j-1) 
           f_k(i,j) = 0.25*(d2fx + d2fy)               |             f_k(i,j) = 0.25*(d2fx + d2fy)
        enddo                                          |           enddo
      enddo                                            |         enddo
!$omp end target teams distribute parallel do simd     |  !$omp end target teams distribute parallel do simd
                                                       |
       max_err=0.                                      |          max_err=0.
                                                       |
!$omp target teams distribute parallel do simd         |  !$omp target teams distribute parallel do simd collapse(2) 
      reduction(max:max_err)                           |         schedule(static,1) reduction(max:max_err) 
      do j=2,ny-1                                      |        do j=2,ny-1
        do i=2,nx-1                                    |          do i=2,nx-1
           max_err = max(dabs(f_k(i,j)-f(i,j)),max_err)|             max_err = max(dabs(f_k(i,j)-f(i,j)),max_err)
           f(i,j) = f_k(i,j)                           |             f(i,j) = f_k(i,j)
        enddo                                          |          enddo 
       enddo                                           |        enddo
!$omp end target teams distribute parallel do simd     |  !$omp end target teams distribute parallel do simd
                                                       |
       iter = iter + 1                                 |        iter = iter + 1 
    enddo                                              |     enddo
                                                       |  !$omp end target data
```

<div align="center">
![Fig5](figs/fig-omp.jpg)

**Fig. 5.** *Performance of different OpenMP directives.*
</div>

### Compiling and running OpenMP-program

Our OpenMP benchmark test runs on AMD Mi100 accelerator. The syntax of the compilation process can be written in the following form:

```bash
flang -fopenmp=libomp -fopenmp-targets=<target> -Xopenmp-target=<target> -march=<arch> laplace_omp.f90
```

The flag *-fopenmp* activates the OpenMP directives (i.e. !$omp [construct] in Fortran). The option *-fopenmp-targets=<target>* is used to enable the target offloading to GPU-accelerators and tells the Flang compiler to use *<target>=amdgcn-amd-amdhsa* as the AMD target. The *-Xopenmp-target* flag enables options to be passed to the target offloading toolchain. In addition, we need to specify the architecture of the GPU to be used. This is done via the flag *-march=<arch>*, where *<arch>* specifies the name of the GPU-architecture. This characteristic feature can be extracted from the machine via the command `rocminfo` for the AMD device. These commands provide a general view of the GPU-accelerator and additional related information. For instance, the AMD Mi100 accelerator architecture is specified by the flag *-march=gfx908 amd-arch*.   

> ```{note} The compilation process requires loading a AOMP module, e.g. `AOMP/13.0-2-GCCcore-10.2.0` or a newer version.```

 
## Mapping OpenACC to OpenMP

In this section, we present a direct comparison between the OpenACC and OpenMP offload features. This comparison is illustrated in the compute code below. A closer look at OpenACC and OpenMP codes reveals some similarities and differences in terms of constructs and clauses. The meaning of these directives are summerized in the [Table 1.](#Table1). Here, evaluating the behavior of OpenACC and OpenMP by one-to-one mapping is a key feature of the conversion procedure. Based on this comparison, it is seen that the syntax of both programming models is so similar, thus making the implementation of the translation procedure at the syntactic level straightforward. Therefore, carying out such a comparison is critical for determining the correct mappings to OpenMP.

```bash
                    **OpenACC**                        |                    **OpenMP**
!$acc data copyin(f) copyout(f_k)                      |  !$omp target data map(to:f) map(from:f_k)
   do while (max_err.gt.error.and.iter.le.max_iter)    |     do while (max_err.gt.error.and.iter.le.max_iter)
!$acc parallel loop gang worker vector collapse(2)     |  !$omp target teams distribute parallel do simd collapse(2) 
                                                       |        schedule(static,1) 
      do j=2,ny-1                                      |        do j=2,ny-1 
        do i=2,nx-1                                    |          do i=2,nx-1 
           d2fx = f(i+1,j) + f(i-1,j)                  |             d2fx = f(i+1,j) + f(i-1,j)
           d2fy = f(i,j+1) + f(i,j-1)                  |             d2fy = f(i,j+1) + f(i,j-1) 
           f_k(i,j) = 0.25*(d2fx + d2fy)               |             f_k(i,j) = 0.25*(d2fx + d2fy)
        enddo                                          |           enddo
      enddo                                            |         enddo
!$acc end parallel                                     |  !$omp end target teams distribute parallel do simd
                                                       |
       max_err=0.                                      |          max_err=0.
                                                       |
!$acc parallel loop collapse(2) reduction(max:max_err) |  !$omp target teams distribute parallel do simd collapse(2) 
                                                       |         schedule(static,1) reduction(max:max_err) 
      do j=2,ny-1                                      |        do j=2,ny-1
        do i=2,nx-1                                    |          do i=2,nx-1
           max_err = max(dabs(f_k(i,j)-f(i,j)),max_err)|             max_err = max(dabs(f_k(i,j)-f(i,j)),max_err)
           f(i,j) = f_k(i,j)                           |             f(i,j) = f_k(i,j)
        enddo                                          |          enddo 
       enddo                                           |        enddo
!$acc end parallel                                     |  !$omp end target teams distribute parallel do simd
                                                       |
       iter = iter + 1                                 |        iter = iter + 1 
    enddo                                              |     enddo
!$acc end data                                         |  !$omp end target data
```

OpenACC | OpenMP | interpretation |
-- | -- | -- |
acc parallel | omp target teams | to execute a compute region on a device|
acc kernels  | No explicit counterpart   | - -|
acc parallel loop gang worker vector | omp target teams distribute parallel do simd | to parallelize a block of loops on a device|
acc data     | omp target data | to share data between multiple parallel regions in a device|
-- | -- | -- |
acc loop | omp teams distribute | to workshare for parallelism on a device|
acc loop gang | omp teams(num_teams) | to partition a loop accross gangs/teams|
acc loop worker | omp parallel simd | to partition a loop across threads|
acc loop vector | omp parallel simd | - - |
num_gangs       | num_teams         | to control how many gangs/teams are created |
num_workers     | num_threads       | to control how many worker/threads are created in each gang/teams |
vector_length   | No counterpart    | to control how many data elements can be operated on |
-- | --  | -- |
acc create() | omp map(alloc:) | to allocate a memory for an array in a device|
acc copy()   | omp map(tofrom:) | to copy arrays from the host to a device and back to the host|
acc copyin() | omp map(to:) | to copy arrays to a device|
acc copyout()| omp map(from:) | to copy arrays from a device to the host|
-- | --  | -- |
acc reduction(operator:var)| omp reduction(operator:var) | to reduce the number of elements in an array to one value |
acc collapse(N)  | omp collapse(N)   | to collapse N nested loops into one loop |
No counterpart  | omp schedule(,)  | to schedule the work for each thread according to the collapsed loops|
private(var)         | private(var)          | to allocate a copy of the variable `var` on each gang/teams|
firstprivate    | firstprivate     | to allocate a copy of the variable `var` on each gang/teams and to initialise it with the value of the local thread| 

**Table 1.** *Description of various directives and clauses: OpenACC vs OpenMP.*
 
Details about library routines can be found [here](https://gcc.gnu.org/onlinedocs/libgomp/OpenACC-Runtime-Library-Routines.html) for OpenACC and [here](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-cpp-fortran-compiler-openmp/top.html) for OpenMP.

# Discussion on porting OpenACC to OpenMP

 For completeness, we provide in this section some highlights of the available open-source tools that provide support of OpenACC in terms of compilers and hardware. According to the work of [J. Vetter et al.](https://ieeexplore.ieee.org/document/8639349) and the [OpenACC website](https://www.openacc.org/tools), the only open-source compiler that supports OpenACC offloading to NVIDIA and AMD accelerators is GCC 10. Recently, there has been an effort in developing an open-source compiler to complement the existing one, thus allowing to perform experiments on a broad range of architectures. The compiler is called [Clacc](https://ieeexplore.ieee.org/document/8639349) and its development is funded by Exascale Computing Project: [Clacc project](https://www.exascaleproject.org/highlight/clacc-an-open-source-openacc-compiler-and-source-code-translation-project/), which is described in the work of [J. Vetter et al.](https://ieeexplore.ieee.org/document/8639349). We thus focus here on providing some basic features of the Clacc compiler platform, without addressing deeply the fundamental aspect of the compiler, which is beyond the scope of this documentation..

Clacc is an open-source OpenACC compiler platform that has support for [Clang](https://clang.llvm.org/) and [LLVM](https://llvm.org/), and aims at facilitating GPU-programming in its broad use. The key behind the design of Clacc is based on translating OpenACC to OpenMP, taking advantage of the existing OpenMP debugging tools to be re-used for OpenACC. Clacc was designed to mimic the exact behavior of OpenMP as explicit as possible. The Clacc strategy for interpreting OpenACC is based on one-to-one mapping of [OpenACC directives to OpenMP directives](https://ieeexplore.ieee.org/document/8639349) as we have already shown in the [Table 1.](#Table1) above.

Despite the new development of Clacc compiler platform, there is still major need to further extend the compiler as it suffers from some limitations, [mainly](https://ieeexplore.ieee.org/document/8639349): (i) translating OpenACC to OpenMP in Clang/Flang is currently supported only in C and Fortran but not yet in C++. (ii) Clacc has so far focused primarily on compute constructs, and thus lacks support of data-sharing between the CPU-host and a GPU-device. These limitations however are expected to be overcame in the near future. So far, Clacc has been tested and benchmarked against a series of different configurations, and it is found to provide an acceptable GPU-performance, as stated [here](https://www.exascaleproject.org/highlight/clacc-an-open-source-openacc-compiler-and-source-code-translation-project/). Note that Clacc is publicly available [here](https://github.com/llvm-doe-org/llvm-project/wiki).

 

# Conclusion

In conclusion, we have presented an overview of the OpenACC and OpenMP offload features via an application based on solving the Laplace equation in a 2D uniform grid. This benchmark application was used to experiment the performance of some of the basic directives and clauses in order to highlight the gain behind the use of GPU-accelerators. The performance here was found to be improved by almost a factor of 22. We have also presented an evaluation of differences and similarities between OpenACC and OpenMP programming models. Furthermore, we have illustrated a one-to-one mapping of OpenACC directives to OpenMP directives in the aim of porting OpenACC to OpenMP. In this context, we have emphasised the recent development of the Clacc compiler platform aiming for such a translation procedure, although the platform support is so far limited to C and fortran and lacks data-transfer in host-device. 

Last but not least, writing an efficient GPU-based program requires some basic knowledge of the target architectures and how regions of a program is mapped into a target device. This documentation thus was designed to provide such basic knowledge in the aim of triggering the interest of developers/users to GPU-programming. It thus functions as a benchmark for future advanced GPU-based parallel programming models. 

{ref}`You can get in touch with the NRIS GPU team for help in porting your application here<extended-support-gpu>`




