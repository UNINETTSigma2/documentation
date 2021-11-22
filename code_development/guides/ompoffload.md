---
orphan: true
---

```{index} GPU; Introduction to OpenMP-Offload;OpenMP
```

Introduction
============
This tutorial provides insights into the GPU programming using OpenMP (OMP) offload. The goal of this document is to quickly enable readers to run and test some OpenMP offload codes on {ref}`betzy` platform. We assume that readers have previous knowledge of C/C++ programming, and are also aware of some of the commonly used OpenMP directives.

After reading this tutorial one should be able to answer:

* What is offloading in OpenMP?
* How to compile OpenMP code with offloading support on {ref}`betzy` platform?
* How to invoke GPUs from the OpenMP code?
* How to optimize the performance by using advanced features of the OpenMP library?

OpenMP - 4.0
============
Let's start with a short introduction to OpenMP-4.0. Heterogeneous systems, including Supercomputers, are optimized for low latency as well as for high throughput. The high throughput comes from the specialized co-processor, which we know by the name GPUs. In general, throughput is about the amount of data processed or transferred by a device in unit-time, whereas latency is about how fast the data can be transferred or loaded. Programmable GPUs offer a low Energy/Performance ratio, which is a good thing, but GPUs also expose issues related to `programmability`, `performance`, and `portability`. OpenMP 4.0 is thus an approach that enables resolving these three issues in one place. 

OpenMP is a popular shared-memory parallel programming specification which pioneered the unification of proprietary languages into an industry standard. OpenMP 4.0 is the next step to expand its scope from traditional multicore CPUs to advanced programmable Accelerators. When OpenMP 4.0 was rolled out, it took a departure from the traditional `openmp` work-sharing constructs and added support for offloading tasks to other devices. Apart from several other key features in OpenMP 4.0, the major shift that distinguishes it from its predecessor is the ability to `offload`. 

Now, what is offloading in the first place? `Offloading` often refers to taking the computation from the `Host` to the `Target`. The OMP program starts executing on the host, which is generally the CPU, and the computing task can be offloaded to the target device. The target device could be an accelerator or even the CPU itself. In this tutorial, we will focus on the GPUs as the target device; however, it is possible in OpenMP-4.0 to offload computation on other types of coprocessors, like FPGAs.

Clang/LLVM compiler with OpenMP 4.0 support
===========================================
Many compilers support OpenMP 4.0, 4.5, and 5.0 directives and are targeted at NVIDIA GPUs; one of the most prevalent hardware GPUs. `Clang` compiler is among the ones that are really moving quickly in accommodating OpenMP offload features. It is a collaborative effort of multiple vendors, like IBM, TI, Intel to bring OpenMP to Clang. Other compilers such as GCC have also some support for OpenMP 4.0; however, the performance might be limited depending on the compiler's version. 

(clang11)=

Clang on Betzy
=============
Building the Clang compiler with OpenMP-offload and Nvidia support is a chore at this point. It takes a lot of work to get it to work with NVIDIA-GPUs. Luckily, now Clang with OpenMP-offload is available on {ref}`betzy` for its users. 
The Clang compiler can be made available on {ref}`betzy` using the following command.

```bash
$ module load Clang/11.0.1-gcccuda-2020b
```

Case study.
===========
For the sake of consistency, we will use the same [Mandelbrot](https://en.wikipedia.org/wiki/Mandelbrot_set) example that has been used in the {ref}`OpenACC<asyncopenacc>` tutorial. We will go through the code and incrementally add OpenMP directives so that the user can write an OpenMP-offload code without any apparent effort. 

Now, we focus our attention on the `mandelbrot.c` code. For the convenience of our readers, we have copied the `mandelbrot.c` code down below.

(mandelbrot_C)=

```{note}
The complete code is provided at the bottom of this page, under resources section.
``` 

```{eval-rst}
:download:`mandelbrot_serial.c <ompoffload/mandelbrot_serial.c>`

```

```{eval-rst}
.. literalinclude:: ompoffload/mandelbrot_serial.c
   :language: c++

```
The Mandelbrot set is a well-known function that refers to fractal sets. It has the form:
```{math} f_{c}(z)=z^{2}+c
```
 In short, the {ref}`mandelbrot.c<mandelbrot_C>` file produces a fractal picture using the mandelbrot function, in which the function generates the Mandelbrot set of complex numbers at a given pixel. Further reading can be found at [Mandelbrot](https://en.wikipedia.org/wiki/Mandelbrot_set).


Here is how the actual code looks like, and the highlighted lines show how the pixel value is updated with the mandelbrot value. 

```{eval-rst}
.. literalinclude:: ompoffload/mandelbrot_serial.c
   :language: c++
   :lines: 121-135
   :emphasize-lines: 8-9

```

OpenMP on CPU.
==============
Let's look at our code again with a special focus on the highlighted region. Here, it can be seen that there is a nested for loop. The first 'y' loop scans the pixels of the image vertically, and the inner loop 'x' scans the image horizontally. In this way, the whole image is scanned pixel by pixel. For each pixel, the mandelbrot function is called and the old pixel value is swapped with the new value.

As we are using serial code, the first step is to see how much time a serial implementation would take on the CPU without any parallelization. 

```{note}
We assume that `Clang` compiler is already loaded in your work environment; as it is mentioned {ref}`above<clang11>`.
```

The serial version of the program can be compiled and run by using the following commands on  {ref}`betzy`.

```bash
$ make serial
$ srun --ntasks=1 --account=<your project number> --time=00:10:00 --mem-per-cpu=125M --partition=accel ./serial
```

We found that executing a `1280x720` pixels image with `10,000` iterations takes `8` seconds. 


Let’s build upon this and start applying OpenMP directives to transform our serial code into a parallel code on the CPU. As we know, the potential parallelizable region is the code with the nested ‘for’ loop, as shown in the figure. 

```{eval-rst}
.. literalinclude:: ompoffload/mandelbrot_serial.c
   :language: c++
   :lines: 121-135
   :emphasize-lines: 6-11

```

Let's start off by parallelizing the code on the CPU using regular OMP directives. We will also use this implementation to benchmark our OMP(GPU) code. The OMP(CPU) compilers are very well tested and optimized for high performance, in addition, it is very easy to use: we simply need to type `omp parallel` before the first 'for' loop. Introducing the `parallel` directive allows the compiler to spawn a team of threads; it won’t parallelize anything but executes the same code on multiple threads.

On {ref}`betzy`, the execution time with the `parallel` directive takes 15 seconds, which is higher than the value obtained using the `serial` version of our code.

We proceed by adding another directive: `parallel for`. This is the work-sharing construct that divides the work across the number of threads in a group.

Here we go, with the use of the `parallel for` clause, the code takes 0.54 seconds to execute. The performance is improved by a factor of 20 compared to the serial case.

[//]:# (It is also possible to fine-tune the `parallel for` by using the `omp schedule` clause, which can be used to determine how the iterations are spread out between the threads.)

```{eval-rst}
.. literalinclude:: ompoffload/omptarget.c
   :language: c++
   :lines: 2-14
   :emphasize-lines: 1,5

```

```{note}
Thread creation is an expensive operation, thread synchronization is also a point of concern which may deteriorate the overall performance. So, don’t let the compiler do all the optimizations for you; use OpenMP wisely.
```

Let’s move ahead with `omp offload`.

OpenMP on GPU.
==============
In leveraging the high throughput capability of the GPUs, OpenMP 4.0 offers special constructs which take the compute-intensive tasks from the CPU, perform computation on the GPU and bring the computed result back to the CPU. The main motivation behind these heterogeneous architecture-specific constructs is to provide performance and portability in one place. A user doesn’t need to know the low-level accelerator-specific language before writing their device-specific code. Also, general-purpose and special-purpose code can be maintained in one place, thus the 'offloading' offloads some of the programmer’s work too.

Accelerator-specific code starts with `#pragma omp target` directive. The region within the `target` scope is also called the target region. As soon as the host thread reaches the target region, a new target thread gets created. Before doing any computation, the required data must also be mapped to the target device. OpenMP uses the `map` keyword to transfer data `to` and `from` the GPUs. Devices have their own memory space where they can store variables and data. It is important to note that the host thread can’t access the device thread or the device data. Also, OpenMP executes the target region in a blocking step, which means the CPU threads wait for the execution of the target region to finish before continuing the execution on the CPU. However, it is possible to override the default blocking mechanism into a nonblocking one. It is also interesting that we can use regular OpenMP directives, like `parallel for`, within the target region.

Now we have enough theory to get our feet wet with the `Offloading` code.

For the OMP(GPU), we are going to use the `target` directive. The GPU/Accelerator-specific code is encompassed between the target region. The target region guarantees that the `target` directive takes your thread of execution to the target device. The target device could be another CPU, GPU, DSP, or FPGA; in a broader sense, the `target` directive takes the thread of execution and moves it elsewhere. Here one important thing to note is that the program starts executing on the host device, and when the `target` construct is hit by the main thread, a new thread is spawned, we call this thread `new initial thread`. This initial thread executes on the target device, and when this new initial thread hits a parallel construct within the target region, it becomes the master thread. It is something different from the regular OpenMP where there was only one initial thread. It is also important to note that the new initial thread may or may not run on the GPU, depending on the condition of whether the GPU support is available on the system or not. If there is no GPU available on the system then the `new initial thread` will run on the host device. To be sure if the thread is running on the GPU or CPU, one could use `omp_is_initial_device()`. The function returns false if the thread is running on the GPU.

The minimal example with OMP(offload) is shown below.

```{eval-rst}
.. literalinclude:: ompoffload/omptarget.c
   :language: c++
   :lines: 17-27
   :emphasize-lines: 1

```
```{note}
`Clang` compiler should be available in your {ref}`betzy` work-environment to compile/run the code, as it is mentioned {ref}`here<clang11>`. Or simply copy/paste the following command on {ref}`betzy` terminal.
```
```bash
$ module load Clang/11.0.1-gcccuda-2020b
```

In our case-study, we create a Makefile, which helps building the code in a simplified way using the `Make` command. To build the code with OMP(offload) support, one must type `make offload` on the terminal. The users have to make sure that all the required modules are loaded into the environment before building the code.

The bare minimum offload-code takes 148 seconds, which is almost 14 times slower than the `serial` version of the code. This is because the code is running on only one thread on the GPU and is not exploiting the entire GPU-accelerator. Moreover, the time cost of the data transfer between CPU and GPU has also been added up to the computation cost.

Let’s try to improve our code by adding `teams` construct. The `teams` construct starts a group of threads on the device, each group is called a "team" in the OpenMP terminology. This is somewhat similar to the `parallel` construct, but the main difference is that in the `teams` construct, each thread forms its team, that is a team of only one thread. Threads within different teams are restricted to a synchronisation with each other, that is, inter-team synchronisation is not possible; however, threads can synchronise within a team.

(targetteams)=

Now let’s try to run our code with `teams` construct.

```{eval-rst}
.. literalinclude:: ompoffload/omptarget.c
   :language: c++
   :lines: 29-39
   :emphasize-lines: 1

```
Remember to rebuild the code before running.

It turns out that introducing the concept of `teams` construct has increased the computation cost to 154 seconds compared to 148 seconds in the previous version. What could be the reason behind this degradation? Well, spawning threads involves some cost. Here, all threads are computing only the same thing, which causes the observed low performance.

In regular OpenMP, when the main thread hits the parallel construct it becomes the master thread, and thus only one master thread gets created. As mentioned in the previous discussion, the `teams` construct forms a group of initial threads where each thread forms its team. When each initial thread hits the ‘parallel construct’ then it becomes the master thread of the team, and in this way multiple master threads get formed. Under the cooperation of the individual master team thread, each thread executes the parallel region simultaneously.

Let’s put our multiple threads to work by applying the `parallel for` construct.

Code example is shown below:

```{eval-rst}
.. literalinclude:: ompoffload/omptarget.c
   :language: c++
   :lines: 41-52
   :emphasize-lines: 1-2

```
 
 Let’s rebuild and run the code.

 We see a significant improvement when using the `parallel for` construct. In this case, the code takes 2.3 seconds to run, which is 67 times faster than our previous version of the code.

 One more thing we can do to optimize the `parallel for` is to use the `collapse` clause. The `collapse` clause unrolls the nested loops into a single iteration space. We now try to test how this clause improves the performance of the code.

```{eval-rst}
.. literalinclude:: ompoffload/omptarget.c
   :language: c++
   :lines: 54-65
   :emphasize-lines: 1-2

```

It is found that the use of the `collapse` clause affects slightly the performance, and now the computation cost is reduced to 2.2 seconds.

Can we get any better at improving our current code? Let’s try the `Distribute` construct. The `Distribute` construct is a work-sharing construct that distributes the iterations of the loop across the teams of threads; remember, {ref}`here<targetteams>`  we started the teams of threads by using `target teams` construct ? 

It is also possible to schedule the loop iterations into chunks using the `Schedule` clause.

Let’s try the final version of our code where we apply all improvements in one place.

```{eval-rst}
.. literalinclude:: ompoffload/omptarget.c
   :language: c++
   :lines: 67-77
   :emphasize-lines: 1

```
The final version of our code takes 0.14 seconds for the computation, which is almost 16 times faster than our previous improvement. 

In our experiment, the first optimized OMP(CPU) code takes 0.54 seconds. We thus use it as a benchmark study to evaluate the performance of the OMP(GPU) code. So, the conclusion is that OMP(GPU) is almost 4 times faster than the OMP(CPU) version.

We are still missing one major part of the OpenMP-Offload, and that is offloading the data to the device. To do the computation somewhere else, we would also require the data on which the computation is going to happen. Since OpenMP supports both ‘distributed’ and ‘shared’ memory architecture, implicit as well as explicit mapping of the variables is possible. In the ‘implicit’ mapping, the compiler decides which variables are going to be sent `to` or `from` the device, whereas in the ‘explicit mapping’, user must use `map` clause within the target region to explicitly map list variables ‘to’, ‘from’ device data environment. A mapped variable may either be in the shared or the distributed memory, and in some cases a copy is required which is determined by OpenMP implementation. 
Note that once the data is moved to the device, the device owns it. And it is not possible to reference the data directly from the CPU. To access the device data one needs to bring the data back to the CPU from the device. 

After incorporating the `map` clause, our code looks like this :

```{eval-rst}
.. literalinclude:: ompoffload/omptarget.c
   :language: c++
   :lines: 79-92
   :emphasize-lines: 1

```

At this point, we conclude that GPUs are optimized for the `throughput` whereas CPUs are optimized for the `latency`. Therefore, to benefit from using GPUs, we must give enough tasks to process per unit time on the GPU. In our code example, for instance, we care more about pixels per second than the latency of any particular pixel. 

To highlight the benefit of using GPUs, we consider an example, in which the size of our input image is increased. As previously, we rerun the code on the GPU as well as on the CPU.


```bash
$ make omp
$ srun --account=<your project number> --cpus-per-task=32 -c 32 --time=10:00 --mem-per-cpu=1G --qos=devel --partition=accel ./omp 8k 10000
``` 
The processing time on the CPU is 19.5 seconds.

```bash
$ make offload 
$ srun --ntasks=1 --time=10:00 --account=<your project number> --mem-per-cpu=1G --partition=accel --gpus=1 ./offload 8k 10000
```
Processing time on the GPU is 0.27 seconds.

Our numerical experiment shows that running the code on the GPU is 72 times faster than on the multi-core CPU.

Summary of the execution times
==========================
```{note}
The benchmarking was performed on {ref}`saga` and not on {ref}`betzy`, and you may find a slight difference in the execution times on {ref}`betzy`.
``` 

Image Size | Iterations |OMP-Directive | CPU time in ms. | GPU time in ms.
-- | -- | -- | -- | --
1280x720 | 10,000 | -- | 10869.028 | -- 
1280x720 | 10,000 | `parallel` | 15025.200 | --
1280x720 | 10,000 | `parallel for` | 542.429 | --
1280x720 | 10,000 | `target`| -- | 147998.497
1280x720 | 10,000 | `target teams` | -- | 153735.213
1280x720 | 10,000 | `target teams parallel for` | -- | 2305.166
1280x720 | 10,000 | `target teams parallel for collapse` | -- | 2296.626
1280x720 | 10,000 | `target teams distribute parallel for collapse schedule` | -- | 143.434
8K	 | 10,000 | `parallel for` |  19591.378 | --
8k	 | 10,000 | `target teams distribute parallel for collapse schedule` | -- | 268.179


Resources
=========

The complete code is available in compressed format and can be downloaded from the given link.

```{eval-rst}
:download:`mandelbrot_gpu.tar.gz <ompoffload/mandelbrot_gpu.tar.gz>`

```

One can download the given `tarball` file on his/her computer and copy it to {ref}`betzy` using `scp` command, as shown below.

```bash
$ scp <source_directory/mandelbrot_gpu.tar.gz> username@betzy.sigma2.no:/cluster/home/<target_directory>
```
`source directory` should be the absolute path of the downloaded `tarball` on your computer, and the target directory should be the directory where you want to keep and uncompress the `tarball`.

To uncompress the `tarball` file, execute the following command on the terminal.

```bash
$ tar -zxvf mandelbrot_gpu.tar.gz
```



Makefile
========
For our sample code, we used `Makefile` to build. `Makefile` contains all the code that is needed to automate the boring task of transforming the source code into an executable. One could argue; why not `batch` script? The advantage of `make` over the script is that one can specify the relationships between the elements of the program to `make`, and through this relationship together with timestamps it can figure out exactly what steps need to be repeated to produce the desired program each time. In short, it saves time by optimizing the build process.

A brief version of the `Makefile` is listed here.

```{eval-rst}
.. literalinclude:: ompoffload/Makefile.txt
   :language: make
   :lines: 1-15

```

Compilation process
===================

We briefly describe the syntax of the compilation process with the Clang compiler to implement the OpenMP offload targeting NVIDIA-GPUs on {ref}`betzy` platform. The syntax is given below:

```console
clang -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda -march=sm_80 gpu_code.c
```

Here the flag `-fopenmp` activates the OpenMP directives (i.e. #pragma omp). The option `-fopenmp-targets` is used to enable target `offloading` to `NVIDIA-GPUs` and the `-Xopenmp-target` flag enables options to be passed to the target offloading toolchain. Last, the flag `-march` specifies the name of the `NVIDIA GPU` architecture.
