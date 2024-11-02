(intel-2024-1API-workshop)=

#  Intel oneAPI Workshop (6, 13 and 27 November) 

The Norwegian Research Infrastructure Services (**NRIS**) is facilitating Intel oneAPI Workshop 
offered by intel. The workshop will cover a wide range of topic from oneAPI, IFX and ICX compilers
openMP, Python, advisor, mpi to vtune. The workshop will run as 3 series of events 
in November (6, 13 and 27).  See the agenda below for more details.  This workshop is ideal for users
 who wants to know more on these topics and developers. A beginner level in either C++/Fortran/Python is 
 needed to understand the courses.

## [Agenda](https://drive.google.com/drive/folders/14MNdoEfvdfwEhDA-_nSKd8AvHSKro6rF)

### Day 1: ONEAPI&ITDC/COMPILERS/OPENMP (6th November 09:00-12:30 CET)

- **09:00 - Welcome**
- **09:05 - Introduction to oneAPI and the new Intel Tiber Developer Cloud (ITDC) infrastructure**
	- Concept and purpose for the oneAPI Standardization initiative
	- Intel’s oneAPI Solutions – Toolkits with Compilers, libs, analysis, and migration tools
        - Short introduction to SYCL: concept, references (eg Gromacs), learning material
  	- Intel oneAPI plug-ins for Nvidia and AMD hardware (CPU and GPUs)
	- ITDC: service platform for developing with the latest Intel HW & SW including learning material and tutorials

- **09:35 - Benefits of the new LLVM based Intel compilers (IFX & ICX)**
    - Improved code optimizations and tighter integration with Intel hardware.
    - Full support for the latest Fortran, C, and C++ standards with better diagnostics.
    - Easy migration from Intel Classic to LLVM compilers, with minimal code changes.
    -  Cross-platform compatibility, compiling for CPUs, GPUs, and FPGAs.

Intel’s new LLVM-based IFX and ICX compilers deliver enhanced performance, modern language support,
and better optimization. This session will highlight the key advantages of these compilers, part of Intel’s oneAPI initiative, and offer insights into migrating from Intel’s classic compilers to the LLVM-based
ones. Perfect for developers aiming to migrate and maximize their application performance using Intel’s
latest compiler tools.

- **10:00 - Break**

- **10:15 - Offloading C++ code to GPU with OpenMP**
    - Use OpenMP directives to accelerate C++ code on GPUs.
    - Minimize code changes while gaining parallel performance.
    - Simplify GPU offloading without complex modifications.
Learn how to easily offload C++ code to GPUs using OpenMP. Learn how to easily offload C++ code to GPUs using OpenMP.
- **11:00 - Offloading with FORTRAN Code with OpenMP**
    - Offloading using oneMKL
    - Automatic offloading using DO CONCURRENT
    - Offloading using OpenMP 5.2

- **11:45 - Hands-on labs on code optimisation**
	- Use OpenMP Offload directives to execute code on GPU.
	- Use OpenMP constructs to effectively manage data transfers to and from the device

- **12:30 - End of Day1 session**

### Day2: PYTHON AND ADVISOR (13th November 09:00-12:30 CET)

- **09:00  - Welcome**
- **09:05  - Intel Distribution of Python**

Explore the benefits of using Intel Distribution of Python and improve the performance where multithreading or multiprocessing are required or preferred. 
    - Demos and labs included.

- **10:00  - Break**

- **10:15 - Application profiling for CPU and mixed hardware with Intel Advisor**
    - Advisor main functionality (Vectorization and Roofline) starting with CPU
    - Estimate performance potential gains with Offload Advisor (CPU -> HW Accelerator)
    - Analyse heterogenous SYCL/OpenMP Workloads with Intel Advisor and Roofline analysis

- **11:00 - Hands-on labs on Advisor**
    - Run Offload Advisor using command line syntax.
    - Use performance models and analyze generated reports.
    - See how Offload Advisor identifies and ranks parallelization opportunities for offload.

- **12:30 - End of Day2 session**

### Day3: MPI AND VTUNE (27th November 09:00-12:30 CET)

- **09:00 - Welcome**
- **09:05 - Intel® MPI and oneCCL in Heterogeneous Environment**

Intel MPI is an Intel optimized version of runtime which provides MPI 4.0 Durnov standard compatible 
solution. As part of the presentation audience will learn about the latest Intel MPI GPU centric features
 and optimization techniques available in the latest releases plus information about other key features
 for CPU currently available in the product.

- **10:00 - Break**
- **10:15 - intel® MPI and oneCCL in Heterogeneous Environment – part2**

oneCCL is a runtime and API targeted on AI and integration with currently
available main AI frameworks. Here audience will learn about communication
stacks for popular AI frameworks and how oneCCL helps to get better scaling.

- **11:00 - Profiling and analysing code performance with VTune**
    - VTune main functionality (Hot spot analysis…) starting with CPU.
    - Profiling Tools Interfaces for GPU
    - Profile heterogenous SYCL/OpenMP Workloads with Intel VTune

- **11:35  - Hands-on labs on VTune**
    - Understand the basics of command line options in VTune Profiler to collect data and generate reports.
    - Profiling GPU applications using Intel® VTune™ Profiler on Intel® DevCloud 
    - Use performance models and analyze generated reports.

- **12:30   - End of Day3 session**

## Registration

The course is free of charge and is offered by intel and facilitated by the NRIS. 
**REGISTER [HERE](https://skjemaker.app.uib.no/view.php?id=17872594)**.

### The Speaker
- **Stephen Blair-Chappell** is an independent software consultant and is an
Intel-certified oneAPI instructor. He was formerly the Technical Director at
Bayncore where he led a team of consultants providing HPC and AI training on
Intel Architecture. For 18 years he was a Technical Consulting Engineer at Intel
helping their strategic customers in software optimization and code
modernization. He is the author of the book "Parallel Programming with Intel
Parallel Studio XE".

### Special presenter
- **Dmitry Durnov** (Intel MPI architect)

### Practical Information

This is an online course via zoom. Participants require access to a computer
(not provided by the course organisers) with internet connectivity to participate 
in the video meeting of the course (zoom). NRIS-Sigma2 users can use the cluster Saga for 
any hands-on exercises assuming that usernames are provided while registering for the event.

You can always contact us by sending an email to [support@nris.no](mailto:support@nris.no).

If you want to receive further information about training events, and other announcements
about IT resources and services for researchers, there are a couple of information channels
 you can subscribe to:
- users at University of Bergen and affiliated institutes: [register to hpcnews@uib.no](https://mailman.uib.no/listinfo/hpcnews)
- users of Sigma2 services: [subscribe to Sigma2's newsletter](https://sigma2.us13.list-manage.com/subscribe?u=4fd109ad79a5dca6dde7e4997&id=59b164c7b6)





