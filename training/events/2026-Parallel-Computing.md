---
orphan: true
---

(2026-Parallel-Computing)=

# The Parallell Computing with Python on Olivia course series

Drawing on our experience with introducing the Olivia machine and associated services, *NRIS Training* is now offering a course series targeted directly on how to utilize the most powerful of the NRIS/Sigma2 HPC machines, Olivia, in the most efficient way. 

In this course series, we will guide you through practical steps and hands-on tasks to help you gain experience with parallel computing on Olivia using Python. Parallel computing can be divided into the following levels:
- Code Optimization – Techniques to speed up Python code on a single CPU core.
- Vector-Threading – Performing parallel computations within a single CPU core.
- Multi-Threading – Parallel computing across multiple CPU cores on a single node.
- Multi-Tasking – Executing parallel computations across multiple nodes (or within a single node).
- Hybrid Parallel Computing – Combining multi-threading and multi-tasking for maximum efficiency by leveraging all levels of parallelism.

By the end of this series, you’ll have a solid understanding of these concepts and how to apply them effectively.


These seminars are at a basic-to-intermediate level, and targeted towards participants at the preceding OnBoarding event. However, these seminars will also be open to others.

## Practical Information

**When:** The course series happens 6 consecutive Wednesdays, starting from Wednesday Sept. 2nd until Oct. 7th 2026.

**Where:** Online (Zoom). Zoom link will be sent to participants before the event start.

**Instructor:** Jim-Viktor Paulsen

**Registration:** [Sign up here](https://docs.google.com/forms/d/e/1FAIpQLSfUzos-tFfbI2lWzpJF8U1s92cW0WYz9dbIH7EGsfTH0biWGw/viewform?usp=dialog)

- Basic command line/linux workflows are expected to be known. (elements of the [HPC Onboarding course given April 14-16-2026](https://documentation.sigma2.no/training/past/2026-04-hpc-on-boarding.html)). Also, a certain level of experience with Olivia is expected.

- The course is open to all and free of charge. However, signup is necessary to get access to course resources.

```{note}
There is no closing date for the course registration, and you can sign up for the episodes you want to follow. However, please register at latest **one week** before the episode you are planning on attending.
```

<H3> Content:

- Episode 1, Sept. 2: The basics and writing job scripts and Python codes with AI assistance.
- Episode 2, Sept. 9: Code Optimization and Vector-Threading
- Episode 3: Sept. 16: Multi-Threading and scaling tests
- Episode 4: Sept. 23: Multi-Tasking and scaling tests
- Episode 5: Sept. 30: Parallel Computing with Containers
- Episode 6: Oct. 7: Hybrid Parallel Computing and threads-per-task scaling tests

<H3> Episodes schedule: 

- 09:00: Start Presentation
- 12:00: Presentation Finished
- 13:00: Start Exercises 

- We will use Olivia for demos and hands-on sessions
- Exercises are estimated at 1 hour
- Breaks are scheduled throughout the episodes 
<br>
<details>
<Summary>Detailed schedule </summary>
<br>

<H4> Episode 1, Sept. 2nd:

- **Session 0: 09.00-09.15: Practical Information**

- **Session 1: 09.15-10.00: Intoduction**
    - Different approaches for teaching parallel computing.
    - The levels of parallel computing.
    - Olivia is a laboratory for numerical experiments.

- **Session 2: 10.15-11.00: Executing the Python code**
    - The Python code and matrix multiplication.
    - Writing a Python function with AI assistance (AI-chat).
    - The software system on Olivia.
    - Slurm job scripts on Olivia.

- **Session 3: 11.15-12.00: Flops and speedup**
    - Computing the number of Flops.
    - Speedup with MKL (dgemm and matmul)
    - Numba Speedup with JIT

- **Exercises: 13.00----: MKL and Numba**
    - Using MKL (dgemm and matmul). Loop ordering with Numba

<H3> Episode 2, Sept. 9th:

- **Session 0: 09.00-09.15: Practical Information**

- **Session 1: 09.15-10.00: JIT/AOT and Vector Threading**
    - Using Numba (Just-In-Time compiling).
    - Loop Ordering and Vector Threading.
    - Cython Speedup with AOT (Ahead-Of-Time compiling).

- **Session 2: 10.15-11.00: Speedup with Fortran and C**
    - Speedup with f2py compiling.
    - Speedup with ctypes Fortran.
    - Speedup with ctypes C.

- **Session 3: 11.15-12.00: Loop Ordering**
    - Loop Ordering with Cython
    - Loop Ordering with f2py Fortran
    - Loop Ordering with ctypes Fortran

- **Exercises: 13.00----: Cython, Fortran and C**
    - Loop ordering with Cython, Fortran and C.

<H3> Episode 3, Sept. 16th:

- **Session 0: 09.00-09.15: Practical Information**

- **Session 1: 09.15-10.00: Numba Scaling**
    - OpenMP directives and affinity.
    - Scaling setup scripts (OpenMP).
    - Matplotlib scripts (OpenMP).
    - OpenMP scaling with Numba.

- **Session 2: 10.15-11.00: Cython and Fortran Scaling**
    - OpenMP scaling with Cython.
    - Thread safety and race conditions.
    - OpenMP scaling with Fortran and MKL.

- **Session 3: 11.15-12.00: NumPy matmul Scaling**
    - OpenMP scaling with NumPy matmul.
    - Comparing OpenMP Scaling: Numba, Cython, Fortran and NumPy

- **Exercises: 13.00----: Multi-Threading (OpenMP) Scaling**
    - Scaling with Numba, Cython, Fortran and NumPy

<H3> Episode 4, Sept. 23rd:

- **Session 0: 09.00-09.15: Practical Information**

- **Session 1: 09.15-10.00: Numba Scaling**
    - Parallel strategy and C/F-style.
    - Scaling setup scripts (MPI).
    - Matplotlib scripts (MPI).
    - MPI scaling with Numba.

- **Session 2: 10.15-11.00: Cython and Fortran Scaling**
    - MPI scaling with Cython.
    - MPI scaling with Fortran and MKL.

- **Session 3: 11.15-12.00: NumPy matmul Scaling**
    - MPI scaling with NumPy matmul.
    - Comparing MPI Scaling: Numba, Cython, Fortran and NumPy

- **Exercises: 13.00----: Multi-Tasking (MPI) Scaling**
    - Scaling with Numba, Cython, Fortran and NumPy

<H3> Episode 5, Sept. 30th:

- **Session 0: 09.00-09.15: Practical Information**

- **Session 1: 09.15-10.00: OpenMP Containers**
    - Apptainer (Container) with pip install.
    - OpenMP Containers
    - Host-binding to Olivia (the Host) software.

- **Session 2: 10.15-11.00: MPI Containers**
    - MPI Containers
    - Host-binding to Olivia MPI software

- **Session 3: 11.15-12.00: Container scaling**
    - OpenMP scaling
    - MPI scaling

- **Exercises: 13.00----: Container Scaling**
    - Scaling with Containers

<H3> Episode 6, Oct. 7th:

- **Session 0: 09.00-09.15: Practical Information**

- **Session 1: 09.15-10.00: Hybrid Parallel Computing**
    - Tasks and memory usage.
    - Threads per task scaling.

- **Session 2: 10.15-11.00: Hybrid Containers**
    - Hybrid Containers.

- **Session 3: 11.15-12.00: Parallel Computing with SAR**
    - System Activity Reporter (SAR)

- **Exercises: 13.00----: Threads per task scaling**
    - Threads per task scaling 


</details>
<br>

The [policy](https://documentation.sigma2.no/hpc_machines/olivia/software_stack.html#python-r-and-ana-conda) on Olivia is that you should not use `pip install` with Python in the same way you would on your laptop, because it will create a large number of files. On Olivia’s shared file system, this will place unnecessary strain on the system and lead to poor performance. To address this, this course will show how to perform `pip install` inside a container and how to use that container for parallel computing with Python on Olivia. 

**See also:** the Story of Python and how it took over the world: [Python: The Documentary](https://www.youtube.com/watch?v=GfH4QL4VqJ0)

### Coordinator

- Eirik Skjerve

### Code of Conduct

All course participants are expected to show respect and courtesy to
others. We follow the [carpentry code of
conduct](https://docs.carpentries.org/topic_folders/policies/code-of-conduct.html#code-of-conduct-detailed-view).
If you believe someone is violating the Code of Conduct, we ask that you report
it to [the training team](mailto:training@nris.no).

### Contact us

You can always {ref}`contact our support team <support-line>`.





