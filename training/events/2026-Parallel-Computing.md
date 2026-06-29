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

The [policy](https://documentation.sigma2.no/hpc_machines/olivia/software_stack.html#python-r-and-ana-conda) on Olivia is that you should not use pip install with Python in the same way you would on your laptop, because it will create a large number of files. On Olivia’s shared file system, this will place unnecessary strain on the system and lead to poor performance. To address this, this course will show how to perform pip install inside a container and how to use that container for parallel computing with Python on Olivia. 

By the end of this series, you’ll have a solid understanding of these concepts and how to apply them effectively.

These seminars are at a basic-to-intermediate level, and targeted towards participants at the preceding OnBoarding event. However, these seminars will also be open to others.

## Practical Information
- Basic command line/linux workflows are expected to be known. (elements of the [HPC Onboarding course given April 14-16.2026](https://documentation.sigma2.no/training/past/2026-04-hpc-on-boarding.html)). Also, a certain level of experience with Olivia is expected.

- The course is open to all and free of charge. However, signup is necessary to get access to course resources.

The course series happens 6 consecutive Wednesdays, starting from Wednesday Sept. 2nd until Oct. 7th 2026.

The Story of Python and how it took over the world: [Python: The Docmentary](https://www.youtube.com/watch?v=GfH4QL4VqJ0)

**Registration:** [Sign up here](https://docs.google.com/forms/d/e/1FAIpQLSfUzos-tFfbI2lWzpJF8U1s92cW0WYz9dbIH7EGsfTH0biWGw/viewform?usp=dialog)

<H3> Instructors: 

- Jim-Viktor Paulsen

<H3> Content:

- Episode 1: The basics and writing job scripts and Python codes with AI assistance.
- Episode 2: Code Optimization and Vector-Threading
- Episode 3: Multi-Threading and scaling tests
- Episode 4: Multi-Tasking and scaling tests
- Episode 5: Parallel Computing with Containers
- Episode 6: Hybrid Parallel Computing and threads-per-task scaling tests

<H3> Episode 1:

- Session 0: 09.00-09.15: Practical Information.
- Session 1: 09.15-10.00: Intoduction: Different approaches for teaching parallel computing. The levels of parallel computing. Olivia is a laboratory for numerical experiments.
- Session 2: 10.15-11.00: Executing the Python code: The Python code and matrix multiplication. Writing a Python function with AI assistance (AI-chat). The software system on Olivia. Slurm job scripts on Olivia.
- Session 3: 11.15-12.00: Flops and speedup: Computing the number of Flops. Speedup with MKL. Numba Speedup with JIT.
- Exercises: 13.00----: MKL and Numba: Using MKL (dgemm and matmul). Loop ordering with Numba.

<H3> Episode 2:

- Session 0: 09.00-09.15: Practical Information.
- Session 1: 09.15-10.00: JIT/AOT and Vector Threading: Using Numba (Just-In-Time compiling). Loop Ordering and Vector Threading. Cython Speedup with AOT (Ahead-Of-Time compiling). 
- Session 2: 10.15-11.00: Speedup with Fortran and C: Python Speedup with f2py compiling. Python ctype Speedup with Fortran. Python ctype Speedup with C.
- Session 3: 11.15-12.00: Loop Ordering and OpenMP: Loop Ordering with Cython, Fortran and C. OpenMP with Numba, Cython and MKL.
- Exercises: 13.00----: Cython, Fortran and C: Loop ordering with Cython, Fortran and C. OpenMP with MKL.

More details on the content will be provided later.

<H3> Event schedule: 

- 09:00: Start Presentation
- 12:00: Presentation Finished
- 13:00: Start Exercises 

- We will use Olivia for demos and hands-on sessions.

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





