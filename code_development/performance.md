# Performance Analysis and Tuning

Understanding application performance on modern HPC architectures
is a very complex task. There are a number of factors that can limit
performance: IO speed, CPU speed, memory latency and bandwidth, thread
binding and correct memory allocation on NUMA architectures,
communication cost in both threaded shared-memory applications, and
in MPI-based codes.

Sometimes the performance can be improved without recompiling the code,
e.g., by arranging the working threads or MPI ranks in a more
efficient way, or by using more / less CPU cores. In other cases it
might be required to perform an in-depth investigation into
the hardware performance counters and re-writing (parts of) the
code. Either way, identifying the bottlenecks and deciding on what
needs to be done can be made simpler by using specialized tools. Here
we describe some of the tools available on Fram.

* {ref}`arm-performance-reports`. An in-depth analysis of
  three synthetic benchmarks. We demonstrate some pitfalls of
  profiling, and show how one can use profiling to reason about the
  performance of real-world codes.

* [VTune Amplifier](performance/vtune.md). Performance analysis of the
  original, and the optimized software package ART - a 3D radiative
  transfer solver developed within the [SolarALMA
  project](https://www.mn.uio.no/astro/english/research/projects/solaralma/).

* [Other (Intel) tools](performance/intel_tuning.md). An overview of the tools and
  a quick guide to which type of tuning, and to what kind of programming
  model they are applicable.
