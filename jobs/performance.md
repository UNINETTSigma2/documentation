

# How to check the performance and scaling


## Arm Performance Reports

[Arm Performance Reports](https://developer.arm.com/docs/101137/latest/contents)
is a performance evaluation tool which is simple to use, produces a
clear, single-file report, and it is used to obtain a
high-level overview of the performance characteristics.

It can report CPU time spent on various types of instructions (e.g.,
floating-point), communication time (MPI), multi-threading level and
thread synchronization overheads, memory bandwidth, and IO performance.
Such a report can help spotting certain bottlenecks in the
code and highlight potential optimization directions, but also suggest
simple changes in how the code should be executed to better utilize
the resources. Some typical examples of the suggestions are

> The CPU performance appears well-optimized for numerical
computation. The biggest gains may now come from running at larger
scales.

or

> Significant time is spent on memory accesses. Use a profiler
to identify time-consuming loops and check their cache performance.


### To profile a statically linked binary, you need to recompile

You can use Arm Performance Reports on dynamically linked binaries without recompilation.
However, you may have to recompile statically linked binaries
(for this please consult the [official documentation](https://developer.arm.com/docs/101137/2003)).


### A successful run will produce two files

A successful Arm Performance Reports run will produce two files, a HTML summary
and a text file summary, like in this example:

```
example_128p_4n_1t_2020-05-23_18-04.html
example_128p_4n_1t_2020-05-23_18-04.txt
```


### Using Arm Performance Reports on Fram and Saga

- [Fram](arm-perf/fram.md)
- [Saga](arm-perf/saga.md)


### Use cases and pitfalls

We demonstrate some pitfalls of profiling, and show how
one can use profiling to reason about the performance of real-world
codes.

- [STREAM benchmark](arm-perf/stream.md) (measures the memory bandwidth)
- [LINPACK benchmark](arm-perf/linpack.md) (measures the floating-point capabilities)
- [OSU benchmark](arm-perf/osu.md) (measures the interconnect performance)
- [Quantifying the profiling overhead](arm-perf/overhead.md)
