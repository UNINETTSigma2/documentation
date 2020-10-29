# Intel tuning tools

Intel provide a set of tuning tools that can be quite useful. The
following sections will give an overview of the tools and provide a
quick guide of the scope of the different tools and for which type of
tuning and what kind of programming model that are applicable for,
vectorization, threading, MPI.

Below is a list of tools that can be used to analyses the
application, starting from text analysis of the source code to massive
parallel MPI runs.

* Intel compiler, analysis of source code.
* Intel XE-Advisor, analysis of Vectorization.
* Intel XE-Inspector, analysis of threads and memory.
* Intel VTune-Amplifier, analysis and profiling of complete program performance.
* Intel MPI, profile the MPI calls.
* Intel Trace Analyzer, analysis of MPI communication.


The PRACE Best Practice Guide for the Intel processor Knights Landing:
[http://www.prace-ri.eu/best-practice-guide-knights-landing-january-2017/#tuning.section]
contain a nice review of the Intel tools and how to use them. Chapter 7 is about
tuning. Only chapters 1 and 2 are specific to the Knights Landing processor.

A short overview is given here with links to the relevant tools.
The Best Practice Guide provide an overview of the tools with a focus on
sampling. The actual usage of the different tools is only covered in depth
by the Intel documentation. Even with the documentation it can be hard to
start using the tools in an effective way. Attending Intel training is advised.


## Intel compiler - optimization report

Compiler flags provide a mean of controlling the optimization done by
the compiler. There are a rich set of compiler flags and directives
that will guide the compiler's optimization process. The details of
all these switches and flags can be found in the documentation, in
this guide we'll provide a set of flags that normally gives acceptable
performance. It must be said that the defaults are set to request a
quite high level of optimization, and the default might not always be
the optimal set. Not all the aggressive optimizations are numerically
accurate, computer evaluation of an expression is as we all know quite
different from paper and pencil evaluation.

Please consult the BPG guide section about the Intel Compiler at:
[http://www.prace-ri.eu/best-practice-guide-knights-landing-january-2017/#id-1.8.3.5]


To use the Intel compiler one of the Intel compiler modules must be loaded,
an example is :
```
module load intel/2018b
```


## Intel MPI library

The MPI Perf Snapshot is a built in lightweight tool that will provide
some helpful information with little effort from the user. Link your
program with -profile=vt and simply issue the -mps flag to the mpirun
command (some environment need to be set up first, but this is quite
simple).

```
mpiifort -o ./photo_tr.x -profile=vt *.o
mpirun -mps -np 16 ./photo_tr.x
```

More information at the Best Practice Guide  :
[http://www.prace-ri.eu/best-practice-guide-knights-landing-january-2017/#id-1.8.3.6]



## Intel XE-Advisor

Vectorization Advisor is an analysis tool that lets you identify if
loops utilize modern SIMD instructions or not, what prevents
vectorization, what is performance efficiency and how to increase
it. Vectorization Advisor shows compiler optimization reports in
user-friendly way, and extends them with multiple other metrics, like
loop trip counts, CPU time, memory access patterns and recommendations
for optimization.

More information at the Best Practice Guide  :
[http://www.prace-ri.eu/best-practice-guide-knights-landing-january-2017/#XE-Advisor]

To have access to the XE-Advisor load the designated module, an example is:
```
module load Advisor/2018_update3
```
Then launch the advisor in GUI or text version.
```
 advixe-gui &
```
Using the GUI X11 version over long distances might be somewhat slow.


## Intel XE-Inspector

Intel Inspector is a dynamic memory and threading error checking tool
for users developing serial and multithreaded applications.  The
tuning tool XE-advisor as tool tailored for threaded shared memory
applications (it can also collect performance data for hybrid MPI
jobs using a command line interface). It provide analysis of memory
and threading that might prove useful for tuning of any application.

More information at the Best Practice Guide  :
[http://www.prace-ri.eu/best-practice-guide-knights-landing-january-2017/#id-1.8.3.8]

To use the XE-Inspector us the Inspector module, like :
```
module load Inspector/2018_update3
```
To launch the GUI:
```
inspxe-gui &
```

## Intel VTune Amplifier

Intel VTune Amplifier provides a rich set of performance insight into
CPU performance, threading performance and scaleability, bandwidth,
caching and much more. Originally a tool for processor development,
hence the strong focus on hardware counters.

Analysis is fast and easy because VTune Amplifier understands common
threading models and presents information at a higher level that is
easier to interpret. Use its powerful analysis to sort, filter and
visualize results on the timeline and on your source. However, the
sheer amount of information is sometimes overwhelming and in some
cases intimidating. To really start using VTune Amplifier some basic
training is suggested.

More information at the Best Practice Guide  :
[http://www.prace-ri.eu/best-practice-guide-knights-landing-january-2017/#id-1.8.3.9]

To get access to the Vtune Amplifier load the corresponding module, like:

```
ml load VTune/2018_update3
```

For a run-through example see [a VTune case study](vtune.md).


## Intel Trace Analyzer
Intel Trace Analyzer and Collector is a graphical tool for
understanding MPI application behavior, quickly finding bottlenecks,
improving correctness, and achieving high performance for parallel
cluster applications based on Intel architecture. Improve weak and
strong scaling for small and large applications with Intel Trace
Analyzer and Collector.  The collector tool is closely linked to the
MPI library and the profiler library must be compiled and linked with
the application. There is an option to run using a preloaded library,
but the optimal way is to link in the collector libraries at build
time.

More information at the Best Practice Guide  :
[http://www.prace-ri.eu/best-practice-guide-knights-landing-january-2017/#id-1.8.3.10]

 To use Intel Trace Analyzer load the corresponding module, e.g.

 ```
 ml load itac/2019.4.036
 ```

