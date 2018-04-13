<h1>Performance Tools</h1>

<ul class='toc-indentation'>
<li><a href='#performance-reports'>ARM Performance Reports</a></li>
</ul>

<h2 id="performance-reports">ARM Performance Reports</h2>

<a href='https://www.arm.com/products/development-tools/hpc-tools/cross-platform/performance-reports'>ARM Performance Reports</a> (former Allinea Performance Reports) is a performance evaluation tool that as result produces a single HTML page with a characterisation of the problems that the evaluated program has. It divides the characterization into the categories CPU, MPI, I/O, and memory and it produces evaluations like: "The per-core performance is memory-bound" and "Little time is spent in vectorized instructions".

<h4>Load the module</h4>

Performance Reports needs to compile a library that is pre-loaded before you program is run. In order to compile this library, the Intel compiler library must be loaded as well, regardless of whether the program to be evaluated has been compiled with the Intel compiler:

    $ module load Arm-PerfReports/18.1.2 intel/2018a

Remember that you must also load these modules in job scripts, when you run your program.

<h4>Compile your program</h4>

When you compile your program then you should use optimizations flag like <code class="code">-O2</code>, <code class="code">-O3</code> and <code class="code">-xAVX</code>, because Performance Reports will then be able to tell you if your program manages to use vectorized instructions. You must also link with the special libraries, such as in:

    $ make-profiler-libraries

The output of this command will give you instructions on what to add to your link-line, in order to link your program with the Performance Reports libraries.

<h4>Analysing your program on login nodes</h4>

If your program only runs for a short while using few processes, then you can analyze it while running it on a login node:

    $ perf-report mpirun -np 4 rank

After your program has run, you can find the produced performance report in an HTML file with a name similar to interFoam_4p_2014-06-20_13-34.html. Copy this file to your local pc and view it there.

<h4>Analysing your program in batch jobs</h4>

In batch jobs, you must call your program link this:

    perf-report mpirun rank

After your program has run, you can find the produced performance report in an HTML file with a name similar to rank_4p_2014-06-20_13-34.html. Copy this file to your local pc and view it there.