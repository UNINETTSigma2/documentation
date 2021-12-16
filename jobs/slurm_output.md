# Understanding the job output file

When a job begins to run it will create a `slurm-<job id>.out`. All output,
both standard output and standard error, is sent to this file. In this way we
can monitor what our application is doing when it is running under Slurm. Slurm
will also ensure that no matter how many tasks or number of nodes everything
printed to standard out will be collected into this output file.

```{tip}
When contacting support with a job error always attach the Slurm script used to
submit the job as well as the Slurm output file. This helps us understand the
error and reduces misscommunication.
```

To illustrate the output we will use the following Slurm script:

```{eval-rst}
.. literalinclude:: slurm_output/run.slurm
   :language: bash
```

This script does nothing special and follows the best practises for a Slurm
script as described in {ref}`our introduction to Slurm scripts<job-scripts>`.
We ran the script on Saga which produced an output file called
`slurm-4677199.out`.

The anatomy of the output file is as follows (we will explain each part in
detail below):

1. Header created when the application launched
2. Application specific output
3. CPU and memory statistics
4. Disk read/write statistics
5. (Optional) GPU statistics

## Output header

Below, we have attached the first 15 lines of the output, from our sample
application above, which contain some shared information and some job specific
output. The first highlighted line shows which node(s) the job ran on and when
it started, this is always added to the output. Next we have highlighted the
beginning of the `module list` output. This output can be very useful to check
if the job does what you expect and allows you to see which software versions
you were using.

```{eval-rst}
.. literalinclude:: slurm_output/slurm-4677199.out
   :language: bash
   :lines: 1-15
   :emphasize-lines: 1, 8-15
```

After this follows the application output, which we will not show here as that
is application specific. However, know that using standard output to log what
your application is doing at different times can be a good way to better
understand how your application is running on the HPC machines.

## CPU statistics

```{note}
Slurm collects statistics every second, which means that for applications that
run for a short amount of time the CPU, memory, disk and GPU statistics can be
a bit missleading. Keep this in mind when developing your Slurm script and when
scaling up.
```

Once the application finishes running, either successfully or not, some
statistics about resource usage is outputted. The first of these are the CPU
and memory statistics. Using our example from above, here is the CPU and memory
statistics.

```{eval-rst}
.. literalinclude:: slurm_output/slurm-4677199.out
   :language: bash
   :lines: 82-94
```

In the above output we can see a few different things, but lets first explain
the meaning of `batch` and `extern`. When you submit a Slurm script the script
itself is counted as `batch` by Slurm. That means that any resources used by
your Slurm script is accounted for under this heading. If you run a command
directly within the Slurm script the time this command used will be accounted
for under `batch`. Looking back at our Slurm script above we can see that the
main application was `pytest`, we could alternatively have used `srun pytest`
which would create a new line in the output which would account for everything
under the `srun` call (multiple `srun` calls are accounted in separate line
items in the output). `extern` is everything outside these two usages and
should be fairly small, one example of an `extern` usage is SSH-ing to the node
where your code is running and inspecting, where the resources used during the
inspection would be accounted for under `extern`.

---

Below, we have highlighted the `batch` step in both the CPU and memory
statistics, this is most likely where you would find the most useful
information. In the CPU statistics we can see that we allocated 16 CPU cores
(`AllocCPUS`), for 1 task (`NTasks`), the script ran for 1 minute (`Elapsed`),
but the 16 CPU cores ran for around 2 minutes in total (`AveCPU`). Lastly, we
can see that the job ended successfully since the `ExitCode` is `0:0` (an exit
code of `0` means success on Linux).

```{eval-rst}
.. literalinclude:: slurm_output/slurm-4677199.out
   :language: bash
   :lines: 82-94
   :emphasize-lines: 5, 12
```

The other line we highlighted above is the memory footprint of our application.
The most pertinent information here is the maximum memory used (`MaxRSS`) and
the average memory used (`AveRSS`). Above, we can see that our application used
a maximum of `150344 KiB` (or around `147 MiB`). This information is very
important as that can be used to optimize your Slurm script to request less
memory (and thus be scheduled quicker).

## Disk statistics

The next information shows our applications disk usage. This information is
important because it can help you reduce unnecessary disk access which is not
the fastest for small files.

```{eval-rst}
.. literalinclude:: slurm_output/slurm-4677199.out
   :language: bash
   :lines: 96-101
   :emphasize-lines: 5
```

As we can see, our test application reads a bit of data, but writes very
little.

## GPU statistics

```{note}
GPU statistics is only outputted if your Slurm script
{ref}`requests one or more GPUs<gpu-intro>`.

Currently, only Saga supports automatic GPU metric collection. We are working
on enabling the same support on Betzy. One alternative that works on both Saga
and Betzy, but require manual intervention,
{ref}`is described in our CUDA introduction<cuda-c>`.
```

To give some indications on how your applications interacts with GPUs, the
following GPU statistics is collected during a run. The first thing to note in
the GPU output is that statistics is collected for all GPUs requested, and each
GPU is displayed separately. Below we have highlighted the information of most
interest.

```{eval-rst}
.. literalinclude:: slurm_output/slurm-4677199.out
   :language: bash
   :lines: 103-141
   :emphasize-lines: 14, 17-20, 34-36
```

`Max GPU Memory Used` is quite self explanatory, but can be important to check
(maybe your application can trade higher memory usage for more performance).`SM
Utilization` describes how well our application used the GPU compute resources.
If the maximum value here is low this could be an indication that your
application is not utilizing the requested GPUs. For our example application
not all steps utilized a GPU so we get a medium average utilization. The other
variable to pay extra attention to is the `Memory Utilization` which describes
how well our application utilized the GPU memory bandwidth. For longer running
application optimizing memory transfer is one of the first optimization that
should be undertaken for GPU optimization.

```{note}
The information under `PID` tends to not be as accurate as the other
information so take this with a grain of salt.
```
