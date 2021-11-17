---
orphan: true
---

(sycl_usm)=

# Unified Shared Memory with SYCL

This example demonstrates:

1. how to allocate USM pointers in SYCL
2. how to submit work tasks to a SYCL device queue
3. how to write a parallel kernel function in SYCL
4. how to perform `memcpy` operations locally in device memory
5. how to perform a reduction operation in a SYCL kernel function

In this tutorial we will SYCL-ify a somewhat more realistic example, which is taken from the
{ref}`OpenACC tutorial <openacc>`. The serial version of the Jacobi iteration program has here been
slightly modified for C++ (and in anticipation of what is to come):

```{eval-rst}
.. literalinclude:: hipsycl/jacobi_serial.cpp
   :language: cpp
```

```{eval-rst}
:download:`jacobi_serial.cpp <./hipsycl/jacobi_serial.cpp>`
```

## Compile and run reference serial code

We can compile and run the reference serial version of the code on Saga. First we load a recent version
of the GNU C++ compiler, and compile a `jacobi_serial` target with `-Ofast` optimization:

```console
[me@login-1.SAGA ~]$ module load GCC/10.2.0
[me@login-1.SAGA ~]$ g++ -Ofast -o jacobi_serial jacobi_serial.cpp
```

Hopefully no errors occurred on this step, and we are ready to run a reference benchmark on a compute node:

```console
[me@login-1.SAGA ~]$ srun --account=<my-account> --time=0:10:00 --ntasks=1 --cpus-per-task=1 --mem=1G time ./jacobi_serial
srun: job 3661704 queued and waiting for resources
srun: job 3661704 has been allocated resources
Iterations : 7214 | Error : 0.00999874
37.91user 0.00system 0:38.08elapsed 99%CPU (0avgtext+0avgdata 32880maxresident)k
3844inputs+0outputs (17major+1097minor)pagefaults 0swaps
```

The execution should take around 40 seconds on a single core (38.08s in this case, the `elapsed` value in
the output). We notice also the printed output from our program, which states that it ran a total of 7214
iterations before reaching an error below 0.01, so this is the number of times we enter the main `while` loop.

## Introducing SYCL and Unified Shared Memory

In contrast to the directives based approaches to GPU programming like OpenMP and OpenACC, which can often be achieved
by strategically placed compiler directives into the existing code, porting to SYCL might require a bit more changes to the
structure and algorithms of the program. SYCL supports fully asynchronous execution of tasks using C++ concepts like futures
and events, and provides two main approaches for data management: using Unified Shared Memory (USM) or Buffers.
USM uses familiar C/C++-like memory pointers in a _unified virtual address space_, which basically means that you
can use the same pointer address on the host and on the device. This approach will likely be more familiar to the
traditional C/C++ programmer, but it requires explicit management of all data dependences and synchronization, which can
be achieved by adding `wait` statements or by capturing an `event` from one task and passing it on as an explicit dependency
for other tasks. Buffers, on the other hand, can only be accessed through special `accessor` objects, which are used by
the runtime to automatically construct a dependecy graph for all the tasks, and thus make sure that they are executed in
the correct order.

In this tutorial we will limit ourselves to the USM approach, and we will for simplicity attach explicit `wait` statements
to all the tasks, which effectively deactivates any asynchronous execution.

### Step 1: Create a SYCL queue

Back to the Jacobi source code, we start by creating a `sycl::queue`. This object is "attached"
to a particular device and is used to submit tasks for execution, in general asynchronously
(out-of-order). As in the {ref}`Hello World <hipsycl-start>` SYCL example, we will print out the name
of the device to make sure that we pick up the correct hardware:

```{eval-rst}
.. literalinclude:: hipsycl/jacobi_shared.cpp
   :language: cpp
   :lines: 5-29
   :emphasize-lines: 4, 19-22
```

### Step 2: Allocate USM memory

USM pointers can be allocated in three ways: `malloc_host`, `malloc_device` or `malloc_shared`.
In this example we will use _shared_ memory pointers only, which are pointer addresses that can be
accessed by both the host and the device. Furthermore, the _physical_ location of such shared
data can actually change during program execution, and the runtime will move the data back
and forth as the access pattern changes from host to device and vice versa. This will make
sure that the data can become accessible from _local_ memory on both the host and the device and it
_allows_ for fast access on the device as long as the data is _allowed_ to reside on the
device local memory throughout the execution of a kernel function, i.e. no data accesses from
the host should occur in the mean time, which would result in costly data migration.

The changes we need to make to our example code in order to use shared memory is to replace
the stack allocated arrays (`arr` and `tmp`) with `sycl::malloc_shared`:

```{eval-rst}
.. literalinclude:: hipsycl/jacobi_shared.cpp
   :language: cpp
   :lines: 28-45
   :emphasize-lines: 5,14
```

We have added a `_s` suffix to the variable name just to indicate it's a shared pointer.
Note that we pass our `sycl::queue` (`Q`) to this memory allocation as it carries the
information of which device this memory should be shared (there could be several queues
with different devices). We see also that the shared data arrays can be filled and
`std::memcpy`'d in exactly the same way as before by the host, so there's no change to
how the host interacts with this data.

```{note}
Memory allocated with `sycl::malloc_host` will also be "accessible" from the device, but it
will always be fetched from host memory and passed to the device through a memory bus, which
is _always_ going to be _much_ slower than fetching directly from local memory on the device.
The fast alternative to shared memory is to use `sycl::malloc_host` and `sycl::malloc_device`
and then _manually_ transfer the data between the host and the device. This is a bit less
convenient, but it gives more fine-grained control to the programmer.
```

### Step 3: Implement the parallel kernel

We now come to the main work sharing construct in our example (beware, this is a mouthful):

```{eval-rst}
.. literalinclude:: hipsycl/jacobi_shared.cpp
   :language: cpp
   :lines: 46-80
   :emphasize-lines: 6-23
```

We will not discuss in detail everything that is going on here, please refer to standard SYCL
literature for more in-depth explanations, e.g. the free e-book on
[Data Parallel C++](https://www.apress.com/gp/book/9781484255735). The take-home message is that
we `submit` to the queue a kernel function which represents a single iteration of a `parallel_for`
loop for execution on the device. Some (probably unnecessary) logic is added to extract the
two array indices `i,j` from the single loop iteration index, but otherwise the body of the kernel
is the same as the nested loop we had in the serial version, except that we need to extract the
computation of the maximum error from this main loop. The reason for this is that the kernel
code will be executed in arbitrary order by many different threads on the device, and no single
thread will be able to compute the true maximum locally.

Since the memory was allocated as `malloc_shared` between the host and the device, the reduction
operation to find the maximum error, as well as the `std::memcpy` operation between `tmp_s` and
`arr_s`, can be performed by the host. Keep in mind, though, that this will require a _migration_
of the shared data back and forth between the device and the host at every iteration of the
`while` loop (more than 7000 iterations), and we will see the effect of this in the timings below.

A critical point in the code snippet above is the `wait()` statement on the tail of the `Q.submit()`
call. This will tell the host to wait for further execution until all the work in the parallel
kernel has been completed. This effectively deactivates asynchronous execution of the device tasks.

```{tip}
`Q.submit(...).wait();` is a concatenation of the slightly more expressive `Q.submit(...); Q.wait();`,
which emphasizes that it's the entire queue that is drained by the `wait`, not just the task loop
that was just submitted. This means that you can submit several independent tasks to the queue for
asynchronous execution, and then drain them all in `Q.wait()` at a later stage.
```


### Step 4: Free USM memory

Finally, as always when allocating raw pointers in C++, one has to manually free the memory:

```{eval-rst}
.. literalinclude:: hipsycl/jacobi_shared.cpp
   :language: cpp
   :lines: 82-89
   :emphasize-lines: 4-5
```

## Compiling for CPU

With the adjustments discussed above we end up with the following source code:

```{eval-rst}
.. literalinclude:: hipsycl/jacobi_shared.cpp
   :language: cpp
```

```{eval-rst}
:download:`jacobi_shared.cpp <./hipsycl/jacobi_shared.cpp>`
```

We can compile an `omp` target of this code on Saga using the `syclcc` compiler wrapper from
the `hipSYCL` module (feel free to ignore the warning):

```console
[me@login-1.SAGA ~]$ module load hipSYCL/0.9.1-gcccuda-2020b
[me@login-1.SAGA ~]$ syclcc --hipsycl-targets=omp -Ofast -o jacobi_shared_cpu jacobi_shared.cpp
clang-11: warning: Unknown CUDA version. cuda.h: CUDA_VERSION=11010. Assuming the latest supported version 10.1 [-Wunknown-cuda-version]
```

And we can run it on a single compute core (please ignore also the hipSYCL warning, which comes
when you run on compute nodes without GPU resources):

```console
[me@login-1.SAGA ~]$ srun --account=<my-account> --time=0:10:00 --ntasks=1 --cpus-per-task=1 --mem=1G time ./jacobi_shared_cpu
srun: job 3671849 queued and waiting for resources
srun: job 3671849 has been allocated resources
[hipSYCL Warning] backend_loader: Could not load backend plugin: /cluster/software/hipSYCL/0.9.1-gcccuda-2020b/bin/../lib/hipSYCL/librt-backend-cuda.so
[hipSYCL Warning] libcuda.so.1: cannot open shared object file: No such file or directory
Chosen device: hipSYCL OpenMP host device
Iterations : 7229 | Error : 0.00999993
65.29user 0.37system 1:05.89elapsed 99%CPU (0avgtext+0avgdata 34300maxresident)k
10337inputs+0outputs (47major+2099minor)pagefaults 0swaps
```

We see from the "Chosen device" output of our program that the `sycl::queue` was bound to the
"hipSYCL OpenMP host device", which means that it is using the host CPU as a "device".
So this took about a minute to run, which is some 50% _slower_ than the reference serial run
we did above. However, one of the benefits of SYCL is that it can use the available CPU threads
of the host as "device" for offloading. Let's try to run the same code on 20 CPU cores:

```console
[me@login-1.SAGA ~]$ srun --account=<my-account> --time=0:10:00 --ntasks=1 --cpus-per-task=20 --mem=1G time ./jacobi_shared_cpu
srun: job 3671925 queued and waiting for resources
srun: job 3671925 has been allocated resources
[hipSYCL Warning] backend_loader: Could not load backend plugin: /cluster/software/hipSYCL/0.9.1-gcccuda-2020b/bin/../lib/hipSYCL/librt-backend-cuda.so
[hipSYCL Warning] libcuda.so.1: cannot open shared object file: No such file or directory
Chosen device: hipSYCL OpenMP host device
Iterations : 7229 | Error : 0.00999993
594.42user 16.34system 0:30.84elapsed 1980%CPU (0avgtext+0avgdata 45092maxresident)k
10337inputs+0outputs (47major+2267minor)pagefaults 0swaps
```

Alright, we're down to ~30s, which is somewhat faster than the serial reference (still not overly
impressive given that we spend 20 times more resources). Let's see if we can do better on the GPU.


## Compiling for Nvidia GPUs

When compiling for the P100 Nvidia GPUs on Saga we simply have to change the `hipsycl-targets`
from `omp` to `cuda:sm_60`, and then submit a job with GPU resources:

```console
[me@login-1.SAGA ~]$ syclcc --hipsycl-targets=cuda:sm_60 -Ofast -o jacobi_shared_gpu jacobi_shared.cpp
[me@login-1.SAGA ~]$ srun --account=<my-account> --time=0:10:00 --ntasks=1 --gpus-per-task=1 --mem=1G --partition=accel time ./jacobi_shared_gpu
srun: job 3672238 queued and waiting for resources
srun: job 3672238 has been allocated resources
Chosen device: Tesla P100-PCIE-16GB
Iterations : 7230 | Error : 0.00999916
77.14user 54.72system 2:12.42elapsed 99%CPU (0avgtext+0avgdata 156600maxresident)k
11393inputs+0outputs (694130major+7440minor)pagefaults 0swaps
```

Good news first: the chosen device is now Tesla P100-PCIE-16GB, which is the name of the graphics
card on the Saga GPU nodes. Our application was actually able to pick up the correct device.
The bad news is of course the elapsed time of 2m12s, which is _significantly_ slower than both
the serial and OpenMP versions above. We already hinted at the reason for this poor performance,
so let's see if we can fix it.

## Optimizing for GPU performance


### Step 5: Move data between USM pointers on the device

In this example we have two `std::memcpy` performed by the host on the USM shared pointer. The first one
is a single operation before we enter the main `while` loop, while the other is performed at the end of
every loop iteration. Since this operation is performed by the host CPU, it will implicitly invoke a
data migration in case the data happens to be located in device memory when the function is called.
Since we are copying data _between_ two USM pointers, we can actually perform this `memcpy` directly
on the device, and thus avoid the costly data migration.

The `memcpy` that we do _before_ the main work loop in our example could be left unchanged.
This single function call should have no noticeable impact on the performance since the data is already
located on the host after the initialization. We will still submit also this `memcpy` operation to the
`sycl::queue` for execution on the device since it will serve as a preporatory step of migrating the
data to device memory _in advance_ of the upcoming kernel execution.

```{eval-rst}
.. literalinclude:: hipsycl/jacobi_memcpy.cpp
   :language: cpp
   :lines: 44-80
   :emphasize-lines: 2, 34
```

```{eval-rst}
:download:`jacobi_memcpy.cpp <./hipsycl/jacobi_memcpy.cpp>`
```

As we can see from the code snippet above, there are two changes to the `memcpy` function calls:
(1) `std::` is replaced by `Q.` and (2) we have put a `.wait()` on the tail of the function call.
(1) will offload the the work to be performed by the device rather than the host, while (2) will
hold back the host from further execution until the `Q` is empty (for now the queue holds only a
single `memcpy` task).

In contrast to the first `memcpy`, the one in the loop is critical for performance.
If this operation is performed as `std::memcpy` by the host, it will require an implicit data
migration from device to host (and back) _in every iteration_ of the `while` loop. Making this
a `Q.memcpy` instead will allow the copy to be executed locally in device memory without ever
involving the host.

```{tip}
The `Q.memcpy(...)` syntax is actually a shorthand for something a bit more cumbersome
`Q.submit([&](sycl::handler &h) { h.memcpy(...); })`, which is more in line with the syntax of the
kernel submission above.
```

### Step 6: Add reduction object to compute maximum error

There's still one more operation inside the `while` loop that needs to be considered, and that is
the computation of the maximum error in each iteration. This could not be straightforwardly included
in the kernel function, so we left it as a separate loop to be executed by the host after the kernel
has completed. However, just as for the `memcpy` that we discussed above, this will also imply a costly
data migration back to the host at every iteration. The way around this problem is to attach a
`sycl::reduction` operation to this error variable, which will allow us to include the maximum reduction
back into the main kernel function. The syntax to achieve this is somewhat involved:

```{eval-rst}
.. literalinclude:: hipsycl/jacobi_reduction.cpp
   :language: cpp
   :lines: 40-90
   :emphasize-lines: 3-4, 11-12, 17, 21-22, 31, 43
```

```{eval-rst}
:download:`jacobi_reduction.cpp <./hipsycl/jacobi_reduction.cpp>`
```

First of all, we need to allocate the variable that is collecting the error as a USM pointer so that it
is accessible on the device. We do this by `sycl::malloc_shared` of a single `float`. Then we need to wrap this USM
pointer into a `sycl::reduction` operation, and pass it as an extra argument to the `parallel_for` kernel.
Notice that the `max_err` object is passed into the kernel as the `max` argument to the lambda function.
Then we call the `combine()` function of this `sycl::reduction` object, which will perform the
`sycl::maximum<float>` operation on the data, and thus compute the _true_ maximum among all the entries
in a thread safe manner. Finally, since the `err_s` pointer is shared between device and host, the host
will still have access to the final error and can print it out in the end.

## Compiling and running optimized code

We now compile a `sm_60` target of the final version, and run on a GPU node:

```console
[me@login-1.SAGA ~]$ syclcc --hipsycl-targets=cuda:sm_60 -Ofast -o jacobi_reduction_gpu jacobi_reduction.cpp
[me@login-1.SAGA ~]$ srun --account=<my-account> --time=0:10:00 --ntasks=1 --gpus-per-task=1 --mem=1G --partition=accel time ./jacobi_reduction_gpu
srun: job 3808343 queued and waiting for resources
srun: job 3808343 has been allocated resources
Chosen device: Tesla P100-PCIE-16GB
Iterations : 7230 | Error : 0.00999916
2.03user 3.83system 0:06.49elapsed 90%CPU (0avgtext+0avgdata 156604maxresident)k
11457inputs+0outputs (1030major+6413minor)pagefaults 0swaps
```

We see that by making sure that the data _remains_ in device local memory throughout the execution of the
kernel, we have reduced the overall run time to about six seconds. Notice also that most of this time is
spent in `system` calls setting up the program, and only two seconds is spent by actually running the program.
This system overhead should (hopefully) remain at a few seconds also for larger application when the total runtime
is much longer.

## Summary

In this guide we have transitioned a serial C++ code into a small GPU application using the SYCL framework.
We have taken several steps from the initial serial implementation to the final accelerated version, using
concepts like Unified Shared Memory and a SYCL reduction operation. We have seen that the path to actual
_accelerated_ code is not necessarily straightforward, as several of the intermediate steps shows execution
times significantly _slower_ than the original serial code. The steps can be summarized as follows:

| Version                  | CPUs     | GPUs   | Run time      | Relative  |
|:------------------------:|:--------:|:------:|:-------------:|:---------:|
| `jacobi_serial`          | 1        | 0      |  38.1 sec     |   100%    |
| `jacobi_shared`          | 1        | 0      |  65.9 sec     |   173%    |
| `jacobi_shared`          | 20       | 0      |  30.8 sec     |    81%    |
| `jacobi_shared`          | 1        | 1      | 132.4 sec     |   348%    |
| `jacobi_reduction`       | 1        | 1      |   6.5 sec     |    17%    |

We have with this example shown in some detail how to compile and run a SYCL code on Saga, and how to make use of
the available GPU resources there. We have highlighted some basic SYCL _syntax_, but we have not gone into much
detail on what goes on under the hood, or how to write _good_ and _efficient_ SYCL code. This simple example only
scratches the surface of what's possible within the framework, and we encourage the reader to check out other more
complete resources, like the [Data Parallel C++](https://www.apress.com/gp/book/9781484255735)
e-book, before venturing into a real-world porting project using SYCL.
