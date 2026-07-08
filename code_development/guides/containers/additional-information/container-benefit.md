(container-benefit)=
# Benefits of Using Container

## Scenario Where Container Is Useful

Consider, you might have an older image classification project that
relies on `NumPy v1.20`, but you want to start a new LLM fine-tuning 
project that requires `NumPy v2.0.` in your machine. Since, a standard 
operating system whether its your laptop or a basic cloud server, 
typically only one version of a library is allowed  to be installed 
globally.

1. Keeping the old version causes the new project to crash.
2. Upgrading the library fixes the new project but completely breaks 
the old code.

Hence, to control these version of these libraries within the runtime 
environment container plays a key role. 

There is also another technology called Virtual Machines(VMs). While 
both Containers and Virtual Machines are virtualization technologies 
used to isolate environments, they do so at completely different levels.

A Virtual Machine virtualizes a computer down to the hardware level. 
It allocates a chunk of physical CPU, RAM and storage and we need to 
install a full copy of Guest Operating System inside it. Because, it 
behaves like an entirely separate computer, it takes minutes to boot up
and consume massive system overhead just to idle. However, a Container 
skips the hardware emulation entirely. It hooks directly into the host 
machine´s existing Linux kernel for computing power, using a built-in 
Linux features (like namespaces and cgroups) to get its own private 
workspace partition. Since, they only carry the specific libraries and 
application file they need, they launch instantly and run with near-zero 
performance loss which is ideal for HPC cluster.

```{note}
While our cluster provides multiple software versions (for instance `NumPy`), 
they are often locked inside rigid compiler bundles. Containers bypass these 
toolchain restrictions, support custom niche dependencies, and ensure your 
workflow 
is perfectly portable to any other HPC system
```


## When to Use Containers on NRIS HPC System

```{note}
Please let us know if you find more reasons for using containers
```
 - If you have a software stack or a pipeline already setup somewhere else and
   you want to bring it as it is to one of the HPC systems
 - Containers give the users the flexibility to bring a full software stack to 
   the cluster which has already been set up, which can make software installations 
   and dependencies more reproducible and more portable across clusters.
 - The software you want to use is only available as a container image
 - You need to use system level installations, e.g. the procedure involved
   `apt-get install SOMETHING` or similar (`yum`, `rpm`, etc).
 - You have a old software that needs some older dependencies.
 - You need a specific version of a software to run another software, e.g. CUDA.

## When Not to Use Containers on NRIS HPC Systems

- If the software you are planing to to use already installed as a module(s), then
   better to use that module or collection of modules
- Windows containers. On NRIS HPC systems only containers that uses UNIX kernel would
   work
- If you do not know what the container is exactly for. i.e. found a command on
   the internet and just want to try it out
