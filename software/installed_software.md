# Installed software

When logging in to Saga and Betzy the `MODULEPATH` is set up to show all the default software module packages available. To get an overview of software installed issue command

	module overview

## Saga
For Saga the default software packages shown are for the Intel CPU partition with the NVIDIA P100 GPUs. If you want to use the AMD CPU partition with the NVIDIA A100 GPUs first issue command
 

	module --force swap StdEnv Zen2Env

## Olivia

Olivia is a quite different machine compared to the other NRIS HPC clusters. It is a HPE Cray computer, which provides the Cray Programming Environment (CPE) as means for advanced users to get the optimal performance when self-compiling code. In addition we have added NRIS software stacks compiled and optimized for Olivia’s nodes to ensure the best performance.

After logging in, no software stack is loaded by default. There are three types of stacks installed on Olivia:

* Cray programming environment (`CrayEnv`) providing compilers, libraries and tooling optimized for HPE Cray systems
* `NRIS` software stacks providing software and libraries installed by us
* `EESSI` software stacks providing software and libraries curated by the European EESSI project
Cray programming environment providing compilers, libraries and tooling optimized for HPE Cray systems

See the available stacks with command

	module available

You should see output similar to this:

```
------------------------------------------- /cluster/software/modules/Core -------------------------------------------
   CrayEnv    EESSI/2023.06 (S)    EESSI/2025.06 (S)    NRIS/CPU    NRIS/GPU    NRIS/Login    init-NRIS (S,L)

  Where:
   S:  Module is Sticky, requires --force to unload or purge
   L:  Module is loaded
```

Since Olivia have different node architectures there are three types of `NRIS` software stacks installed on Olivia:

- **`NRIS/CPU`**: Contains software packages and libraries optimized for the **CPU compute nodes**.
- **`NRIS/GPU`**: Currently includes libraries, compilers, and tools for building software for the Grace-Hopper 200 GPUs.
  In the future, this stack will also include AI frameworks such as PyTorch and TensorFlow.
- **`NRIS/Login`**: Includes tools for pre- and post-processing data. 
  **Note**: Do not use this stack for running workflows on the compute nodes.


Load one of the stack modules and see available software for that partition, e.g.:

	module load NRIS/CPU
	module available


```{note}
When building your software for production jobs make sure that you compile your code on a compute node. Allocate a compute node, e.g. running an interactive job (see [Interactive jobs](https://documentation.sigma2.no/jobs/interactive_jobs.html) and load one of the NRIS/CPU or NRIS/GPU stacks.
```

## EESSI
For software provided by EESSI see the [EESSI](https://documentation.sigma2.no/software/eessi.html) on NRIS clusters documentation.


## Which extensions are available?

Modules can contain extensions (**Perl modules**, **Python packages** and **R packages**). 
Extensions are marked with an `(E)` and can not be loaded directly. You will have to find and load the module that contains the extension.

For example, list all versions of the Python package `numpy` that are installed:

```
module spider numpy
```

Five different version of numpy is installed in this example:
```
     Versions:
        numpy/1.22.3 (E)
        numpy/1.24.2 (E)
        numpy/1.25.1 (E)
        numpy/1.26.2 (E)
        numpy/1.26.4 (E)

```

Now list the module that contains a specific version of `numpy`:

```
module spider numpy/1.26.4
```

In this case it was the `SciPy-bundle/2024.05-gfbf-2024a` module that provides the Python package `numpy` version `1.26.4`. In order to see what other extensions are included in the module, you can run:

```
module spider SciPy-bundle/2024.05-gfbf-2024a
```

We see that these Python packages are included in the module:
```

      Included extensions
      ===================
      beniget-0.4.1, Bottleneck-1.3.8, deap-1.4.1, gast-0.5.4, mpmath-1.3.0,
      numexpr-2.10.0, numpy-1.26.4, pandas-2.2.2, ply-3.11, pythran-0.16.1,
      scipy-1.13.1, tzdata-2024.1, versioneer-0.29
```

```{note}
If you do not want to see extensions in the output you can run the module command with the `--nx` option, like this: `module --nx available`
```














