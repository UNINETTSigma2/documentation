(eessi-access-on-nris)=

# Accessing EESSI on NRIS systems

On the HPC machines Betzy, Fram, Olivia and Saga, EESSI software stacks are
accessible on login nodes and all compute nodes. To
configure the environment, all that needs to be done is to run the single
command:

``` bash { code-block }
# to access the EESSI software stack
module load EESSI/2023.06
```

The module file will detect the CPU hardware of the machine and pick the best
software that was pre-built and is provided by EESSI. This works
seamlessly on systems which provide different CPU micro-architectures (e.g.,
Olivia and Saga).

**EESSI supports the following CPU micro-architectures:**

- **x86_64**: Intel Haswell (alternative for Broadwell on Fram and `hugemem` nodes on Saga),
Intel Cascade Lake, Intel Ice Lake, Intel Sapphire Rapids, Intel Skylake (Saga), AMD Rome/zen2 (Betzy and `a100` nodes on Saga), AMD
Milan/zen3, AMD Genoa/zen4 (compatible with `x86_64` nodes on Olivia) and a "generic" build which excludes optimisations for modern CPUs, thus should work on a wide spectrum of available hardware
- **aarch64**: ARM family, A64FX, Neoverse N1, Neoverse V1, NVIDIA/Grace(`aarch64` nodes on Olivia) and a "generic" build is available

**Read about [using software packages provided by EESSI](eessi-using).**
