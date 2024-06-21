(eessi-access-on-nris)=

# Accessing EESSI on NRIS systems

On the HPC machines Betzy, Fram and Saga, both the EESSI software stacks is
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
Saga).

[eX3](https://www.ex3.simula.no) and AWS.

**EESSI supports the following CPU micro-architectures:**

- **x86_64**: Intel Haswell (alternative for Broadwell on Fram and `hugemem` nodes on Saga),
Intel Skylake (Saga), AMD Rome/zen2 (Betzy and `a100` nodes on Saga), AMD
Milan/zen3 and a "generic" build which excludes optimisations for modern CPUs, thus should
work on a wide spectrum of available hardware
- **aarch64**: ARM family, Neoverse N1, Neoverse V1 and a "generic" build is available

All software packages available in EESSI have been automatically built on AWS.

**Read about [using software packages provided by EESSI](eessi-using).**
