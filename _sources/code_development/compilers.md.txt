# Compilers

## Compiling MPI Applications

### Open MPI

The Open MPI compiler wrapper scripts listed in the table below add in all relevant compiler and link flags, and then invoke the underlying compiler, i.e. the compiler that the Open MPI installation was built with.

| Language     | Wrapper script     | Default compiler	| Environment variable |
| :------------- | :-------------: |:-------------: |:-------------: |
| C |`mpiicc`  | `gcc` | `OMPI_CC` |
| C++ |	`mpicxx, mpic++, mpiCC`  | `g++` | `OMPI_CXX` |
| Fortran | `mpifort`  | `gfortran` | `OMPI_FC` |

It is possible to change the underlying compiler that is invoked when calling the compiler wrappers using the environment variables listed in the table. Use the option `-showme` to see the underlying compiler, the compile and link flags, and the libraries that are linked.

### Intel MPI

The following table shows available Intel MPI compiler commands, the underlying Intel and GNU compilers, and ways to override underlying compilers with environment variables or command line options.
| Language     | Wrapper script     | Default compiler	| Environment variable | Command line |
| :------------- | :-------------: |:-------------: |:-------------: |:-------------: |
| C |`mpiicc`  | `icc` [^1]| `I_MPI_CC` | `-cc=`<compiler> |
| C |`mpiicx` [^2]  | `icx` [^3]| `I_MPI_CC` | `-cc=`<compiler> |
| C      |`mpicc`  | `gcc` | `I_MPI_CC` | `-cc=`<compiler> |
| C++ |	`mpiicpc`  | `icpc` [^1] | `I_MPI_CXX` | `-cxx=`<compiler> |
| C++ |	`mpiicpx` [^2]  | `icpx` [^3] | `I_MPI_CXX` | `-cxx=`<compiler> |
| C++     |	`mpicxx`  | `g++`| `I_MPI_CXX` | `-cxx=`<compiler> |
| Fortran | `mpiifort`  | `ifort` [^1] | `I_MPI_FC` | `-fc=`<compiler> |
| Fortran | `mpiifx` [^2]  | `ifx` [^3] | `I_MPI_FC` | `-fc=`<compiler> |
| Fortran     |	`mpif90`  | `gfortran` | `I_MPI_FC` | `-fc=`<compiler> |

Specify option `-show` with one of the compiler wrapper scripts to see the underlying compiler together with compiler options, link flags and libraries.

**Example:** use the available MPI C wrapper command before the Intel oneAPI 2023.2 release but with the LLVM based compiler
```
$ module load intel/2022a
$ mpiicc -cc=icx mpi_hello_world.c
```

Notice, the Intel Compiler Classic drivers commands `icc`and `icpc` have been removed since the Intel oneAPI 2024.0 release (i.e. 2024 toolchains for NRIS clusters). Use the LLVM-based Intel Compiler drivers `icx` and `icpx` instead, for more information see the [Porting Guide for ICC Users to DPCPP or ICX](https://www.intel.com/content/www/us/en/developer/articles/guide/porting-guide-for-icc-users-to-dpcpp-or-icx.html).
The Classic `ifort` command will be discontinued in the oneAPI 2025 release. Use LLVM-based Intel Compiler driver `ifx` instead, for more information see the [Porting Guide for ifort Users to ifx.](https://www.intel.com/content/www/us/en/developer/articles/guide/porting-guide-for-ifort-to-ifx.html).

See also {ref}`running-mpi-applications`.

[^1]: Intel Compiler Classic driver commands, available before the Intel oneAPI 2024.0 release (`icc/icpc`), and before the Intel oneAPI 2025 release (`ifort`).
[^2]: Intel LLVM based compiler based wrappers available since the Intel oneAPI 2023.2 release (i.e. `intel/2023b` toolchain on NRIS clusters)
[^3]: LLVM-based backend Intel Compiler drivers available since 2022
