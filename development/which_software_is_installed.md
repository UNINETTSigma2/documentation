# Installed Software

The `module` command is for loading or listing available modules.

```
module [options] [module name]
```

To view the full list of options, enter `man module` in the command line. Here is a brief list of common module options:

* _avail_ - list the available modules
* _list_ - list the currently loaded modules
* _load  <module name>_ - load the module called modulename
* _unload  <module name>_ - unload the module called module name
* _show <module name>_  - display dependencies and environment variables
* _spider <module name>_  - print module description

For example, to display all available modules and load the Intel toolchain on Fram, enter:

```
module avail
module load intel/2017a
```

Modules may load other modules as part of its dependency. For example, loading the Intel version loads related modules to satisfy the module's dependency.
The `module show` command displays the other modules loaded by a module. The `module spider` command displays the module's description.

## Applications

**Application**    | **GCC 6.3.0** | **Intel 17.0.1** | **Core**
---|---|---|---
**CESM**           |               | 1.2.2.1          |
**CP2K**           |               | 4.1              |
**ESMF**           |               | 6.3.0rp1         |
**Fluent**         |               |                  | 18.2
**GROMACS**        | 2016.3        | 2016.3           |
**LAMMPS**         | 11Aug17       |                  |
**MATLAB**         |               |                  | 2017a
**NAMD**           | 2.12, 2017-11-06 | 2.12, 2017-11-06 |
**NWChem**         |               | 6.6              |
**OpenFOAM**       |               | 4.1, 5.0         |
**OpenFOAM-Extend**|               | 4.0              |
**QuantumESPRESSO**|               | 6.1              |
**VASP**           |               | 5.4.4            |
** WPS/WRF**       |               | 3.9.1            |

## Libraries

**Library**        | **GCC 6.3.0** | **Intel 17.0.1**
---|---|---
**Armadillo**      | 8.300.0       | 8.300.0
**arpack-ng**      | 3.5.0         | 3.5.0
**Boost**          | 1.63.0        | 1.63.0
**FFTW**           | 3.3.6         | 3.3.6
**GDAL**           | 2.2.0         | 2.2.0
**grib_api**       | 1.24.0        | 1.24.0
**GSL**            | 2.3           | 2.3
**HDF5-1.8**       | 1.8.18        | 1.8.18
**HDF5-1.10**      | 1.10.1        | 1.10.1
**Hypre**          | 2.11.2        | 2.11.2
**METIS**          | 5.1.0         | 5.1.0
**netCDF**         | 4.4.1.1       | 4.4.1.1
**netCDF-Fortran** | 4.4.4         | 4.4.4
**netCDF-C++4**    | 4.3.0         | 4.3.0
**OpenBLAS**       | 0.2.19        | 
**ParMETIS**       | 4.0.3         | 4.0.3
**PETSc**          | 3.8.0         | 3.8.0
**PnetCDF**        | 1.8.1         | 1.8.1
**ScaLAPACK**      | 2.0.2         |
**UDUNITS**        | 2.2.24        | 2.2.24


## Tools
**Tool**           | **GCC 6.3.0** | **Intel 17.0.1**
---|---|---
**CDO**            |               | 1.8.2
**Ferret**         | 7.2           |
**GEOS**           | 3.6.1         | 3.6.1
**git**            | 2.14.2        | 2.14.2
**Mercurial**      |               | 4.3.3
**NCL**            |               | 6.4.0
**NCO**            |               | 4.6.6
**ncview**         |               | 2.1.7
**OpenBabel**      | 2.4.1         |
**ParaView**       |               | 5.2.0
**Perl**           | 5.24.1        | 5.24.1
**PROJ**           | 4.9.3         | 4.9.3
**Python2**        | 2.7.13        | 2.7.13
**Python3**        | 3.6.1         | 3.6.1
**R**              |               | 3.4.0
**Valgrind**       | 3.13.0        | 3.13.0
