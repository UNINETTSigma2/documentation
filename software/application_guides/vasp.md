(VASP)=

# VASP (Vienna Ab initio Simulation Package)

VASP is a software package for performing ab-initio quantum-mechanical calculation of a periodic arrangement of atoms using the projector-augmented wave method and a plane wave basis set. The package can perform density-functional-theory (DFT) calculations, or many-body-perturbation-theory (MBPT) like GW etc. Please consult the documentation to get a more detailed overview of its feature set.

**NOTE: we are currently in the process of simplifying the way in which VASP is accessed, see the section on FRAM. In the future we will aim to provide a smaller core of standard installs, but offer support for users interested in specialized modules.**

## Online information from VASP developers targeted towards users

* [Homepage](https://www.vasp.at)
* [Documentation](https://www.vasp.at/wiki/index.php/The_VASP_Manual)
* [User forum](https://www.vasp.at/forum/)
* [Tutorials](https://www.vasp.at/wiki/index.php/Category:Tutorials)
* [Workshops](https://www.vasp.at/wiki/index.php/VASP_workshop)

## Supported Versions

We aim to provide fully installed versions for the last two releases of VASP that require separate licenses. At time of writing this covers VASP 6.4.x and VASP 6.5.x licenses. VASP 5.x will be supported on a case-by-case basis for users lacking a 6+ license, but we note that this has proven more and more challenging to install and test on the latest hardware. We aim to have the latest patch version for every minor release please open a ticket if you require a different patch version.

| Version  | Fram | Saga | Betzy | Olivia |
| -------- | ---- | ---- | ----- | ------ |
| 5.4.4pl2 | Yes  | Yes  | no    | no     |
| 6.3.x    | no   | no   | no    | no     |
| 6.4.x    | Yes  | Yes  | Yes   | Yes    |
| 6.5.x    | Yes  | Yes  | Yes   | Yes    |


## License and access policy

VASP is a commercial software package that requires a license for all who wants to use it. For a user to get access to VASP installed on the Sigma2 systems they must perform the following steps:

* The users group must have a valid VASP licence. To acquire a licence, please consult the [Get a license](https://www.vasp.at/sign_in/registration_form/) section at the VASP website.

* User identification is performed using email, so please make sure you supply the correct email. Send us a message at `contact@sigma2.no` where you request access to VASP and supply your email address that is associated with your VASP license. Remember that this might not be the address you are currently using to communicate. In order to figure this out, log in to your VASP portal and double check the email address listed there. Or ask your license holder to verify what email address you should use. We will then, using our maintainer access to the VASP portal verify that you hold a valid license to VASP 6 and/or VASP 5. If you have access we will add you to the `vasp6` and/or `vasp5` group. Members of these groups have access to the VASP modules containing the necessary software.

Notice that the VASP license is backwards compatible, meaning that if you are issues a VASP 6 license you also have access to VASP 5.

## Parallelization 

Good performance with VASP is extremely dependant on good parallelization settings. For most users this means tuning the NPAR and KPAR tags in the INCAR file. While we cannot share detailed benchamrking we find that good results are roughly in the line with those given by vasp: https://www.vasp.at/wiki/index.php/NPAR https://www.vasp.at/wiki/index.php/KPAR

For many users one can use a stable NPAR across many systems and nodes and get good results. KPAR tends to be much more system dependant and we strongly recommend some benchmarking depending on the number of nodes and 'irreducible kpoints'. 

Users can check the first few lines of the OUTCAR on a new run, e.g.,

```
executed on             <System>
running  XXX mpi-ranks, on    X nodes
each image running on  XXX cores
distrk:  each k-point on  XXX cores,    X groups
distr:  one band on NCORE=  XX cores,   XX groups
```
The user should confirm the outputs here match their expectations. 

For quick benchmarking it is often useful to look at the speed of a single SCF step, e.g. by setting `NELM=1`. Often most information regarding benchmarking happens in the first step, e.g., if your settings are 2x faster for this first step it will likely remain so for the entire run. 
It can be helpful to look at both the final timing information at the end of the OUTCAR as well as the `LOOP` information for each SCF step (not to be confused with the `LOOP+` information at the end of each ionic step)

## Usage: Olivia
At time of writing we only support the CPU compiled version of VASP. A single-node GPU version is in progress, if you are interested in testing the GPU version please open a ticket. To use VASP please run

	$ module load NRIS/CPU
	$ module load VASP/6.4.3-intel-2024a

Or for `6.5.1` 
	$ module load VASP/6.5.1-intel-2024a

For those migrating from FRAM please be mindful to adjust your parallelization settings. A good starting point is using `--ntasks-per-node=250` with `NPAR = 25` to be a good starting point for many cases. We found that for many systems using `KPAR = <1/2 number nodes>` to work well, but *only* if your system contains more than one k-point. 


## Usage: Fram
**NOTE: Fram will be made obsolete in 2026, please see the Olivia running guide above


We are currently in the process of simplifying the way in which users can load VASP. On Fram it is now the following:

We now offer direct access to both `VASP5.4.4` and `VASP6.4.2`, eliminating the need to load VASPModules or VASPExtra beforehand. E.g.to run standard vasp:

	$ module load VASP/6.4.2-intel-2022b
	$ srun vasp_std

The available binaries include standard (`vasp_std`), noncollinear (`vasp_ncl`), and gamma point (`vasp_gam`).
Currently only basic VASP is offered, we plan to soon offer Wannier90 and HDF5 where applicable. 
* Specific modules, such as libxc, and adjustments like relaxation in the z-only direction can be provided upon special request.

## Usage: Saga and Betzy

You can check which VASP versions are installed by executing:

	$ module load VASPModules
	$ module avail VASP

And load a particular module by executing (case sensitive):

	$ module load VASP/6.4.1-intel-2021b-std-wannier90-libxc-hdf5-beef-<somehash>
	
where `<somehash>` is a hash that is computed based on the modifications done to the VASP source code, or extra libraries included. See below for more details. Most of the modules should be self explanatory for experienced VASP users and you might be able to just get the hash from inspecting the module names from `module avail VASP`. If not, please read below for addition description.

Please remember to use two `module load` commands. The first loads the location of all the VASP modules and the second command loads the actual VASP module. It is also possible, if you know what module you want to execute::

	$ module load VASPModules VASP/6.4.1-intel-2021b-std-wannier90-libxc-hdf5-beef-<somehash>

Users have to supply the necessary input files, including any `POTCAR` files needed. They can be downloaded from the VASP portal you get access to with a valid VASP license. Also, please note that the `POTCAR` files are protected by the license so do not share them with anyone that does not have a license, including the support team, unless explicitly notified to do so.

The currently available modules (as of April 4th 2024) are as follows:

VASP/6.4.2-intel-2022b-wHDF5-nohash-wWannier recommended by default 
VASP/6.4.2-intel-2022b-wHDF5-wvtst-wsol if you need the vtst or sol packages
VASP/5.4.4-intel-2022b if you need VASP5
 VASP/5.4.4-intel-2022b-wvtst if you need VASP5 and the vtst package 

### Module naming scheme

There are now one module per VASP flavor (`std` - standard, `gam` - gamma only and `ncl` - non-collinear). Meaning when you now load a module, there is only one executable, `vasp`. For the example given above we load the module `VASP/6.4.1-intel-2021b-std-wannier90-libxc-hdf5-beef-<somehash>`. Here, `<somehash>` is a generated hash. i.e. `d7238be44ec2ed23315a16cc1549a1e3` based on the versions of libraries and extensions. This is done to avoid a very long module names containing the version numbers if each library included etc. In the future we might chose to go with something more compact, e.g. `VASP/6.4.1/<somehash>`. For now, we believe it is convenient to directly see which VASP version and what libraries are there etc., but not necessarily the Wannier90, BEEF, libxc version etc. The mapping between the hash and exact version numbers can be found here:

| VASP version | Hash                             | Wannier90 (tag) | VTST (svn) | BEEF (tag) | SOL (commit)                             | libxc (tag) | hdf5 (tag) | note      |
|--------------|----------------------------------|-----------------|------------|------------|------------------------------------------|-------------|------------|-----------|
| 5.4.4 pl2    | 6dca52e0464347588557bc833ad7aef9 | 2.1.0           | -          | 0.1.1      | -                                        | -           | -          | Fram/Saga |
| 5.4.4 pl2    | a695b2f1ed198f379d85666aef427164 | 2.1.0           | -          | 0.1.1      | 0dc6b89b17e22b717cb270ecc4e1bbcfbb843603 | -           | -          | Fram/Saga |
| 5.4.4 pl2    | 3022db58e4b43f1ae0c4d395698b6f43 | 2.1.0           | 74         | 0.1.1      | -                                        | -           | -          | Fram      |
| 5.4.4 pl2    | 14c961080ada8703431c19f060ae7c61 | 2.1.0           | 74         | 0.1.1      | 0dc6b89b17e22b717cb270ecc4e1bbcfbb843603 | -           | -          | Saga      |
| 6.3.2/6.4.1  | d7238be44ec2ed23315a16cc1549a1e3 | 3.1.0           | -          | 0.1.1      | -                                        | 5.2.2       | 1.12.1     | Fram      |
| 6.3.2/6.4.1  | 036257e2962196f7eed8c289f961c450 | 3.1.0           | -          | 0.1.1      | 0dc6b89b17e22b717cb270ecc4e1bbcfbb843603 | 5.2.2       | 1.12.1     | Fram      |
| 6.3.2        | 097e6cb5a78f237dc588ba9c7877f23b | 3.1.0           | 74         | 0.1.1      | -                                        | 5.2.2       | 1.12.1     | Fram      |
| 6.3.2        | 80241dda52da1b720557debb2cb446fe | 3.1.0           | 74         | 0.1.1      | 0dc6b89b17e22b717cb270ecc4e1bbcfbb843603 | 5.2.2       | 1.12.1     | Fram      |
| 6.4.1        | 4ba477b1eae2cc6c43418b4f06b9150c | 3.1.0           | 127        | 0.1.1      | -                                        | 5.2.2       | 1.12.1     | Fram      |
| 6.4.1        | d6e4c52911032df6d5793df2ce4aaf19 | 3.1.0           | 127        | 0.1.1      | 0dc6b89b17e22b717cb270ecc4e1bbcfbb843603 | 5.2.2       | 1.12.1     | Fram      |
| 6.4.1        | 0a928426e459cf2aeab3d0bf8f441c74 | 3.1.0           | -          | 0.1.1      | -                                        | 5.2.2       | 1.14.1     | Saga      |
| 6.4.1        | 17dde9df298cd4ade20f0051444fd46a | 3.1.0           | -          | 0.1.1      | 0dc6b89b17e22b717cb270ecc4e1bbcfbb843603 | 5.2.2       | 1.14.1     | Saga      |
| 6.4.1        | 95b5f370a6e28c12e3ceb0addd48deb2 | 3.1.0           | 127        | 0.1.1      | -                                        | 5.2.2       | 1.14.1     | Saga      |
| 6.4.1        | 14b83fae861f986c955e51fe132d765d | 3.1.0           | 127        | 0.1.1      | 0dc6b89b17e22b717cb270ecc4e1bbcfbb843603 | 5.2.2       | 1.14.1     | Saga      |
| 6.4.1        | 7ec7231149f216987659e140f81251f9 | 3.1.0           | -          | 0.1.1      | -                                        | -           | -          | Betzy     |
| 6.4.1        | 1328e4b3fe680ad53aed047704ea8b90 | 3.1.0           | -          | 0.1.1      | 0dc6b89b17e22b717cb270ecc4e1bbcfbb843603 | -           | -          | Betzy     |
| 6.4.1        | c6ed3d20b98b13020d9bd2d8bce019c7 | 3.1.0           | 127        | 0.1.1      | -                                        | -           | -          | Betzy     |
| 6.4.1        | 2aed04d9a614793f1f8902b72d65bfb5 | 3.1.0           | 127        | 0.1.1      | 0dc6b89b17e22b717cb270ecc4e1bbcfbb843603 | -           | -          | Betzy     |

Notice that the VASP version is not included when generating the hash, only the libraries and associated mods from the additions in the table above. Meaning, if we add a module with a new VASP version and do not need or do an update on any of the libraries, the hash should be the same. 

Please note that existing modules with names not containing a hash in the end is now considered legacy. You might use them to reproduce old results and we will keep them around until the system is end of life. Also try to migrate your workload to the most recent version.

We would like feedback from the community how we can improve the module naming scheme going forward. Please keep in mind that whatever we chose have to support reproducability, which means that we should be able to say, _give me VASP version `x.y.z`, with BEEF version `x1.y1.z1` and Wannier90 with `x2,y2,z2`_ etc.

The naming schemes of the modules are `VASP/version-toolchain-vasp_flavor-extra_libraries_and_functionality-adaptions_source_code`. Where:

- `version` determines the VASP version, e.g. 6.4.1.
- `toolchain` determines the toolchain used, typically which compilers, LAPACK, BLAS etc. routines have been used. This is based on the existing toolchains on the system. These can be inspected with `module show intel/2021b` for the particular system (e.g. `fram`). Typically, the `toolchain` is the vendor, e.g. `intel` followed by the version, e.g. `2021b`. Notice that on Betzy, the name is GCC, even though we have used AOCC/AOCL to compile and link the VASP modules.
- `vasp_flavor` determines the VASP flavor, e.g. `std` for the standard flavor, `gam` for the gamma only (only works for one k-point) and `ncl` for the non-collinear flavor (makes it possible to let the spin go in any direction).
- `extra_libraries_and_functionality` determines if an additional package has been included, e.g. `wannier90` (support for maximally-localised Wannier functions and the [Wannier90](http://www.wannier.org/)), `beef` (to yield support for the [BEEF](https://github.com/vossjo/libbeef) functional and Bayesian error estimates), `vtst` (to yield support for additional transition state tools [VTST](https://theory.cm.utexas.edu/vtsttools/)), `sol` (to yield support for solvation models using [VASPsol](https://github.com/henniggroup/VASPsol)) and `libxc` (to yield support for the exchange and correlation library using [libxc](https://www.tddft.org/programs/libxc/)).
- `adaptions_source_code` determines if there has been adaptions to the source code, e.g. restrictions in the ionic motions. This can be the `nor_<direction>` which does not enable relaxation along the first, second and third lattice vector (with `<direction>` set as `x`, `y` and `z`, respectively). Or, `onlyr_<direction>`, with similar `<directions>`. Finally, `nor_angles` will not allow any relaxation of angles.

### Further notes about the additional packages and how the modules have been constructed

The VTST scripts are available if you load a module with `vtst` and can be found in `$EBROOTVASP/vtst` after loading the VASP module containing `vtst` in
its name.

The `bee` executable from the BEEF library can be found in `$EBROOTBEEF/bin/bee` after loading the VASP module containing `beef` in its name.

The `wannier90.x` and `postw90.x` executables of Wannier90 can be found in `$EBROOTWANNIER90/bin/wannier90.x` and `$EBROOTWANNIER90/bin/postw90.x` after
loading the VASP module with `wannier90` in its name.

## Parallel functionality and library support.

All VASP and Wannier90 binaries are compiled with Intel MPI (Fram and Saga) or OpenMPI (Betzy) support. 
No OpenMP is presently enabled, but we are working to extend the modules to also include that for VASP 6. 
This also includes GPU support for the methods in VASP that support this. Hybrid OpenMP+MPI support is still not widely tested and it is
rather complicated to reach an optimum with respect to tuning the distribution of load and very often the job ends up being slower than
for only the MPI enabled VASP version.

## Memory allocation for VASP

VASP is known to be potentially memory demanding. Quite often, you might experience to use less than the full number of cores on the node, but still all of the memory.

For relevant core-count, node-count, and amounts of memory, see the pages about {ref}`fram` and {ref}`saga`. There are two ways of increasing the memory pr. cpu over the standard node configuration:

- Increase the Slurm setting `mem-per-cpu`
- Utilize the nodes with more memory per cpu.

Remember you are accounted for the CPUs that would be reserved due to your demand for increased memory.

## Special note about Betzy and AMD systems

Notice that on Betzy, `libxc` and `hdf5` is not enabled due to issues with the AOCC/AOCL compilation setup. If you need this functionality,
use Saga and Fram for now. Also, on Saga and Fram VASP has been compiled with an Intel toolchain, but on Betzy, we have used AOCC and AOCL, which
gives the a similar setup for AMD. Due to the fact that there is no Easybuild setup for AOCL at the time of writing, the VASP modules on
Betzy seem to be using the GCC toolchain in their name. This is entirely due to the fact that we used the GCC toolchain as a skeleton to build
the necessary components to assemble what is in essence an AOCL toolchain.
When an AOCL Easybuild setup is officially supported, we will remedy this by recompiling. This should then give a more correct name for the module.

## Citation

When publishing results obtained with the software referred to, please do check your license agreement and the developers web page in order to find the correct citation(s). Also, remember to acknowledge Sigma2 for the computational resources.

## Getting additional help

We have an application liaison at our disposal for VASP which can help users with particular VASP issues and/or possibly also domain specific problems. In order to get in contact with the application liaison, please submit a support request as documented {ref}`here <support-line>`. If the ticket does not fall within regular support it will be forwarded.

However, before asking for help, please make sure you have gone through the tutorials and workshop material above that is relevant to you. Also, if your group would be interested in a dedicated VASP workshop, please reach out to [support@nris.no](mailto:support@nris.no) with a request and we will try to gauge general interest and arrange it, possibly in collaboration with the VASP developers if need be.
