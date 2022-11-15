(VASP)=

# VASP (Vienna Ab initio Simulation Package)

VASP is a software package for performing ab-initio quantum-mechanical calculation of a periodic arrangement of atoms using the projector-augmented wave method and a plane wave basis set. The package can perform density-functional-theory (DFT) calculations, or many-body-perturbation-theory (MBPT) like GW etc. Please consult the documentation to get a more detailed overview of its feature set.

## Online information from VASP developers targeted towards users

* [Homepage](https://www.vasp.at)
* [Documentation](https://www.vasp.at/wiki/index.php/The_VASP_Manual)
* [User forum](https://www.vasp.at/forum/)
* [Tutorials](https://www.vasp.at/wiki/index.php/Category:Tutorials)
* [Workshops](https://www.vasp.at/wiki/index.php/VASP_workshop)

## Installed on facilities

These are the installed VASP versions on our facilities.

| Version  | Fram | Saga | Betzy |
|----------|------|------|-------|
| 5.4.4pl2 | Yes  | Yes  | No    |
| 6.3.0    | Yes  | Yes  | Yes   |
| 6.3.2    | Yes  | Yes  | Yes   |

## Tests passed

This is the present test status for the installed modules for VASP.

| Version  | Fram | Saga | Betzy |
|----------|------|------|-------|
| 5.4.4pl2 | N/A  | N/A  | N/A   |
| 6.3.0    | Fast | Fast | Fast  |
| 6.3.2    | Fast | Fast | Fast  |

`Fast` or `Full` refer to the [VASP tests](https://www.vasp.at/wiki/index.php/Validation_tests).

## Supported versions

* 5.4.4 pl2 (not maintained, only for those that do not have VASP 6 licenses)
* 6.3.0
* 6.3.2 (recommended)

Note that at Betzy we only support VASP >= 6.3.

VASP 6 is the first VASP version that was supplied with an official test suite, which makes it easy to recommend users
to use this over previous VASP 5 as we are able to verify that the installation works across a wide set of systems, including corner cases.

## License and access policy

VASP is a commercial software package that requires a license for all who wants to use it. For a user to get access to VASP installed on the Sigma2 systems they must perform the following steps:

* The users group must have a valid VASP licence. To acquire a licence, please consult the [Get a license](https://www.vasp.at/sign_in/registration_form/) section at the VASP website.

* User identification is performed using email, so please make sure you supply the correct email. Send us a message at `contact@sigma2.no` where you request access to VASP and supply your email address that is associated with your VASP license. Remember that this might not be the address you are currently using to communicate. In order to figure this out, log in to your VASP portal and double check the email address listed there. Or ask your license holder to verify what email address you should use. We will then, using our maintainer access to the VASP portal verify that you hold a valid license to VASP 6 and/or VASP 5. If you have access we will add you to the `vasp6` and/or `vasp5` group. Members of these groups have access to the VASP modules containing the necessary software.

Notice that the VASP license is backwards compatible, meaning that if you are issues a VASP 6 license you also have access to VASP 5.

## Usage

You can check which VASP versions are installed by executing:

	$ module load VASPModules
	$ module avail VASP

And load a particular module by executing (case sensitive):

	$ module load VASP/6.3.2-intel-2021b-std-wannier90-libxc-hdf5-beef-<somehash>
	
where `<somehash>` is a hash that is computed based on the modifications done to the VASP source code, or extra libraries included. See below for more details. Most of the modules should be self explanatory for experienced VASP users and you might be able to just get the hash from inspecting the module names from `module avail VASP`. If not, please read below for addition description.

Please remember to use two `module load` commands. The first loads the location of all the VASP modules and the second command loads the actual VASP module. It is also possible, if you know what module you want to execute::

	$ module load VASPModules VASP/6.3.2-intel-2021b-std-wannier90-libxc-hdf5-beef-<somehash>

Users have to supply the necessary input files, including any `POTCAR` files needed. They can be downloaded from the VASP portal you get access to with a valid VASP license. Also, please note that the `POTCAR` files are protected by the license so do not share them with anyone that does not have a license, including the support team, unless explicitly notified to do so.

### Module naming scheme

There are now one module per VASP flavor (`std` - standard, `gam` - gamma only and `ncl` - non-collinear). Meaning when you now load a module, there is only one executable, `vasp`. For the example given above we load the module `VASP/6.3.2-intel-2021b-std-wannier90-libxc-hdf5-beef-<somehash>`. Here, `<somehash>` is a generated hash. i.e. `d7238be44ec2ed23315a16cc1549a1e3` based on the versions of libraries and extensions. This is done to avoid a very long module names containing the version numbers if each library included etc. In the future we might chose to go with something more compact, e.g. `VASP/6.3.2/<somehash>`. For now, we believe it is convenient to directly see which VASP version and what libraries are there etc., but not necessarily the Wannier90, BEEF, libxc version etc. The mapping between the hash and exact version numbers can be found here:

| VASP version | Hash                             | Wannier90 (tag) | VTST (svn) | BEEF (tag) | SOL (commit)                             | libxc (tag) | hdf5 (tag) | note                           |
|--------------|----------------------------------|-----------------|------------|------------|------------------------------------------|-------------|------------|--------------------------------|
| 5.4.4 pl2    | 86f69b2cbd5b5987c9dd0bf21a1b7e82 | 2.1.0           | -          | -          | -                                        | -           | -          | -                              |
| 6.3.0        | 5289f748cfc70eba91b5b6c81efedad4 | 3.1.0           | -          | -          | -                                        | 5.2.2       | 1.12.1     | HDF5 functionality not enabled |
| 6.3.2        | d7238be44ec2ed23315a16cc1549a1e3 | 3.1.0           | 74         | 0.1.1      | 0dc6b89b17e22b717cb270ecc4e1bbcfbb843603 | 5.2.2       | 1.12.1     | -                              |

Notice that the VASP version is not included when generating the hash, only the libraries and associated mods from the additions in the table above. Meaning, if we add a module with a new VASP version and do not need or do an update on any of the libraries, the hash should be the same. 

Please note that existing modules with names not containing a hash in the end is now considered legacy. You might use them to reproduce old results and we will keep them around until the system is end of life. Also try to migrate your workload to the most recent version.

We would like feedback from the community how we can improve the module naming scheme going forward. Please keep in mind that whatever we chose have to support reproducability, which means that we should be able to say, _give me VASP version `x.y.z`, with BEEF version `x1.y1.z1` and Wannier90 with `x2,y2,z2`_ etc.

The naming schemes of the modules are `VASP/version-toolchain-vasp_flavor-extra_libraries_and_functionality-adaptions_source_code`. Where:

- `version` determines the VASP version, e.g. 6.3.2.
- `toolchain` determines the toolchain used, typically which compilers, LAPACK, BLAS etc. routines have been used. This is based on the existing toolchains on the system. These can be inspected with `module show intel/2021b` for the particular system (e.g. `fram`). Typically, the `toolchain` is the vendor, e.g. `intel` followed by the version, e.g. `2021b`.
- `vasp_flavor` determines the VASP flavor, e.g. `std` for the standard flavor, `gam` for the gamma only (only works for one k-point) and `ncl` for the non-collinear flavor (makes it possible to let the spin go in any direction).
- `extra_libraries_and_functionality` determines if an additional package has been included, e.g. `wannier90` (support for maximally-localised Wannier functions and the [Wannier90](http://www.wannier.org/)), `beef` (to yield support for the [BEEF](https://github.com/vossjo/libbeef) functional and Bayesian error estimates), `vtst` (to yield support for additional transition state tools [VTST](http://theory.cm.utexas.edu/vtsttools/)), `sol` (to yield support for solvation models using [VASPsol](https://github.com/henniggroup/VASPsol)) and `libxc` (to yield support for the exchange and correlation library using [libxc](https://www.tddft.org/programs/libxc/)).
- `adaptions_source_code` determines if there has been adaptions to the source code, e.g. restrictions in the ionic motions. This can be the `nor_<direction>` which does not enable relaxation along the first, second and third lattice vector (with `<direction>` set as `x`, `y` and `z`, respectively). Or, `onlyr_<direction>`, with similar `<directions>`. Finally, `nor_angles` will not allow any relaxation of angles.

### Further notes about the additional packages and how the modules have been constructed

The VTST scripts are available if you load a module with `vtst` and can be found in `$EBROOTVASP/vtst` after loading the VASP module containing `vtst` in
its name.

The `bee` executable from the BEEF library can be found in `$EBROOTBEEF/bin/bee` after loading the VASP module containing `beef` in its name.

The `wannier90.x` and `postw90.x` executables of Wannier90 can be found in `$EBROOTWANNIER90/bin/wannier90.x` and `$EBROOTWANNIER90/bin/postw90.x` after
loading the VASP module with `wannier90` in its name.

### Parallel functionality and library support.

All VASP and Wannier90 binaries are compiled with Intel MPI (Fram and Saga) or OpenMPI (Betzy) support. 
No OpenMP is presently enabled, but we are working to extend the modules to also include that for VASP 6. 
This also includes GPU support for the methods in VASP that support this. Hybrid OpenMP+MPI support is still not widely tested and it is
rather complicated to reach an optimum with respect to tuning the distribution of load and very often the job ends up being slower than
for only the MPI enabled VASP version.

### Memory allocation for VASP

VASP is known to be potentially memory demanding. Quite often, you might experience to use less than the full number of cores on the node, but still all of the memory.

For relevant core-count, node-count, and amounts of memory, see the pages about {ref}`fram` and {ref}`saga`. There are two ways of increasing the memory pr. cpu over the standard node configuration:

- Increase the Slurm setting `mem-per-cpu`
- Utilize the nodes with more memory per cpu.

Remember you are accounted for the CPUs that would be reserved due to your demand for increased memory.

## Citation

When publishing results obtained with the software referred to, please do check your license agreement and the developers web page in order to find the correct citation(s). Also, remember to acknowledge Sigma2 for the computational resources.

## Getting additional help

We have an application liaison at our disposal for VASP which can help users with particular VASP issues and/or possibly also domain specific problems. In order to get in contact with the application liaison, please submit a support request as documented {ref}`here <support-line>`. If the ticket does not fall within regular support it will be forwarded.

However, before asking for help, please make sure you have gone through the tutorials and workshop material above that is relevant to you. Also, if your group would be interested in a dedicated VASP workshop, please reach out to [support@nris.no](mailto:support@nris.no) with a request and we will try to gauge general interest and arrange it, possibly in collaboration with the VASP developers if need be.
