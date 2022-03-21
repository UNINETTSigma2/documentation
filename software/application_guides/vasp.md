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

* Saga
* Fram
* Betzy

## Supported versions

* 5.4.4 pl2
* 6.3.0

## License and access policy

VASP is a commercial software package that requires a license for all who wants to use it. For a user to get access to VASP installed on the Sigma2 systems they must perform the following steps:

* The users research group must have a valid VASP licence. To acquire a licence, please consult the [How can I purchase a VASP license?](https://www.vasp.at/faqs/).

* User identification is performed using email, so please make sure you supply the correct email. Send us a message at `support@nris.no` where you request access to VASP and supply your email address that is associated with your VASP license. Remember that this might not be the address you are currently using to communicate. In order to figure this out, log in to your VASP portal and double check the email address listed there. Or ask your license holder to verify what email address you should use. We will then, using our maintainer access to the VASP portal verify that you hold a valid license to VASP 6 and/or VASP 5. If you have access we will add you to the `vasp6` and/or `vasp5` group. Members of these groups have access to the VASP modules containing the necessary software.

Notice that the VASP license is backwards compatible, meaning that if you are issues a VASP 6 license you also have access to VASP 5.

## Usage

You can check which VASP versions are installed by executing:

	$ module load VASPModules
	$ module avail VASP

And load a particular module by executing (case sensitive):

	$ module load VASP/5.4.4-intel-2019a-std-wannier90-somehash

Please remember to use two `module load` commands. The first loads the location of all the VASP modules and the second command loads the actual VASP module. It is also possible, if you know what module you want to execute::

	$ module load VASPModules VASP/5.4.4-intel-2019a-std-wannier90-somehash


Users have to supply the necessary input files, including any `POTCAR` files needed. They can be downloaded from the VASP portal you get access to with a valid VASP license. Also, please note that the `POTCAR` files are protected by the license so do not share them with anyone that does not have a license, including the support team, unless explicitly notified to do so.

### Module naming scheme

There are now one module per VASP flavor (standard, gamma only and non-collinear). Meaning when you now load a module, there is only one executable, `vasp`. For the example given above we load the module `VASP/5.4.4-intel-2019a-std-wannier90-somehash`. Here, `somehash` is a generated hash. i.e. `86f69b2cbd5b5987c9dd0bf21a1b7e82` based on the versions of libraries and extensions. This is done to avoid a very long module name. In the future we might chose to go with something even more compact, e.g. `VASP/5.4.4/somehash`. For now, we believe it is convenient to directly see which VASP version and what libraries are there etc., but not necessarily the Wannier90, BEEF, libxc version etc. The mapping between the hash and exact version numbers can be found here:

| Hash                             | Wannier90 | VTST | BEEF | SOL | libxc | hdf5 |
|----------------------------------|-----------|------|------|-----|-------|------|
| 86f69b2cbd5b5987c9dd0bf21a1b7e82 | 2.1.0     | -    | -    | -   | -     | -    |
|                                  |           |      |      |     |       |      |
|                                  |           |      |      |     |       |      |

The module list will grow in the next weeks (spring 2022). Also, notice that when we add new versions, we will add new modules and the only difference will be a different hash if the same inclusions is present. Users should then check back on this table to make sure they load a version that is to their specifications. In the future we will make sure the table is also updated in the repository that builds VASP and included here as a reference. Also, modules with names not containing a hash in the end is now considered legacy. You might use them to reproduce old results and we will keep them around until the system is end of life.

We would like feedback from the community how we can improve the module naming scheme going forward. Please keep in mind that whatever we chose have to support reproducability, which means that we should be able to say, _give me VASP version x.y.z, with BEEF version x1.y1.z1 and Wannier90 with x2,y2,z2_ etc.

The naming schemes of the modules are `VASP version-Toolchain-Additional-VASP flavor-Packages-Adaptions to source code`. Where:

- `VASP version` determines the VASP version, e.g. 5.4.4.
- `Toolchain` determines the toolchain used, typically which compilers, LAPACK, BLAS etc. routines have been used. This is based on the existing toolchains on the system. These can be inspected with `module show intel-2019a` for the particular system (e.g. `fram`). Typically, the `Toolchain` is the vendor, e.g. `intel` followed by the version, e.g. `2019a`.
- `VASP flavor` determines the VASP flavor, e.g. `std` for the standard flavor (`-DNGZhalf` added to `FPP`), `gam` for the gamma flavor (`-DNGZhalf -DwNGZhalf` added to `FPP`) and `ncl` for the non-collinear flavor.
- `Additional Packages` determines if an additional package has been included, e.g. `wannier90` (support for maximally-localised Wannier functions and the [Wannier90](http://www.wannier.org/)), `beef` (to yield support for the [BEEF](https://github.com/vossjo/libbeef) functional and Bayesian error estimates), `vtst` (to yield support for additional transition state tools [VTST](http://theory.cm.utexas.edu/vtsttools/)), `sol` (to yield support for solvation models using [VASPsol](https://github.com/henniggroup/VASPsol)) and `libxc` (to yield support for the exchange and correlation library using [libxc](https://www.tddft.org/programs/libxc/)).
- `Adaptions to source code` determines if there has been adaptions to the source code, e.g. restrictions in the ionic motions. For instance for `nor_x` the ionic motion/relaxation along the `x` (`x`, `y` and `z` is the unit cell axis supplied to VASP) direction.

### Further notes about the additional packages and how the modules have been constructed

The VTST scripts are available if you load a module with `vtst` and can be found in `$EBROOTVASP/vtst` after loading the VASP module containing `vtst`.

The `bee` executable from the BEEF library can be found in `$EBROOTBEEF/bin/bee`.

The `wannier90.x` and `postw90.x` executables of Wannier90 can be found in `$EBROOTWANNIER90/bin/wannier90.x` and `$EBROOTWANNIER90/bin/postw90.x`.

### Parallel functionality and library support.

All VASP and Wannier90 binaries are compiled with Intel MPI support, if they support it. No OpenMP is presently enabled, but we are working to extend the modules to also include that for VASP 6. This also includes GPU support for the methods in VASP that support this.

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
