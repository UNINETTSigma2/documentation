# VASP (Vienna Ab initio Simulation Package)

VASP is a software package for performing ab-initio quantum-mechanical calculation of a periodic arrangement of atoms using the projector-augmented wave method and a plane wave basis set. The package can perform density-functional-theory (DFT) calculations, or many-body-perturbation-theory (MBPT) like GW etc. Please consult the documentation to get a more detailed overview of its feature set.

## Online information from VASP developers

* [Homepage](https://www.vasp.at)
* [Documentation](https://cms.mpi.univie.ac.at/wiki/index.php/The_VASP_Manual)
* [User forum](https://www.vasp.at/forum/)

## Installed on facilities

* Saga
* Fram

## Supported versions

* 5.4.4
* 6.1 (not yet installed, ETA will be notified, we are pending updates from the VASP group pertaining to agreements to install/maintain it on our clusters, in essence the agreements of a cluster maintenance license)

## License and access policy

VASP is a commercial software package that requires a license for all who wants to use it. For a user to get access to the VASP executables installed on the Sigma2 systems they must perform the following steps:

* The users research group must have a valid VASP licence. To acquire a licence, please consult the [How can I purchase a VASP license?](https://www.vasp.at/faqs/).

* We need to get a confirmation from a VASP representative to confirm that the user have access to the license. The group representative of the user needs to contact the VASP team using licensing@vasp.at and ask them to send a confirmation email to `support@metacenter.no` to confirm that the user have a valid license. It is very important that you communicate clearly to the VASP group that we want emails on this address.

Once we receive the confirmation email we will add the user in question to the VASP group, which would give access to the VASP modules on the Sigma2 systems.

Notice that the VASP license is backwards compatible, meaning that if you are issues a VASP 6 license you also have access to VASP 5.

## Usage

You can check which VASP 5.4.4 versions are installed by executing:

	$ module load VASPModules
	$ module avail VASP

And load a particular module by executing (case sensitive):

	$ module load VASP/5.4.4-intel-2019a-std

Please remember to use two `module load`. The first loads the location of all the VASP modules and the second command loads the actual VASP module. It is also possible, if you know what module you want to execute::

	$ module load VASPModules VASP/5.4.4-intel-2019a-std

Users have to supply the necessary input files, including any `POTCAR` files needed. They can be downloaded from the VASP portal you get access to with a valid VASP license. Also, please note that the `POTCAR` files are protected by the license so do not share them with anyone that does not have a license, including the support team, unless explicitly notified to do so.

### Module naming schemes

There are now one module per VASP version. Meaning when you now load a module, there is only one executable, `vasp`. In order to make it crystal clear to the users what versions of the additional packages have been used, the module names are unfortunately quite long. However, we hope this will at least give transparency and better facilitate reproducibility.

The naming schemes of the modules are `VASP version-Toolchain-Additional Packages-Adaptions to source code-VASP flavor`. Where:

- `VASP version` determines the VASP version, e.g. 5.4.4
- `Toolchain` determines the toolchain used, typically which compilers, LAPACK, BLAS etc. routines have been used. This is based on the existing toolchains on the system. These can be inspected with `module show intel-2019a` for the particular system (e.g. `fram`). Typically, the `Toolchain` is the vendor, e.g. `intel` followed by the version, e.g. `2019a`.
- `Additional Packages` determines if an additional package has been included, e.g. `wannier90` (support for maximally-localised Wannier functions and the [Wannier90](http://www.wannier.org/)), `beef` (to yield support for the [BEEF](https://github.com/vossjo/libbeef) functional and Bayesian error estimates), `vtst` (to yield support for additional transition state tools [VTST](https://theory.cm.utexas.edu/vtsttools/)) and `sol` (to yield support for solvation models using [VASPsol](https://github.com/henniggroup/VASPsol)). Following the package name is the version of that specific package, e.g. `beef-0.1.1`, meaning the `beef` package is included using version `0.1.1`. For multiple packages and combination, the list continues.
- `Adaptions to source code` determines if there has been adaptions to the source code, e.g. restrictions in the ionic motions. For instance for `nor_x` the ionic motion/relaxation along the `x` (`x`, `y` and `z` is the unit cell axis supplied to VASP) direction. It does not have any version following its label.
- `VASP flavor` determines the VASP flavor, e.g. `std` for the standard flavor (`-DNGZhalf` added to `FPP`), `gam` for the gamma flavor (`-DNGZhalf -DwNGZhalf` added to `FPP`) and `ncl` for the non-collinear flavor. As for the adaptions, no version is following these labels.

i.e. for the example module `5.4.4-intel-2019a-beef-0.1.1-nor_x-gam`.

### Further notes about the additional packages and how the modules have been constructed

Since `sol`, `beef` and `wannier90` does not modify the run-time behavior in any way (you have to enable special flags to enable its functionality, please consult the respective documentations), they are included for all the versions. `vtst` do however modify the original behavior of VASP for some cases and is thus included as a separate additional package.

The VTST scripts are available if you load a module with `vtst` and can be found in `$EBROOTVASP/vtst` after loading the VASP module containing `vtst`.

The `bee` executable from the BEEF library can be found in `$EBROOTBEEF/bin/bee`.

The `wannier90.x` and `postw90.x` executables of Wannier90 can be found in `$EBROOTWANNIER90/bin/wannier90.x` and `$EBROOTWANNIER90/bin/postw90.x`.

### Patches

We try to upload new versions if the VASP group issues new official patches and the naming scheme above does not indicate which patch is used as that is implicitly assumed to be using the latest released patch.

### A few notes and special modules

There are modules were `NMAX_DEG` (`ndegX`) is adjusted to `X=64, 128 and 256` from the default value of 48. Since this is statically defined value, a special compile is necessary. If you get issues involving `NMAX_DEG`, please try a minimal working example using this executable and let us know if that solves your problem. Most likely you will encounter it again and we could try to compile an even larger value. However, also try to change your problem, like the symmetry and the representation you work in.

### Parallel functionality and library support.

All VASP and Wannier90 binaries are compiled with Intel MPI support, if they support it. No OpenMP is enabled. For the binaries of the additional packages, no parallelization is available.

### Memory allocation for VASP

VASP is known to be potentially memory demanding. Quite often, you might experience to use less than the full number of cores on the node, but still all of the memory.

For relevant core-count, node-count and amounts of memory, see [About Fram](https://documentation.sigma2.no/hpc_machines/fram.html) and [About Saga](https://documentation.sigma2.no/hpc_machines/saga.html). There are two ways of increasing the memory pr. cpu over the standard node configuration:

- Increase the SLURM setting `mem-per-cpu`
- Utilize the nodes with more memory per cpu.

Remember you are accounted for the CPUs that would be reserved due to your demand for increased memory.

## Citation

When publishing results obtained with the software referred to, please do check your license agreement and the developers web page in order to find the correct citation(s). Also, remember to acknowledge Sigma2 for the computational resources.
