# VASP (Vienna Ab initio Simulation Package)

VASP is a software package for performing ab-initio quantum-mechanical calculation of a periodic arrangement of atoms using the projector-augmented wave method and a plane wave basis set. The package can perform density-functional-theory (DFT) calculations, or many-body-perturbation-theory (MBPT) like GW etc. Please consult the documentation to get a more detailed overview of its feature set.

## Online information from vendor

* Homepage: https://www.vasp.at
* Documentation: https://cms.mpi.univie.ac.at/wiki/index.php/The_VASP_Manual
* User forum: https://www.vasp.at/forum/

## Installed on facilities

* Saga
* Fram

## Supported versions

* 5.4.4
* 6.1 (not yet installed, ETA will be notified)

## License and access policy

VASP is a commercial software package that requires a license for all who wants to use it. For a user to get access to the VASP executables installed on the Sigma2 systems they must perform the following steps:

* The users research group must have a valid VASP licence. To acquire a licence, please consult the "*How can I purchase a VASP license?*" in this link: https://www.vasp.at/faqs/.

* We need to get a confirmation from a VASP representative to confirm that the user have access to the license. The group representative of the user needs to contact the VASP team using licensing@vasp.at and ask them to send a confirmation email to support@metacenter.no to confirm that the user have a valid licence.

Once we receive the confirmation email we will add the user in question to the VASP group, which would give access to the VASP modules on the Sigma2 systems.


## Usage

You can check which VASP 5.4.4 versions are installed by executing:

	$ module load VASPModules
	$ module avail VASP

And load a particular module by executing (case sensitive):

	$ module load VASP/5.4.4-intel-2019a-std

Please remember to use two `module load`. The first loads the location of all the VASP modules and the second command loads the actual VASP module. It is also possible, if you know what module you want to execute::

	$ module load VASPModules VASP/5.4.4-intel-2019a-std

Users have to supply the necessary input files, inclusing any `POTCAR` files needed. They can be downloaded from the VASP portal you get access to with a valid VASP license. Also, please note that the `POTCAR` files are protected by the license so do not share them with anyone that does not have a license.

### Module naming schemes

There are now one module per VASP version. Meaning when you now load a module, there is only one executable, `vasp`.

The naming schemes of the modules are `VASP version-Toolchain-VASP flavor-Additional Package-Adaptions in source code.` where:

- `VASP version` determines the VASP version, e.g. 5.4.4
- `Toolchain` determines the toolchain used, typically which compilers, LAPACK, BLAS etc. routines have been used. This is based on the existing toolchains on the system. These can be inspected with `module show intel-2019a` for the particular system (e.g. `fram`).
- `VASP flavor` determines the VASP flavor, e.g. `std` for the standard flavor (`-DNGZhalf` added to `FPP`), `gam` for the gamma flavor (`-DNGZhalf -DwNGZhalf` added to `FPP`) and `ncl` for the non-collinear flavor.
- `Additional Package` determines if an additional package has been included, e.g. `beef` (to yield support for the `BEEF` functional and Bayesian error estimates, https://github.com/vossjo/libbeef).
- `Adaptions in source code` determines if there has been adaptions to the source code, e.g. restrictions in the ionic motions. For instance for `nor_x` the ionic motion/relaxation along the `x` (`x`, `y` and `z` is the unit cell axis supplied to VASP) direction.

for the example `5.4.4-intel-2019a-gam-beef-nor_x`.

We try to upload new versions if the VASP group issues new official patches and the naming scheme above does not indicate which patch is used as that is implicitly assumed to be using the latest released patch.

In addition, all binaries are compiled with support for maximally-localised Wannier functions and the [Wannier90](http://www.wannier.org/) program v2.1 and MPI enables. No OpenMP is enabled.

### A few notes and special modules

There is a module were `NMAX_DEG` is adjusted to 64 from the supplied value of 48. Since this is statically defined value, a special compile is necessary. If you get issues involving `NMAX_DEG`, please try a minimal working example using this executable and let us know if that solves your problem. Most likely you will encounter it again and we could try to compile an even larger value. However, also try to change your problem, like the symmetry and the representation you work in.

### Memory allocation for VASP

VASP is known to be potentially memory demanding. Quite often, you might experience to use less than the full number of cores on the node, but still all of the memory.

For relevant core-count, node-count and amounts of memory, see [About Fram](../../quick/fram.md) and [About Saga](,,/,,/quick/saga,md). There are two ways of increasing the memory pr. cpu over the standard node configuration:

- Increase the SLURM setting `mem-per-cpu`
- Utilize the nodes with more memory per cpu.

Remember you are accounted for the CPUs that would be reserved due to your demand for increased memory.

## Citation

When publishing results obtained with the software referred to, please do check the developers web page in order to find the correct citation(s). Also, remember to acknowledge Sigma2 for the computational resources.
