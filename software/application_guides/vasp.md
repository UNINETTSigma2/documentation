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

You load the application by typing:

	$ module load VASP/<version>

For more information on available versions, type:

	$ module avail VASP

Users have to supply the necessary input files, inclusing any `POTCAR` files needed. They can be downloaded from the VASP portal you get access to with a valid VASP license. Also, please note that the `POTCAR` files are protected by the license.

## Specifics regarding the VASP install on Fram.

Note as of Febn 2020 that this is similar on Saga, but we are currently moving to a more automatic deployment system and the naming scheme is thus different on Saga. Expect Fram and Saga to be similar and this document updated by the end of spring 2020.

### VASP Binary naming scheme on Fram

Note that the VASP installation on Fram mainly follows the standard syntax introduced by the VASP team with their new installation scheme.

If you do check the bin directories of the different VASP installs on Fram you will see that there is quite a few binaries - and this might appear confusing. So here is an explanation:

1. All binaries are compiled with support for maximally-localised Wannier functions and the [Wannier90](http://www.wannier.org/) program, library for Bayesian error estimation functionals ([libbeef](https://github.com/vossjo/libbeef)) and also the MPI flag in FPP (-DMPI).

1. The binaries comes with a unmodified and a modified flavour (modifications are done in the file constr\_cell\_relax.F and are for simulating epitaxially constrained thin films (abfix) and for simulting cells with point defects which break the symmetry (noshear)). Basically, modification in this sense means a modification in the original source code without adding additional functionality (aka tools).

1. Support for various tools are added, including [occupation matrix control] (https://github.com/WatsonGroupTCD/Occupation-matrix-control-in-VASP), [transition state tools for VASP](http://theory.cm.utexas.edu/vtsttools/) and [implicit solvation model for VASP](http://vaspsol.mse.ufl.edu/)(file extensions *ocm*, *tst*, *sol* respectively).

To minimize the number of binaries, they are built in layers:
First, untooled versions were built - with and without modifications. vasp\_std/gam/ncl is totally unmodified in every way, while vasp\_std/gam/ncl\_abfix/noshear contains the modifications of constr\_cell\_relax.F as mentioned above.

Then, for the tooled binaries - we have assumed that vTST does not harm anything, thus all tooled versions are with this tool. Thus only the one with file extension *tst* has only vTST support and not the other tools mentioned. VASPsol is build on top of vTST - so binaries with the file extesion *sol* has got both vTST and VASPsol support. On top of this again, the binaries with occupation matrix support, with extension *ocm* has both vTST and VASPsol support together with occupation matrix support.

Also the tooled binaries are compiled with both unmodified and modified constr\_cell\_relax.F, making the total number of binaries 36 for this setup.

_Short summary:_

* umodified binaries: vasp_std, vasp_gam, vasp_ncl
* modified binaries: vasp\_std\_abfix, vasp\_ncl\_noshear
* tooled binaries: vasp\_std\_tst, vasp\_std\_tst\_noshear (both modified and unmodified.)
* tooled binaries comes in layers: tst is tst only, sol is tst *and* sol, ocm is tst *and* sol *and* ocm.

### FPP settings for each binary

The VASP installation on Fram mainly follows the build instructions provided by the VASP team. The makefile.include we use for Fram is the file called *makefile.include.linux_intel* in the *arch* folder in the vasp.5.4.4 distro. On top of this, we have added the line:

	CPP_OPTIONS+= -DVASP2WANNIER90v2 -Dlibbeef

for all our binaries. (for full build setup, feel free to ask)

##### All FPP settings in the makefile follows standard setup provided by VASP:

1. vasp_std is compiled with additional FPP flag -DNGZhalf.
2. vasp_gam is compiled with additional FPP flags -DNGZhalf -DwNGZhalf.
3. vasp_ncl is compiled with no additional FPP flags.

### Memory allocation for VASP in Fram

VASP is known to be potentially memory demanding. Quite often, you might experience to use less than the full number of cores on the node, but still all of the memory.

For relevant core-count, node-count and amounts of memory, see [About Fram](../../quick/fram.md).

For fram, currently the only way of increasing the memory per core available for jobs is to reduce the number of cores per node, please read up in the {ref}`job-scripts` section of the documentation.

## Citation

When publishing results obtained with the software referred to, please do check the developers web page in order to find the correct citation(s). Also, remember to acknowledge Sigma2 for the computational resources.
