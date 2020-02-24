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

Note that this is similar to Saga, but we are currently moving to a more automatic deployment system and the naming scheme is thus different on Saga.



## Citation

When publishing results obtained with the software referred to, please do check the developers web page in order to find the correct citation(s). Also, remember to acknowledge Sigma2 for the computational resources.
