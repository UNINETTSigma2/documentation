# GaussView

Gaussview is a visualization program that can be used to open Gaussian output files and checkpoint files (.chk) to display structures, molecular orbitals, normal modes, etc. You can also set up jobs and submit them directly.

Official documentation: [https://gaussian.com/gaussview6/](https://gaussian.com/gaussview6/)

## License and access
The license for Gaussian is commercial/proprietary and currently only UiT holds a site license. Thus GaussView on Sigma2 machines is available for UiT users only, unless a user or a group holds an invividual license to the code.

### GaussView on Fram
To load and run GaussView on Fram, load the relevant Gaussian module, and then call GaussView:

	$ module avail GaussView
	$ module load GaussView/6.0.16
	$ gview
