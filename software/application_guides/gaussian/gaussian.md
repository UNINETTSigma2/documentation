---
orphan: true
---

# The GAUSSIAN program system

* [Gaussian NRIS machines job examples](gaussian_job_example.md)
* [Memory and number of cores](gaussian_resources.md)
* [Tuning Gaussian](gaussian_tuning.md)
* [GaussView](gaussview.md)
* [NRIS do´s and don´ts with Gaussian](gaussian_dosndonts.md)

[Gaussian](http://gaussian.com/) is a popular computational chemistry program.
Official documentation: <http://gaussian.com/man>


## License and access

The license for Gaussian is commercial/proprietary and constitutes of 4 site
licenses for the 4 current host institutions of Metacenter sites; NTNU, UiB,
UiO, UiT. Only persons from one of these institutions have access to the
Gaussian Software system installed on NRIS machines. Note that users that do not come
from one of the above mentioned institutions still may be granted access, but
they need to document access to a valid license for the version in question
first.

- To get access to the code, you need to be added to the `gaussian` group of
  users. Contact {ref}`support-line`.
- To be in the `gaussian` group of users, you need be qualified according to abovementioned criterias.


## Citation

For the recommended citation, please consult [http://gaussian.com/citation/](http://gaussian.com/citation/).


## Gaussian on NRIS machinery

Currently, the Gaussian software is installed on {ref}`fram` and {ref}`saga`. We use a slightly unorthodox setup for Gaussian - redirecting LD library path to rsocket instead of socket library before loading and starting binaries, which ensures securing satisfactory scaling beyond the 2 nodes/Linda instances (See whitepaper: [Improving Gaussian’s parallel performance using Infiniband](gaussianoverib.pdf)).

So, if you see this warning in your Slurm output, there is not a reason for concern:
```text
ntsnet: WARNING: /cluster/software/Gaussian/g16_C.01/linda-exe/l302.exel may
not be a valid Tcp-Linda Linda executable Warning: Permanently added the ECDSA
host key for IP address '10.33.5.24' to the list of known hosts.
```

Also note that there are internal differences between the different NRIS machines in terms of better practice for running Gaussian jobs. This will be further discussed in the [Gaussian NRIS machines job examples](gaussian_job_example.md) sections and/or the [Memory and number of cores](gaussian_resources.md) section.
