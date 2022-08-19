---
orphan: true
---

# The GAUSSIAN program system

* [Gaussian NRIS machines job examples](gaussian_job_example.md)
* [Memory and number of cores](gaussian_resources.md)
* [Tuning Gaussian](gaussian_tuning.md)
* [GaussView](gaussview.md)

[Gaussian](http://gaussian.com/) is a versatile program package for for electronic structure modeling and computational chemistry, and frequently used on NRIS Hardware platforms. 
<p>
**Official documentation:** <http://gaussian.com/man>

## License and access

The installed license of GAUSSIAN on NRIS machines is an academic license and consists of four individual site licenses bought by the four partner Universities of NRIS (NTNU, UiB, UiO, UiT). Users from these institutions should be autmatically added to the `gaussian`group of users. Other users need to document valid access to the relevant license (academic and/or full commercial) before being granted access to Gaussian in NRIS.

- To have access to the code, you need to be in the `gaussian` group of
  users. Check this with `id | grep gaussian`. 
- If not in the group, contact {ref}`support-line` and ask to be added.
- Provide necessary documentation to be added in the group. 

## Citation

For the recommended citation, please consult [gaussian.com/citation](https://gaussian.com/citation/)


## Gaussian on NRIS machinery

Currently, the Gaussian software is installed on {ref}`fram` and {ref}`saga`. We use a slightly unorthodox setup for Gaussian - redirecting LD library path to rsocket instead of socket library before loading and starting binaries, which ensures securing satisfactory scaling beyond the 2 nodes/Linda instances (See whitepaper: [Improving Gaussian’s parallel performance using Infiniband](gaussianoverib.pdf)).

So, if you see this warning in your Slurm output, there is not a reason for concern:
```text
ntsnet: WARNING: /cluster/software/Gaussian/g16_C.01/linda-exe/l302.exel may
not be a valid Tcp-Linda Linda executable.
```

Also note that there are internal differences between the different NRIS machines in terms of better practice for running Gaussian jobs. This will be further discussed in the [Gaussian NRIS machines job examples](gaussian_job_example.md) sections and/or the [Memory and number of cores](gaussian_resources.md) section.
