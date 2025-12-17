# The GAUSSIAN program system

* {ref}`gaussian-job-examples`
* {ref}`gaussian-resource-utilization`
* {ref}`gaussian-tuning`
* {ref}`gaussview`

[Gaussian](https://gaussian.com/) is a versatile program package for for electronic structure modeling and computational chemistry, and frequently used on NRIS Hardware platforms. 

**Official documentation:** <https://gaussian.com/man>

## License and access

The installed license of GAUSSIAN on NRIS machines is an academic license and consists of four individual site licenses bought by the four partner Universities of NRIS (NTNU, UiB, UiO, UiT). Users from these institutions should be autmatically added to the `gaussian`group of users. Other users need to document valid access to the relevant license (academic and/or full commercial) before being granted access to Gaussian in NRIS.

- To have access to the code, you need to be in the `gaussian` group of
  users. <br>Check this with the command `id | grep gaussian`. 
- If not in the group, {ref}`contact us <support-line>` and ask to be added.
- Provide necessary documentation to be added in the group. 

## Citation

For the recommended citation, please consult [gaussian.com/citation](https://gaussian.com/citation/)


## Gaussian on NRIS machinery

Currently, the Gaussian software is installed on {ref}`saga`, {ref}`betzy` and {ref}`olivia`. In saga and fram we use a slightly unorthodox setup for Gaussian - redirecting LD library path to rsocket instead of socket library before loading and starting binaries, which ensures securing satisfactory scaling beyond the 2 nodes/Linda instances (See whitepaper: [Improving Gaussian’s parallel performance using Infiniband](gaussianoverib.pdf)).

On betzy and olivia, Gaussian has been installed in a more standard way. Though, some preparations are needed user-side to use `Linda` parallelization. Look at {ref}`gaussian-job-examples` for information on how to use `Linda` properly.

So, if you see this warning in your Slurm output, there is not a reason for concern:
```text
ntsnet: WARNING: /cluster/software/Gaussian/g16_C.01/linda-exe/l302.exel may
not be a valid Tcp-Linda Linda executable.
```

Also note that there are internal differences between the different NRIS machines in terms of better practice for running Gaussian jobs. This will be further discussed in the {ref}`gaussian-job-examples`sections and/or the {ref}`gaussian-resource-utilization` section.
