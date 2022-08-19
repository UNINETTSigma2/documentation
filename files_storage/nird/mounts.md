# NIRD mounts on clusters

When relevant, the NIRD Storage project areas are also mounted on the login
nodes of Betzy, Fram, or Saga.

Only the primary data volumes for projects are mounted to the HPC clusters:
- projects from NIRD-TOS to Fram
- projects from NIRD-TRD to Betzy and Saga

You can check what the primary site is for a project by running the following
on a NIRD login-node (replace "xxxx" with the actual project number you want to check):
```console
$ readlink /projects/NSxxxxK
```

This will print out a path starting either with /tos-project or /trd-project.
- If it starts with “tos” then the primary site is in Tromsø (login-tos.nird.sigma2.no)
- If it starts with “trd” then the primary site is in Trondheim (login-trd.nird.sigma2.no)
```

```{warning}
To avoid performance impact and operational issues, NIRD $HOME and project
areas are _not_ mounted on any of the compute nodes of the HPC clusters.
```
