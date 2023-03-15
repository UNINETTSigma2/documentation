# NIRD mounts on clusters

 The NIRD Storage project areas,tiered storage(TS) 
 dismounted on the login nodes of Betzy, Fram, or Saga.

The path for tiered storage(TS) project is

`/nird/projects/NSxxxxK`


where ` NSxxxxK` is the ID of the project.

```{note}
The new NIRD is not yet mounted on HPCs.
Mount points for NIRD on the HPC systems and DNS entries will be updated 
as soon as all projects are migrated.
```

```{warning}

To avoid performance impact and operational issues, NIRD $HOME and project
areas are _not_ mounted on any of the compute nodes of the HPC clusters.
```

