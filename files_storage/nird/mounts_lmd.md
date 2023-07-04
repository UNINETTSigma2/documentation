# NIRD mounts on clusters

 The NIRD Storage project areas,tiered storage(TS) 
and datalke (DL), are mounted  on the login nodes of Betzy, Fram, or Saga.
One can directly access the NIRD project area from the login nodes of Betzy, Fram, or Saga.

The path for tiered storage(TS) project is

`/nird/projects/NSxxxxK`

and the path for datalake (DL) project is 

`/nird/datalake/NSxxxxK`

where ` NSxxxxK` is the ID of the project.



```{warning}

To avoid performance impact and operational issues, NIRD $HOME and project
areas are _not_ mounted on any of the compute nodes of the HPC clusters.
```

