# NIRD mounts on clusters

The NIRD project storage areas, namely NIRD Data Peak (DP) and NIRD Data Lake (DL) are mounted on the login nodes of Betzy, Fram, and Saga.
One can directly access the NIRD project area from the login nodes of the aforementioned compute clusters.

The path to Data Peak (DP) project areas is

`/nird/datapeak/NSxxxxK`

while the path to Data Lake (DL) project is 

`/nird/datalake/NSxxxxK`

where ` NSxxxxK` is the ID of the project.

```

```{warning}

To maintain optimal performance and prevent operational disruptions, the NIRD $HOME and Data Peak/Data Lake project areas are deliberately not mounted on any of the compute nodes within the HPC clusters.
```

