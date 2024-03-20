(backup service)=



# NIRD Backup

NIRD provides backup as a service. Projects using the NIRD Data Peak (TS) service can opt-in to use the **NIRD Backup** service to ensure a secondary copy of the dataset(s) that require a higher level of security. Data owners/data custodians can make decisions regarding the level of security needed for each dataset in a granular manner. This will persist during the tenure of the project. The backup service can be requested by flagging this in the application form for storage resources during regular calls.

The backup is from TS to DL.

Should there be requirement for backup for certain dataset(s), those **must** use the Data Peak service, i.e., be placed on the TS resource.

For example, a project may be using both Data Peak and Data Lake services, and sunch, has an allocation on both TS and DL resources. In this case the project can decide how to use the dedicated storage resources. However, should one require backup for a particular dataset, then: 
 
 - that shall be flagged during the application process,
 - and that dataset shall be placed on the TS resource.


```{note}
Notice that datasets using the NIRD Data Lake (DL) service *cannot* opt in for the NIRD Backup service.
```


- The Data Peak (TS) path on the system is `/nird/datapeak`. For backward compatibility, a parallel mount point to `/nird/projects` is maintained.
- The Data Lake (DL) path on the system is `/nird/datalake`

We advice projects to assess which of the dataset needs a higher level of 
security and should be backed up.

In general, one can consider which data can be easily reproduced, and which 
are copies of files stored on other storage resources. These data normally 
do not need backup service.


Restoring data from backup is done by NRIS. Should you need to restore data from backup, please contact NRIS support.


## Include, exclude rules
```{warning}
Projects will be able to control which file are included and which are excluded from backup.

Notice, this system is currently only partially implemented. Please notify operations if you have adjusted the control file.
```

The solution for including or excluding data from backup for projects using the Data Peak service is implemented by using a control file.

The control file is named `.replication_exclude` and must be placed in the
root of the project directory.
 e.g.: `/nird/projects/NS1234K/.replication_exclude`

To exclude specific files or directories, those shall be listed in the
`.replication_exclude` control file. Each file or directory which is to be
excluded from replication, shall be added as a separate line.

Lines in the `.replication_exclude` control file starting with `#` or `;` are
ignored.

### Excluding a specific file

To exclude the `/nird/projects/NS1234K/datasets/experiment/tmp_file.nc` file,
add `/datasets/experiment/tmp_file.nc` into the `.replication_exclude` control
file as a line on it's own.


### Excluding a directory

To exclude the `/nird/projects/NS1234K/datasets/non_important/` directory,
add `/datasets/non_important` into the `.replication_exclude` control file
as a line on it's own.

Mentions of `/datasets` on its own, would exclude everything in that directory.


