(backup service)=



# Backup as a Service on NIRD

NIRD provides backup as a service. NIRD projects on Tiered Storage (NIRD TS)
can utilise the service for the dataset(s) that needs a higher level of security.
This will stay during the tenure of the project. The backup service can be requested by flagging this in the application form for storage resources during regular calls.

The backup is from NIRD TS to NIRD DL. 

Should there be requirement for backup for certain dataset(s), those **must** be placed on NIRD TS.

For example, if a project has an allocation on both NIRD TS and NIRD DL, the project can decide how to use the dedicated storage resources. However, should one require backup for a particular dataset, then: 
 
 - that shall be flagged during the application process,
 - and that dataset shall be placed on the NIRD TS.


```{note}
Notice, that there is no backup service for the data in the Data Lake.
```


- Tiered Storage (NIRD TS) path on the system is `/nird/projects`
- Data Lake (NIRD DL) path on the system is `/nird/datalake`

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

The solution for including or excluding data from backup for project data stored on NIRD TS, 
is implemented by using a control file.

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


