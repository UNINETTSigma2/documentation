(backup service)=



# Backup as a Service on NIRD

NIRD provides backup as a service. NIRD projects on Tiered Storage (NIRD TS)
can utilise the service for the dataset that needs a higher level of security.
This will stay during the tenure of the project. The backup service will be 
activated after a mutula agreement with the project leader during the allocation or later.

The backup is from Tiered Storage (NIRD TS) to Data Lake (NIRD DL).
There will not be any backup service for the data in the Data Lake.

- Tiered Storage (NIRD TS) path on the system is `/nird/projects`
- Data Lake (NIRD DL) path on the system is `/nird/datalake`

We advice projects to assess which of the dataset needs a higher level of 
security and should be backedup.

In general, one can consider which data can be easily reproduced, and which 
are copies of files stored on other storage resources. These data normally 
do not need backup service.

The solution for backup for project data stored on NIRD TS, 
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

```{note}
Backup is configured on the new NIRD and will be fully activated during the 
initial production phase.

```

