# Data locality and replication


## Data replication

NIRD data storage projects are - with some exceptions, mutually agreed with the
project leader - stored on two sites and asynchronously geo-replicated.

The main purpose for the replica is to ensure data integrity and resilience in
case of large damage at the primary site.

We advice projects to assess which of the dataset needs a higher level of 
security and should be replicated. This helps in optimizing the storage space 
used by the project.

In general, one can consider which data can be easily reproduced, and which 
are copies of files stored on other storage resources. These data normally do 
not need replication, and can be considered excluded from replication.


## Data locality

For every project that has requested replication, the data is stored on a 
primary data volume on one site and the replica on the other site.

The primary site is chosen based on operational convenience, that is to be the 
one closest to where the data is consumed, namely NIRD-TOS if data is 
analysed on the Fram HPC cluster, or NIRD-TRD if data is analysed on the Saga 
or on the Betzy HPC clusters.

Projects have the possibility to read from and write to the primary site, while
they cannot read from or write to the secondary site.

```{warning}
The users should log onto the login container nearest to the primary data
storage.
```


## Granular replication for NIRD projects

The solution for granular data replication for project data stored on 
NIRD data storage, is implemented by using a control file.

The control file is named `.replication_exclude` and must be placed in the 
root of the project directory. 
 e.g.: `/tos-project1/NS1234K/.replication_exclude`

To exclude specific files or directories, those shall be listed in the 
`.replication_exclude` control file. Each file or directory which is to be
excluded from replication, shall be added as a separate line.

Lines in the `.replication_exclude` control file starting with `#` or `;` are 
ignored.


### Excluding a specific file

To exclude the `/tos-project1/NS1234K/datasets/experiment/tmp_file.nc` file, 
add `/datasets/experiment/tmp_file.nc` into the `.replication_exclude` control
file as a line on it's own.


### Excluding a directory

To exclude the `/tos-project1/NS1234K/datasets/non_important/` directory, 
add `/datasets/non_important` into the `.replication_exclude` control file
as a line on it's own.

Mentions of `/datasets` on its own, would exclude everything in that directory.


### Example

```
# exclude all files from non_important directory
/datasets/non_important
# exclude a specific file from the experiment_A directory
/datasets/experiment_A/tmp_file.nc
# exclude all files from tmp subdirectory from the experiment_B directory
/datasets/experiment_B/tmp
```
