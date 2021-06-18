# Granular replication for NIRD projects

The solution for granular data replication for project data stored on 
NIRD Data Storage, is implemented by using a control file.

The control file is named `.replication_excludes` and must be placed in the 
root of the project directory. 
 e.g.: `/tos-project1/NS1234K/.replication_exclude`

To exclude specific files or directories, those shall be listed in the 
`.replication_exclude` control file. Each file or directory which is to be
excluded from replication, shall be added as a separate line.

* Exclude a directory
To exclude the `/tos-project1/NS1234K/datasets/non_important/` directory, 
add `/datasets/non_important` into the `.replication_excludes` control file
as a line on it's own.

* Exclude a specific file
To exclude the `/tos-project1/NS1234K/datasets/experiment/tmp_file.nc` file, 
add `/datasets/experiment/tmp_file.nc` into the `.replication_excludes` control
file as a line on it's own.

Mentions of `/datasets` on its own, would exclude everything in that directory.

Lines in the `.replication_exclude` control file starting with `#` or `;` are 
ignored.

e.g.:
```
# exclude all files from non_important directory
/datasets/non_important
# exclude temporary data from the experiment_A directory
/datasets/experiment_A/tmp_file.nc
# exclude tmp subdirectory from the experiment_B directory
/datasets/experiment_B/tmp
```

