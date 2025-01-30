(backup service)=



# NIRD Backup

NIRD provides backup as a service. Projects using the NIRD Data Peak (DP) service can opt in to use the NIRD Backup service to ensure a secondary copy of the dataset(s) that require a higher level of security. Data owners/data custodians can make decisions regarding the level of security needed for each dataset in a granular manner. This will persist during the tenure of the project. The NIRD Backup service can be requested by flagging this in the application form for storage resources during regular calls.

Should there be requirement for backup for certain dataset(s), those must use the Data Peak service, i.e., be placed on the DP resource.

For example, a project may be using both Data Peak and Data Lake services, and such, has an allocation on both DP and DL resources. In this case the project can decide how to use the dedicated storage resources. However, should one require backup for a particular dataset, then:
 
 - that shall be flagged during the application process.
 - and that dataset shall be placed on the TS resource.


```{note}
Notice that datasets using the NIRD Data Lake (DL) service *cannot* opt in for the NIRD Backup service.
```


- The Data Peak (TS) path on the system is `/nird/datapeak`. For backward compatibility, a parallel mount point to `/nird/projects` is maintained.
- The Data Lake (DL) path on the system is `/nird/datalake`

We advise projects to assess which of the dataset needs a higher level of 
security and should be backed up.

In general, one can consider which data can be easily reproduced, and which 
are copies of files stored on other storage resources. These data normally 
do not need backup service.


Restoring data from backup is done by NRIS. Should you need to restore data from backup, please contact NRIS support.

## Backup policy
### Prerequisite
1. During the application process, the project requested the NIRD Backup service by selecting “Yes” at the “Do you need backup?” question in the application form.

2. The project has data on the NIRD Data Peak (DP) resource.

### Policy
1. By default, all data stored by the project on Data Peak is backed up, with the exception of data located under the directory `/nird/datapeak/$project/No_backup`.

2. Custom rules for the scenario where files cannot be stored in the No_backup directory can be defined in the `/nird/datapeak/$project/.backup_exclude` control file. Each project has its own `.backup_exclude` file.

3. The backup system maintains two versions of each file: an active and an inactive version. The retention policy for active files is 60 days, while the retention period for inactive files is 30 days. For further information on file versioning, please refer to the following link: https://www.ibm.com/support/pages/file-versioning-file-retention-and-file-expiration-explained

4. The utilization of NIRD Backup is governed by the user contribution model outlined at https://www.sigma2.no/user-contribution-model. Backup utilization can be monitored using the `dusage -p NSxxxxK` command. Note, that accounting information pertaining to NIRD Backup will be made accessible to project leaders within the MAS platform during 2025.

```{note}
Please be advised that backup utilization may be affected by the number of file changes (active and inactive file versions) and exclusion rules.

Additionally, please note that operations are currently working on enabling accounting for all projects. As per plan, accounting will be enabled for all projects by the 2025.1 allocation period. 
```

### Exclude rules
1. Projects will be granted the ability to specify which files are excluded from backup. However, for the time being, the control file can only be modified by system administrators. Kindly inform NRIS operations if you require adjustments to the control file.

2. Operations have transferred the rules from the former `.replication_exclude` control file to the `.backup_exclude control file`.