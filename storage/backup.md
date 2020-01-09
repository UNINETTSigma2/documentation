# Backup

This page describes the backups taken on NIRD and the HPC clusters.

**NOTE:** _ONLY_ home directories having quotas enabled are being backed up. Any $HOME
area using more then 20GiB is skipped. To have your $HOME backed up, you need to shrink   the disk usage below the 20GiB limit and notify support. `dusage` reporting 0 Bytes limit means, that you do _not_ have disk quotas activated.

## Fram

Home directories on the Fram HPC cluster (`/cluster/home/$USER`) are backed up nightly to user's home directory on NIRD (`/nird/home/$USER/backup/fram`). `$HOME/nobackup` and `$HOME/tmp` directories are excluded from the backup.

The `nnXXXXk` project directories (`/cluster/projects/nnXXXXk`) are backed up
nightly to NIRD.

The shared areas in `/cluster/shared` are *not* backed up to NIRD.

The scratch areas in `/cluster/work/jobs` and `/cluster/work/users` do *not* have any backup.

## Saga

**Note: Backup for project areas on Saga has not been implemented yet, but will be
soon!**

Home directories on the Saga HPC cluster (`/cluster/home/$USER`) are backed up nightly to user's home directory on NIRD (`/nird/home/$USER/backup/saga`). `$HOME/nobackup` and `$HOME/tmp` directories are excluded from the backup.

The `nn*k` project directories (`/cluster/projects/nnXXXXk`) are backed up
nightly to NIRD.

The shared areas in `/cluster/shared` are *not* backed up to NIRD.

The scratch areas in `/cluster/work/jobs` and `/cluster/work/users` do *not* have any backup.

## NIRD

Both home directories (`/nird/home/$USER`) and project areas (`/nird/projects/NSxxxxK`) have
backup in the form of snapshots and geo-replication.

Snapshots are taken with the following frequencies:
* `/nird/home/$HOME`:
  - daily snapshots for the last 7 days
  - weekly snapshots for the last 6 weeks
  - Note that this also includes snapshots of the Fram and Saga
    homedir backup
* `/nird/projects/NSxxxxK`:
  - daily snapshots for the last 7 days
  - weekly snapshots for the last 6 weeks


### List snapshots

Snapshots are available at the following places:
* `/nird/home`:
  - /nird/home/u1/.snapshots
* `/nird/projects/NSxxxxK`:
  - /nird/projects/NSxxxxK/.snapshots

### Restore data

To recover a file 

     /nird/projects/NSxxxxK/dataset1/myfile

inadvertently deleted (or to recover an older version) do

    cp /nird/projects/NSxxxxK/.snapshots/DATE/dataset1/myfile /nird/projects/NSxxxxK/dataset1/
    
Select DATE accordingly to your case.  Similarly, for a file in the
home directory:

	cp /nird/home/u1/.snapshots/DATE/$USER/mydir/myfile /nird/home/$USER/mydir/

Note that snapshots are taken every night only. So file created the
day of the deletion cannot be recovered with snapshots.
