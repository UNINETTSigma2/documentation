# Backup

## Fram

Home directories on the Fram HPC cluster (`/cluster/home`) are backed up each night user's home directory on NIRD (`/nird/home/$USER/backup/fram`).

`NN*K` project directories (`/cluster/projects`) are backed up nightly to NIRD. To not to be confused with `NS*K` projects in `/nird/projects`.

Meta-project areas (`/cluster/shared`) are *not* backed up to NIRD.

Scratch areas (`/cluster/work` and `/node/scratch`) do *not* have any backup.

## NIRD

Both home directories (`/nird/home`) and project areas (`/nird/projects`) have
backup in the form of snapshots and geo-replication.

Snapshots are taken with the following frequencies:
* `/nird/home`: 
  - daily snapshots for the last 7 days
  - weekly snapshots for the last 6 weeks
* `/nird/projects`:
  - daily snapshots for the last 7 days
  - weekly snapshots for the last 6 weeks

---

Note `/nird/nome` and `/nird/projects` are no longer mounted on Fram. To restore files, you have to login to NIRD.

If you have not accessed NIRD before, you will need to enable your account by asking for activation key and
temporary password at `https://www.metacenter.no` (select  User Login (Passwords)) or contact support in order to have your files restored.

---


## List snapshots

Snapshots are available at the following places:
* `/nird/home`: 
  - /nird/home/u1/.snapshots
* `/nird/projects`:
  - /nird/projects/nird/NSxxxxK/.snapshots

## Restore data

To recover the file 

     /nird/projects/NSxxxxK/dataset1/myfile

inadvertently deleted (or to recover an older version) do

    cp /nird/projects/nird/NSxxxxK/.snapshots/DATE/dataset1/myfile /nird/projects/nird/NSxxxxK/dataset1/
    
select the date accordingly to your case.
This procedure is applicable to files in home directories.      
Note that snapshots are taken every night only. So file created the day of the deletion   
cannot be recovered with snapshots.

