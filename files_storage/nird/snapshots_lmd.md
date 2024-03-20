# Snapshots on NIRD

Both home directories (`/nird/home/$USER`) and all project areas in 
NIRD Data Peak (`/nird/datapeak/NSxxxxK`) and NIRD Data Lake (`/nird/datalake/NSxxxxK`)
have temporary, point-in-time copy of data in the form of snapshots.

Snapshots are taken with the following frequencies:
* `/nird/home/$USER`:
  - daily snapshots for the last 7 days
  - weekly snapshots for the last 6 weeks

* `/nird/datapeak/NSxxxxK`:
  - daily snapshots for the last 7 days
  - weekly snapshots for the last 6 weeks

* `/nird/datalake/NSxxxxK`:
  - daily snapshots for the last 7 days
  - weekly snapshots for the last 6 weeks

```{warning}
 Kindly note that snapshots are temporary and if the datasets needs higher
 level of security in the form of backup on a secondary storage, project leaders must use the {ref}`backup service`.
```

## Where the snapshots are located

The NIRD `$HOME` and NS project snapshots are available under:
- `/nird/home/.snapshots`
- `/nird/datapeak/NSxxxxK/.snapshots`
- `/nird/datalake/NSxxxxK/.snapshots`

A deleted/overwritten file in the home directory on NIRD can be recovered like this:

```console
$ cp /nird/home/.snapshots/DATE/$USER/mydir/myfile /nird/home/$USER/mydir/
```
Note that snapshots are taken every night only. This means that deleted files
which did not exist yet yesterday cannot be recovered from snapshots.

To recover a deleted or overwritten file in NIRD Data Peak `/nird/datapeak/NSxxxxK/dataset1/myfile`,
you can copy a snapshot back to the folder and restore the deleted/overwritten file like this::

```console
$ cp /nird/datapeak/NSxxxxK/.snapshots/DATE/dataset1/myfile /nird/datapeak/NSxxxxK/dataset1/
```

To recover a deleted or overwritten file in NIRD Data Lake `/nird/datalake/NSxxxxK/dataset1/myfile`,
you can copy a snapshot back to the folder and restore the deleted/overwritten file like this:

```console
$ cp /nird/projects/NSxxxxK/.snapshots/DATE/dataset1/myfile /nird/projects/NSxxxxK/dataset1/
```

Select the DATE accordingly to your case.

