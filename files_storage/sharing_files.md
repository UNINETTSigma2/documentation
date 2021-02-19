# Data handling and storage policy

```{warning}
**User areas and project areas are private**

You can share files with other project members using project areas.
```

All data accessed, stored, communicated, or transferred on any national HPC
system (Betzy, Fram and Saga) or the National e-Infrastructure for Research Data (NIRD),
must be handled in compliance to legal and regulatory requirements.

In addition, all data has to be directly related to the work effectuated and/or
the research project(s) the user is participating.


## User areas

User's private data (such as keys, sessions, e-mails, etc.) may reside in their
home directory (`$HOME`).
`$HOME` **is not a shared area** and all data stored there has to be treated as
being private, regardless of its content.

To limit access to `$HOME` only to the user and designated system administrators,
the directory permissions are set to 0700 (meaning: only the user can read, write, and execute).
Permissions are regularly controlled, and in case of mismatch, reset.

On the HPC clusters, users also have a *user work area*,
`/cluster/work/users/$USER` (`$USERWORK`). This is also a **private**
area, and the permissions are set so that only the user has access to
the area.


## Project areas

Project data is private to the project and shared between the project members.
The project leader (PL) has sole discretion over project members, thus access
to the project area(s).

Project local to a particular HPC system has its own directory, created with
permissions set to 2770 (meaning that only the group can read, write, and execute)
to set the global group ID.

Group ownership is regularly controlled for each project directory and reset in
case needed. This is required for storage accounting purposes.


## Shared project areas

In special cases there might be a need for sharing data between projects for
collaboration and possibly preventing data duplication.

If such a need is justified, a meta-group and the corresponding directory is
created. Access to the shared project area is at the Project Leader's (PL) sole discretion.
For example, if the PL of the project owning the file group `acme` wants 
`/cluster/shared/acme/inventory_db` to be world readable, the project leader is allowed to do this 
(e.g., by running, `chmod -R o+r /cluster/shared/acme/inventory_db`).

Note that the shared project areas **must not** contain any private data.

Also note that you **must never** set any directory or file to world writable.

For accounting purposes, the group ownerships are regularly controlled, and
in case needed, reset.

If you need to share with the outside world (outside of your HPC systems),
please refer to our [File transfer documentation](file_transfer.md).
