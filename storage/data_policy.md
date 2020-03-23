

# Data handling and storage policy

<div class="alert alert-warning">
  <h4>User areas and project areas are private</h4>
  <p>
    You can share files with other project members using project areas.
  </p>
</div>

All data accessed, stored, communicated, or transferred on any national HPC
system (Betzy, Fram, Saga, Stallo, Vilje) or the National e-Infrastructure for Research Data (NIRD),
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
created. Access to the shared project area is at the PL's sole discretion.
For accounting purposes, the group ownerships are regularly controlled, and
in case needed, reset.
