# Data Handling and Storage Policy

All data accessed, stored, communicated or transferred on any national HPC
system (e.g. Fram) and the National e-Infrastructure for Research Data (NIRD),
must be handled in compliance to legal and regulatory requirements.

In addition, all data has to be directly related to the work effectuated and/or
the research project(s) the user is participating.


## User Area

User's private data (such as keys, sessions, e-mail, etc.) may reside in their
home directory ($HOME).
$HOME is **not** a shared area and all data stored there has to be treated as 
being private, regardless of it's content.

To limit access to $HOME only to the user and designated system administrators,
the default directory permissions are set to 0700.
Permissions are regularly controlled and in case of mismatch reset to defaults.


## Project Area

Project data is private to the project and shared between the project members.
The project leader (PL) has sole discretion over project members, thus access 
to the project area(s).

Projects local to a particular HPC system has it's own directory, created with
permissions set to 2770 to set the global group ID.
Group ownership is regularly controlled for each project directory and reset in
case needed. This is required for storage accounting purposes.


### Project collaborations

In special cases there might be a need for sharing data between projects for 
collaboration and possibly preventing data duplication.

If such a need is justified, a meta-group and it's according directory is
created. Access to the shared project area is at the PL's sole discretion.
For accounting purposes, the group ownerships are regularly controlled and
in case needed reset.
