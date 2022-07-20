# NIRD migration user guide    

This page is an information guide for the process of migrating data 
from old NIRD to new NIRD Storage system which is set in production
over summer 2022. 

The NIRD is redesigned for the evolving needs of Norwegian researchers 
and has been procured through 
[the NIRD2020 project](https://www.sigma2.no/procurement-project-nird2020).
 You can find the status updates in the projects page 
[here](https://www.sigma2.no/procurement-project-nird2020).

Note that the project page will be continuously updated as new information 
becomes available. Hence we recommend that you check this page frequently.
Additionally, the most important messages will be communicated by e-mail.

[The preparation for operation working group (PoWG) migration team](https://www.sigma2.no/be-ready-migrate) will 
take care of your data and services throughout the transfer to effectively 
migrate during the first quarter of 2022. However, NIRD project leaders/executive 
officers(PI/XO) and NIRD users are requested to cooperate with the PoWG team to select,
 prepare and verify the data that need to migrate.  This is crucial to facilitate 
the migration of important data.

```{note}
User home ($HOME) migration is user's responsibility.

```
Below you will find the guidance to prepare and par down the data before
the migration. 


## What can you do as a project leader?

  - Review the members of the project in MAS (https://www.metacenter.no)
  - Make sure that there is no orphan data that you no longer need
  - Prioritize your data for the migration
  - If you have any questions, contact the PoWG team by sending 
email to nird-migration@nris.no
  - Communicate and co-ordinate with your project members and urge them to follow the
 steps listed below


## What can you do as a NIRD user?

  - Delete the data you no longer need 
  - Make sure that you don’t have duplicate files/folders, delete the unneeded copy    
  - Make sure that there are no zero-length files or empty folders  
  - Try to compress large number of small size files to a single tar file
  - If you have any questions, contact the PoWG team by sending 
email to nird-migration@nris.no 
  - Please see the Frequently Asked Questions below to find the necessary commands. 


## Frequently asked questions


### How do I find zero length files?

 Use the following command.

```console
$ find -type f -empty
```


### How do I find empty directory?

 Use the following command.
```console
$ find -type d -empty
```   

### How do I compress large number of small files to a single file?

Use `tar` command. See different options on man pages `man tar`.

For example move all the files that needs to archive, into a single directory,
 and compress it using 
the following command.

```console
$ tar -czvf name_archive.tar.gz dirname
```
  - `-c`: Create an archive
  - `-z`: Compress the archive with gzip/ use `–j` for bzip compression
  - `-v`: Display progress in the terminal while creating the archive,
 also known as “verbose” mode. The v is always optional in these commands, but it’s helpful.
  - `-f`: Allows you to specify the filename of the archive.
  - Remember to delete the directory after successful compression

### How do I find out cold/not used data since a long time?

  Use the following command to find files and directories which were not accessed/modified since last six months.

- for files

```console
$ find . -type f -mtime +180 -atime +180
```         
- for directories 
```console
$ find . -type d -mtime +180 -atime +180
```  
 - `atime +180` is accessed time 180 days
 - `mtime +180` is modified time 180 days 
