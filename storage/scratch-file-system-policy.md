# IMPORTANT

 - Scratch space for each user MUST go under /cluster/work/user/$USER.
 - The /cluster file-system is a scratch file-system without any backup and files
are subject to automatic deletion based on the policies below!


## POLICIES
-------------------------------------------------------------------------------

 - /cluster file system structure
   o /cluster/bin:		locally developed shared scripts (i.e. cost)
   o /cluster/installations:
   o /cluster/shared:
   o /cluster/software:		deployed software available and compiled only
                                for Fram
   o /cluster/tmp:
   o /cluster/work:		shared folder - SUBJECT FOR AUTOMATIC DELETION
     * /cluster/work/jobs:	scratch space for each job, automatically
                                created and deleted by the queue system
     * /cluster/work/users:	semi-permanent scratch space for each user
                                that will be cleaned up following specific
                                deletion rules

 - Automatic deletion policies for /cluster/work
   o Deletion depends on newest of the creation-, modification- and access time
     and the total usage of the file system.
   o The oldest files will be deleted first.
   o Weekly scan removes files older than 42 days.
   o When file system usage reaches 70%, files older than 21 days become
     subject for automatic deletion.
