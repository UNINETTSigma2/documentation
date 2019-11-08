# How to recover files before a job times out

Possibly you would like to clean up the work directory or recover
files for restart in case a job times out.  This is perhaps most
useful when using the `$SCRATCH` work directory (see [Storage
Areas](../storage/clusters.md)).

In this example we ask Slurm to send a signal to our script 120
seconds before it times out to give us a chance to perform clean-up
actions.

[include](files/timeout_cleanup.sh)

Download the script: <a
href="files/timeout_cleanup.sh">timeout_cleanup.sh</a> (you might have
to right-click and select `Save Link As...` or similar).

Also note that jobs which use `$SCRATCH` as the work directory can use
the `savefile` and `cleanup` commands to copy files back to the submit
directory before the work directory is deleted (see [Work
Directory](work_directory.md)).
