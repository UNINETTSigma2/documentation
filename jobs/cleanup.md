# How to recover files before a job times out

Possibly you would like to clean up the work directory or recover
files for restart in case a job times out. In this example we ask Slurm
to send a signal to our script 120 seconds before it times out to give
us a chance to perform clean-up actions.

[include](files/slurm-timeout-cleanup.sh)
