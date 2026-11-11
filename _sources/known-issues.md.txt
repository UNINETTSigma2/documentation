---
orphan: true
---

(known-issues)=
# Known issues

For latest changes and events please follow the
[OpsLog page](<https://opslog.sigma2.no>)
which tracks a lot over changes and events.

This "Known issues" page on the other hand is meant to list more longer-term
problems that we know about and either work on or where we recommend
work-arounds, so that you don't have to scroll the
[OpsLog page](<https://opslog.sigma2.no>)
weeks back to find a known issue.


- **Issues with software license servers**: if you are experiencing issues with 
  license, check first that your license server is reachable. You can do this by
  running `nc -vz ip.address port`` following (both on `login` and `compute`nodes),
  for example (either / or):
  ```console
    nc -vz 192.168.91.99 9898
    nc -vz license-server.no 9898 
  ```
  where `192.168.91.98` is the license server address`, and `9898` is the port
  (license servers are proveded in `port@address` format). If you are not able to
  reach the server, you must contact your license provider/local IT-dep. But If 
  you are able to contact the server, then {ref}`contact our support team <support-line>` 
  and include the terminal output (check also `tracepath 192.168.91.99 -p 9898` to see
  where in the route it might be failing).

- **Email notification from completed Slurm scripts is currently disabled** on all
  machines and it looks like it will take quite a while (months?) before we can
  re-enable it. Sorry for the inconvenience. The reason is technical due to the
  way the infrastructure is set up. It is non-trivial for us to re-enable this in
  a good and robust and secure way.

- **InfiniBand problems on Betzy**: There is a problem with high core count
  jobs failing due to an InfiniBand problem which emits messages of the "excess
  retry". This is an ongoing problem, but shows up (more) on high core count
  jobs.

- **Running jobs hang and consume compute resources**: on Betzy
  there is a randomly occurring problem that results in Zombie / unkillable
  processes. Amongst others, this happens when some of the application processes
  execute `MPI_Abort`, or otherwise crash while other ranks are performing MPI communication.
  With Intel MPI, this often results in the job hanging forever, or until it runs out
  of the SLURM allocated time. At this stage, to avoid this issue users should
  either make sure that all ranks call `MPI_Abort` at the same time (which might
  of course be impossible), or use OpenMPI. In the latter case, although the Zombie
  processes might also be created, we believe this does not result in a hanging
  application and waste of compute resources.

- **Slow performance with netCDF4 files**: Few users have reported this and this
  seems to be related to the problem described [here](https://github.com/Unidata/netcdf-c/issues/489).
  A solution seems to be to convert the files from the netcdf-4 format to the
  netcdf-64bit offset format with the command `$ nccopy -k nc6 file1 file2`.

