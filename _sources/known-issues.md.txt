# Known issues

For latest changes and events please follow the
[OpsLog page](<https://opslog.sigma2.no>)
which tracks a lot over changes and events.

This "Known issues" page on the other hand is meant to list more longer-term
problems that we know about and either work on or where we recommend
work-arounds, so that you don't have to scroll the
[OpsLog page](<https://opslog.sigma2.no>)
weeks back to find a known issue.

- **Accessing NIRD Toolkit services**: Due to a recent change made by Feide in
  response to new national directives in the sector, all services are now
  opt-in. This means that when you try to access a service in the Toolkit, you
  may get a message stating that the service is not activated thus preventing
  you from accessing it. We have documented {ref}`a temporary workaround <service-not-activated>` for this.

- **Email notification from completed Slurm scripts is currently disabled** on all
  machines and it looks like it will take quite a while (months?) before we can
  re-enable it. Sorry for the inconvenience. The reason is technical due to the
  way the infrastructure is set up. It is non-trivial for us to re-enable this in
  a good and robust and secure way.

- **InfiniBand problems on Betzy**: There is a problem with high core count
  jobs failing due to an InfiniBand problem which emits messages of the "excess
  retry". This is an ongoing problem, but shows up (more) on high core count
  jobs.
