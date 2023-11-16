# ParaView using X11 Forwarding

## X Server for running the application

You first need to download an X server so the GUI can be forwarded and you can interact with ParaView.
**For Windows, you can use Xming or VcXsrv:** If you use the latter, select "One large window", "Start no client", uncheck "Native opengl" and check "Disable access control"
**For Mac, you can use XQuartz**

More information here: https://documentation.sigma2.no/getting_started/ssh.html#x11-forwarding and here: https://documentation.sigma2.no/jobs/interactive_jobs.html#graphical-user-interface-in-interactive-jobs

Make sure the application is running and it says, when you hover the mouse over it: "nameOfTheMachine:0.0"


## Running SSH with forwarding capabilities

### Windows PowerShell
Open Windows PowerShell and run the following commands:
- `$env:DISPLAY = "localhost:0"`
- `ssh -X -Y username@server.sigma2.no` (replace "server" with fram, betzy or saga)

In case the connection is not very stable while running with PowerShell, you can try with Putty

### Putty
- Install the software from https://www.putty.org/
- On "Session" tab, under "Host Name", write down `betzy.sigma2.no` (or fram or saga)
- On "Connection" tab, write 240 on "Seconds between keepalives". Also enable "Enable TCP keepalives (SO_KEEPALIVE option)"
- On "SSH > X11" tab, check "Enable X11 forwarding" and write down on "X display location": localhost:0.0
- Go back to the "Session" tab, write a name for the session under "Saved Sessions" and click "Save"
- Click "Open" and log in normally


## Allocating resources for the project

Run the following command: ```salloc --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --time=00:30:00 --qos=devel --account=nnxxxxk```
If the command above doesn't work, take a look at this [documentation](https://documentation.sigma2.no/jobs/interactive_jobs.html#requesting-an-interactive-job).

Please, note that here we are asking 1 CPU only for 30 minutes in the Devel queue. **If you need more resources and time, adjust the parameters accordingly.**

The output will be similar to this one:

```
salloc: Pending job allocation 5442258
salloc: job 5442258 queued and waiting for resources
salloc: job 5442258 has been allocated resources
salloc: Granted job allocation 5442258
salloc: Waiting for resource configuration
salloc: Nodes c84-5 are ready for job
```


## Running ParaView

Run the following commands:
```ml avail | grep ParaView
module load ParaView/versionDesired (replace "versionDesired" with the options available)
paraview```

The ParaView user interface should load on the X Server within a few seconds.
