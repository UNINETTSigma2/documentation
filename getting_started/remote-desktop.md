# Remote desktop

## Introduction

The remote desktop service makes it possible to run graphical applications on the Fram system with reasonable performance over the network. The service provides a simple Linux desktop based on the very lightweight XFCE desktop environment (<https://www.xfce.org/>). Currently (Aug. 2018) the system has no hardware acceleration so running advanced 3D rendering applications will probably not have adequate performance.

## Neccessary client software

Although it is possible to run the desktop in a web-browser (<https://desktop.fram.sigma2.no:6080> or <https://desktop.saga.sigma2.no:6080>)it is recommended to use a VNC client as it gives better performance and user experience. The recommended VNC client is TigerVNC which can be downloaded from <https://tigervnc.org/> (Many Linux distros have tigervnc in their software repos).

## Using the service

Start TigerVNC and give `desktop.fram.sigma2.no:5901` or `desktop.saga.sigma2.no:5901` as the server to connect to. You will then be presented with a graphical login window where you can fill in username and password on Fram.

## Short video tutorial.

Direct link <https://www.youtube.com/watch?v=tjOQ39DRUdc>

<iframe width="560" height="315" src="https://www.youtube.com/embed/tjOQ39DRUdc?rel=0" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

## Troubleshooting

### Cannot connect to server

The service is blocked outside the academic network in Norway, e.g. UNINETT, universities and colleges. It is possible to use ssh-tunneling to connect from the outside world:

#### Windows with Putty

Open cmd.exe to get a DOS prompt and run

```shell
plink.exe -L 5901:localhost:5901 USERNAME@desktop.fram.sigma2.no
```

and use `localhost:5901` as the server address. (The space after -L must be there) If you want to avoid typing you can create a .bat script with the correct plink command. ([Example](./ssh-tunnel-fram.bat) edit in Notepad and save it to a location where you can click on it, e.g the desktop. DO NOT USE WORD TO EDIT THIS!)

#### Linux/MAC/Windows with OpenSSH

Run the command

```shell
ssh -L5901:localhost:5901 USERNAME@desktop.fram.sigma2.no
```

(or saga) and use `localhost:5901` as the server address.

#### Incompatible VNC-clients

As stated above we recommend to use the TigerVNC client for performance and security reasons. If you cannot use this client you can connect to the remote desktop using another port number, 6901, on fram that is blocked by the firewall and only allows connections from within the Fram cluster:

```shell
ssh -L6901:localhost:6901 USERNAME@desktop.fram.sigma2.no
```

Then you can connect to `localhost:6901` with your VNC-client.

### Slurm jobs with DISPLAY export (`srun -x`) doesn't work

A relogin to localhost with display export helps. In a terminal window run

```shell
ssh -X localhost
sbatch -x .......
```

We have no idea what is the cause of the problem or why the workaround helps. Just some Linux magic to please the system gods...
