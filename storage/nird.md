NIRD - National e-Infrastructure for Research Data
==================================================

The new data infrastructure, named **NIRD** (National Infrastructure for
Research Data), will provide storage resources with yearly capacity
upgrades, data security through geo-replication (data stored on two
physical locations) and adaptable application services, multiple storage
protocol support, migration to third-party cloud providers and much
more. Alongside the national high-performance computing resources, NIRD
forms the backbone of the national e-infrastructure for research and
education in Norway, connecting data and computing resources for
efficient provisioning of services.

The NIRD storage system consists of SFA14K controllers, 10TB NL-SAS
drives with a total capacity of 12PiB in addition to a centralized file
system (IBM GridScaler) supporting multiple file, block and object
protocols. Sigma2 will provide the storage infrastructure with resources
for the next 4 – 5 years through multiple upgrades and is expected to
triple in capacity during its life-time.

The NIRD infrastructure offers Storage services, Archiving services and
processing capacity for computing on the stored data.\
More info here

[Research Data](https://www.sigma2.no/content/data-storage)

Project data storage
--------------------

### Getting Access

To gain access to the storage services, a formal application is needed.
The process is explained at the [User
Access](https://www.sigma2.no/node/36) page.

### Logging In

Access to the Project data storage area is through front-end (login)
node:

    login.nird.sigma2.no

Note that this hostname is actually a DNS alias for:\
login1.nird.sigma2.no, login2.nird.sigma2.no, login3.nird.sigma2.no,
login4.nird.sigma2.no\
those are containers each one running the image of a login node.\
A login container offers resources for a maximum of 16 cpus and 128MB of
memory.

Users must be registered and authorized by the project responsible
before obtaining access.

To access or transfer data use the following tools: ssh, scp or stfp.
Visit the [Transferring
files](https://documentation.sigma2.no/storage/file-transfering.html)
page for details.

### Home directories

Home directories are located in `/nird/home/<username>`. Quota for home
is 20GB and 100000 files. To check the disk usage type

     dusage
     

Home directories are also visible at /nird/home from FRAM login nodes.\
Since those are mounted with NFS on FRAM it is very important not to use
home directories\
as storage for jobs running of FRAM.

### Project area

NIRD project areas are located in `/nird/projects/<project_ID>`.

The project area is quota controlled and current usage is obtained by
running the command:

    dusage -p <project_ID>

FRAM projects are only available from FRAM login nodes.\
For more information, visit the [Storage Systems on
Fram](storagesystems.md) page.

### File transfering

Access to NIRD is permitted only through SSH. One can use *scp* and
*sftp* to upload or download data from NIRD.

-   scp - secure copy files between hosts on a network

<!-- -->

    # copy single file to home folder on NIRD
    # note that folder is ommitted, home folder being default
    scp my_file.tar.gz <username>@login.nird.sigma2.no:

    # copy a directory to project area
    scp -r my_dir/ <username>@login.nird.sigma2.no:/projects/<projectname>/

-   sftp - interactive secure file transfer program (Secure FTP)

<!-- -->

    # copy all logs named starting with "out" from project1 folder
    # to /projects/project1
    sftp <username>@login.nird.sigma2.no
    sftp> cd /projects/project1
    sftp> lcd project1
    sftp> put out*.log

### Backup

-   Geo-replication is set up between Tromsø and Trondheim.
-   For backup, snapshots are taken with the following frequency:
    -   daily snapshots of the last 7 days
    -   weekly snapshots of the last 5 weeks.
-   See [Backup](backup.md).

Using WinSCP to access NIRD
===========================

To access data on NIRD using a graphical interface, you will need to
install and use specialized software. As an example, we will here
present how to use the program WinSCP to access NIRD. The first step
will be to download the program and install it. Thereafter we will look
at how to connect to NIRD. Finally we will show you how to download and
upload files to NIRD.

WinSCP is a Windows program. If you are on Linux or Mac you need to use
a different program, like for instance sftp, FileZilla, or Cyberduck.

Installation on Windows
-----------------------

1.  Download the program from <https://winscp.net/eng/download.php>.
2.  When the installation file is downloaded, double click on it and
    follow the instructions for installing the program.

Connecting to NIRD with WinSCP
------------------------------

0.  Make sure you have a NIRD user account. If you do not, you can apply
    for one at
    <https://www.metacenter.no/user/application/form/norstore/>. Once it
    has been accepted, might take a day, you can continue along the
    steps below.
1.  Start WinSCP. You should see something like this:

    ![WinSCP startup](images/WinSCP_start.jpg)

2.  Fill in these values
    -   Host name: login.nird.sigma2.no
    -   User name: *Your username*
    -   Password: *Your password*

3.  Save the configuration by pressing the *Save* button.
4.  Log in to NIRD by pressing the *Login* button
5.  If this is your first login, you should see your home folder at NIRD
    in the right column of the WinSCP window. Since data at NIRD is
    located in projects we need to navigate there.
6.  Use the dropdown menu:

    ![Dropdown](images/WinSCP_dropdown.jpg)

And navigate to the */ \<root\>* folder. 7. From the */* folder move
into the *projects* folder and select your project. 8. Navigate to the
desired place in the project folder.

Downloading and uploading files and folders
-------------------------------------------

When connected to a server (NIRD in this case), the WinSCP window shows
two file areas in the two large panels. The left panel is your local
computer, the right panel the server.

To download data, navigate to the folder on the server where the files
of interest are located. In the left panel, navigate to the place you
want to download the files to. Now you can either right click the file
or folder and select *Download*, or *drag and drop* the file or folder
from the right panel to the left panel. Follow the dialogue boxes and
wait for the download to finish. Usually, the standard suggestions
should be reasonable.

Uploading works similarly. Just navigate to the right folders on your
local computer and on the server and *drag and drop* a file or folder
from your local computer (left) to the server (right). Alternatively,
you can, again, right click on the file or folder of interest and choose
*Upload*. Follow the dialogue boxes and wait for the upload to finish.
Usually, the standard suggestions should be reasonable.

