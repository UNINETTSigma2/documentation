# Editing files

We highly recommend using the text mode when using Emacs on Fram, Saga
or NIRD, e.g.,

```sh
$ emacs -nw
```

If you wish to use the graphical user interface, then we recommend to run
Emacs on your local computer and open the file remotely.  For this you do not
need to copy the files from the cluster to your local computer, you simply open them
as you would open a remote web page in your browser. The procedure uses an
Emacs package called TRAMP (Transparent Remote (file) Access, Multiple
Protocol). See their web page https://www.gnu.org/software/tramp/ for more details.

Procedure for Fram (Saga and NIRD will be similar):
 * Open emacs on your laptop/machine
 * C-x C-f (<kbd>Ctrl</kbd>+<kbd>x</kbd>, then <kbd>Ctrl</kbd>+<kbd>f</kbd> (or Mac equivalent), then you will get a “find file” prompt)
 * ```/ssh:username@fram.sigma2.no:pathname``` (**note the leading slash**)
 * You may get the following message “Offending key for IP in …. Are you sure you want to continue connecting (yes/no)?“ type yes and enter
 * Depending on the network state you might see the message “Waiting for prompt from remote shell” for few seconds to a minute, before the connection opens.

For example if your user name is “newuser” and if you want to open a file called “myfile.txt”, located in your home area on Fram, you would use the following:

 * ```/ssh:newuser@fram.sigma2.no:/cluster/home/newuser/myfile.txt```

If you specify a directory name, you can browse the remote file system until you have found the file you wish to open.
