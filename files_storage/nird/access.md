# Access and login


## Getting access

To gain access to the storage services, a formal application is required. The
process is explained at the
[How to apply for a user account](https://www.sigma2.no/how-apply-user-account)
page.

Users must be registered and authorised by the project responsible
before getting access.

To access or transfer data, you may use the following tools: `ssh`, `scp` or
`sftp`.  Visit the {ref}`file-transfer` page
for details.


## Logging in

Access to your `$HOME` on NIRD and the project data storage area is through the
login containers.
Login containers are running on servers directly connected to
the storage on both sites -that is Tromsø and Trondheim- to facilitate data
handling right where the primary data resides. Each login container offers a
maximum of 16 CPU cores and 128GiB of memory.

Login containers can be accessed via following addresses:
```
login-tos.nird.sigma2.no
login-trd.nird.sigma2.no
```

```{note}
We run four login containers per site.

If you plan to start a `screen` session on one of the login containers or
you wish to copy data with the help of `scp` or `WinSCP`, you should log in
to a specific container.

Addresses are:
- login**X**-tos.nird.sigma2.no
- login**X**-trd.nird.sigma2.no
- **X** - can have values between 0 and 3.
```
