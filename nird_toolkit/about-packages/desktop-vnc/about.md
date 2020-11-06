## Usage
### What can the desktop-vnc package be used for?
The purpose of the desktop-vnc package is to provide a lightweight linux desktop environment    
for performing interactive tasks involving graphical applications and to speed-up their display on local desktop machines.   
Note: 3D hardware acceleration is not supported. So it is not suitable for use with heavy 3D rendering application.   


### How to add new packages
In case you are missing some packages from the default application image, you can add those packages yourself by creating a custom docker image.
See   {ref}`this tutorial <custom-docker-image>` for generic instructions on how to add packages.

After having read the tutorial above, you can use the dockerfile below as a starting point when creating the dockerfile that adds new packages.
```
# See the value of dockerImage in
#
#   https://github.com/Uninett/helm-charts/blob/master/repos/stable/desktop-vnc/values.yaml
#
# to determine the latest base image

FROM quay.io/uninett/desktop-vnc:<use latest tag here>

# Install system packages
USER root
RUN apt update && apt install -y some_package
USER vncuser
```
