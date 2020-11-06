(custom-docker-image)=

# Using a custom docker image to add new packages
In case you are missing some packages from the default application image, you can add those packages yourself.
To do so, you need to create a custom docker image that use the default application image as the base image.
[This tutorial](https://docs.docker.com/get-started/part2/) shows how to build a docker image and push it to a container registry.

See the documentation of the specific package (ex. Jupyter) you want to add packages to for information about what the base dockerfile should look like.

Typically, the dockerfile looks similar to the following
```
# The image to use as a base image
FROM quay.io/uninett/example-docker-image:20180102-xxxxxxx

# Install system packages
USER root
RUN apt update && apt install -y vim

# Install other packages
USER notebook
RUN pip install scikit-learn
```

You need to have this image pushed to a public repository e.g. [Docker hub](https://hub.docker.com/) or [Quay Registry](https://quay.io/).
Once pushed, you can use the docker image by specifying the `dockerImage` under `Show advanced configuration` button on the `Installation/Reconfigure` page.
Note that the exact name of the field may very, but the field name should end with `Image` (ex. `workerImage`, `userImage` etc.).
After specifying your custom image and applying those changes, your image will be used in the given instance of application and have all the newly added packages.
