# Glossary
### Package
A package represents a configuration of an application.
For instance, installing the Jupyter package will create a
Jupyter notebook (i.e. the application) that you can access through the
web browser.
You may sometimes see packages referred to as a chart. A package is an
extension of a chart.

Several different packages are available. There are for
instance packages for creating an Apache Spark cluster, setting up a GPU
enabled deep-learning environment, and creating a personal cloud storage
server.

### Application
An application represents a specific installation of a package. That is,
it encapsulates the configuration specified by the user when installing
the application.

### Projectspace
Each application belongs to a single projectspace. A projectspace is essentially a
way of grouping applications together. Every projectspace is allocated a given
amount of resources (that is, RAM, CPUs and GPUs) and volumes.

Note: if you are familiar with Kubernetes namespaces, then it is worth noting
that a projectspace is just a different name for a namespace.
