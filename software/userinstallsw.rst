Installing software as user
===========================

Our software team is supporting a quite extensive software stack
of the most commonly used tools and applications within our user base.
However, due to our wide range of users, it is not feasible for us to maintain
all the different packages with particular versions and toolchains that you
might depend on in your daily work flow. While *system-wide* installations can
only be done by the Metacenter personnel, there is nothing preventing users
from installing their own software *locally*.

**If you have a software need that is currently not available on our systems:**

1.  If you think the package will be useful for a wider scope of users, you can
    request a global installation by sending an email to support@metacenter.no.
2.  Otherwise you can try to install it yourself locally under your home or
    project area. If you run into problems, we are happy to help out, just send
    a support request to support@metacenter.no.

There are several different ways to install the software, depending on the
package. Building from source is often the best when it comes to customization
options and performance, but this can be quite complicated. The procedure is
usually described in the particular software's documentation, so please refer
to their respective home pages for this. Below we will describe some of the
more standardized ways of installing software using EasyBuild and Python
package managers.

.. toctree::
   :maxdepth: 1

   userinstallsw/easybuild.md
   userinstallsw/python.md
   userinstallsw/R.md

Note that the software install team **never** do installations in
``$HOME`` for users, and preferably not in ``/cluster/projects``.
