# Installing perl packages

Many Perl packages have numerous dependencies, and different modules can have conflicting
dependencies.  The current 'best practices' regarding Perl modules is therefore to install your own
modules, and even have separate modules installed for different projects/programs.  Thus, on our HPC
systems, we only install a small set of modules that are heavily used, or are compiled against
special versions of libraries.

It is easy to install your own Perl modules.  Most Perl modules are available on
[`CPAN`](https://metacpan.org/). This section describes how you can install Perl modules from CPAN
in your home directory.

```bash
# First load an appropriate Perl module (use 'module avail Perl' to see all)
module load Perl/5.32.0-GCCcore-10.2.0
# Make Perl install Perl modules in your $HOME:
eval $(perl -Mlocal::lib)
# `PERL_CPANM_HOME` is the directory where cpanm builds the packages, not where they are installed
export PERL_CPANM_HOME=/tmp/cpanm_$USER
# Install for example Perl module Chess:
cpanm Chess
```

#### Other useful commands:
```bash
# Check if the Perl module installed correctly:
perl -MChess -e ’print "ok\n"’
# Check all installed Perl modules:
instmodsh
cmd? l
# Check where a Perl module is installed:
cmd? m Chess
d
# Uninstall a Perl module:
cpanm --uninstall Chess
```

#### Advanced Options

Install from local file:

```bash
wget http://www.cpan.org/authors/id/M/MA/MAKAMAKA/JSON-2.90.tar.gz
cpanm JSON-2.90.tar.gz
```

Install to a separate directory (good for keeping projects separate or have in a shared directory):

```bash
eval $(perl -Mlocal::lib=/the/path)
cpanm Some::Module
```
