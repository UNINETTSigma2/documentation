#How can I, as a user, install software for myself or my project with EasyBuild?

Currently, the software-team in the Metacenter is using the [EasyBuild system](https://easybuild.readthedocs.io/en/latest/) for installing system-wide software and scientific applications. It is, actually, quite easy (hence the name) and straigt forward for users to do install with the same tool.

There are two distinct scenarios covered in this tutorial; first installations for single user only - placing the software in standard user home folder. Second, installations for a project/group - placing the softeware in a project folder for many to share. 

Note that the software install team at the Norwegian Metacenter **never** do installations in "$HOME" for users, and preferably not in /cluster/projects.

## Installing softeware in home-folder:

Log in to Fram your preferred way, then type

	module load EasyBuild/3.8.0 

Note that there is, currently at least, no default modules on Fram. Thus you need to be explicit. The easybuild version changes at least a couple of times a year, so do check what is the available version(s) by typing:

	module avail easybuild

Now, as of January 2019, you will see this:
	
	module avail EasyBuild

	----------------------- /cluster/modulefiles/all ------------------------
	   EasyBuild/3.8.0

	Use "module spider" to find all possible modules.
	Use "module keyword key1 key2 ..." to search for all possible modules
	matching any of the "keys".

Choose 3.8.0 in this case. Now, we advice to do an install in two steps, first download the sources of your software and then do the full install. Say you want to install [rjags 4.6](http://cran.r-project.org/web/packages/rjags), then you type:

	eb rjags-4-6-intel-2017b-R-3.4.3.eb --fetch

if this proves sucessfull, then type:

	eb rjags-4-6-intel-2017b-R-3.4.3.eb -r
	
Then the process should go absolutely fine, and you will receive a nice little message on the command line stating that installation succeeded. 

Now the software and the module(s) you installed are in a folder called ".local". You can inspect it by typing (note the path!)

	cd .local/easybuild

There you should see the following:

	build  ebfiles_repo  modules  software  sources

## Installing software in project folder:

Do as described above regarding login, loading of the EasyBuild module and considerations regarding what to install. 

Then do as follows:

	mkdir -p -prefix=/cluster/projects/nnXXXXk/easybuild 	eb rjags-4-6-intel-2017b-R-3.4.3.eb --fetch --prefix=/cluster/projects/nnXXXXk/easybuild

where XXXX is your project id number. When a suksessfull download of sources is made, then type:

	eb rjags-4-6-intel-2017b-R-3.4.3.eb --prefix=/cluster/projects/nnXXXXk/easybuild

Note the easybuild folder in the path, this is a tip for housekeeping and not strictly required. This will give the path structure as for the local case, with the software and modulefiles installed in **cluster/projects/nnXXXXk/easybuild**. 

###For more advanced settings:

Please check upon options with

	eb --help

or read up on the [EasyBuild documentation](https://easybuild.readthedocs.io/en/latest/) on web. 

## Using software installed in non-standard path:

The default path for modulefiles only contains the centrally installed modules. Thus, if you want the modulefilesystem to find the software you installed either for your own usage or on behalf of the project group, you need to make the module-system aware of alternative paths. 

**For the case of install in user-home: (still using the rjags example)**

	module use .local/easybuild/modules/all 
	module avail rjags #Just to check if it is found
	module load rjags/4-6-intel-2017b-R-3.4.3
	
	
**For the case of installing i group/project folder:**

	module use /cluster/projects/nnXXXXk/easybuild/modules/all
	module avail rjags #Just to check if it is found
	module load rjags/4-6-intel-2017b-R-3.4.3
		
**For more information about the module system, please see:** <https://lmod.readthedocs.io/en/latest/>

