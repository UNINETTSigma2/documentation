# The GAUSSIAN program system

Gaussian is a computational chemistry software program system initially released in 1970. Gaussian has a rather low user threshold and a tidy user setup, which, together with a broad range of possibilities and a graphical user interface (gaussview), might explain its popularity in academic institutions.

## Online info from vendor

* Homepage: http://www.gaussian.com
* Documentation: http://gaussian.com/man

## Running Gaussian on Fram

To see available versions when logged into Fram issue command

    module avail Gaussian
    
To use Gaussian, type

    module load Gaussian/<version>
    
specifying one of the available versions.

* Run script example:

[include](files/run_g16.sh)

Download run script example here: <a href="files/run_g16.sh" download>run_g16.sh</a>

* Water input example:

[include](files/water.com)

Download water input example: <a href="files/water.com" download>water.com</a>


### Important aspects of Gaussian setup on Fram:

On Fram, we have not allocated swap space, that means the heap size for the linda processes in Gaussian is very important for making parallell jobs work. The line 

	export GAUSS_LFLAGS2="--LindaOptions -s 20000000"

contains info about the heap size. 20 GB (the number above) is sufficient for most calculations, but if you plan to run 16 nodes or more, you may need to increase this number to at least 30 GB. 

As for the Gaussian setup on Stallo, we have "tricked" the software to use rsocket libraries and Infiniband network, this is done by introducing explicit IB network addresses into input files. This is done with a wrapper around the gXX executable, called gXX.ib. This wrapper does two things; first it introduces explicit IB network adresses into input file, secondly it distributes the jobs onto two linda processes per node and halves the processes per linda compared to processes per node. 

Syntax is shown here:


	g16.ib $input.com > g16_$input.out


Please inspect the run script example carefully before submitting Gaussian jobs on Fram!


## About Gaussian on Fram:

After thorough testing, we would generally advice user to run 4-8 nodes with heap-size 20 GB (see above) and 2 Linda instances per node for running Gaussian on Fram. Note that due to the "two-Lindas-per-node" policy, memory demand is approximately the double as similar jobs on Stallo. 

Please consider the memory size in your input if jobs fails; the ```%mem``` number. Job example is set up with 500MB, test-jobs were ran with 2000MB. Memory demand also increases with an increasing number of cores, for jobs with 16 nodes or more - doubling the ```%mem``` number would be advicable. But this would also limit the size of problems possible to run at a certain number of cores. 

## Citation

When publishing results obtained with the software referred to, please do check the developers web page in order to find the correct citation(s).