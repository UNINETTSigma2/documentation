# We ignore ANSYS CFX and Fluent links as they give
# https://github.com/curl/curl/issues/4409, which needs a
# rather new version of OpenSSL.
# The cytopia/linkcheck is rather slow. There are faster alternatives,
# especially https://github.com/filiph/linkcheck, but that
# requires Dart etc.
# Consider to utilize a faster link checker as soon as we have
# all the exclude links configured and it works.
git clone https://github.com/cytopia/linkcheck.git /tmp/linkchecker && /tmp/linkchecker/linkcheck -i '^http(s)?:\/\/(localhost)|(127.0.0.1)|(documentation.sigma2.no/page/on/same/site.html)|(download.open-mpi.org/release/open-mpi/v4.0/openmpi-)|(example.org/institution/simulationDataq)|(rt.uninett.no/SelfService)|(desktop.saga.sigma2.no)|(desktop.fram.sigma2.no)|(example.org/institution/simulationData)|(www.pythonware.com/products/pil)|(www.linuxconfig.org/Bash_scripting_Tutorial)|(https://documentation.sigma2.no/_downloads/bdfbca90a90a8d1b824fc6b1154ceee7/serial.zip)|(https://www.ansys.com/products/fluids/ansys-cfx)|(https://www.ansys.com/products/fluids/ansys-fluent)' -e 'md,txt,rst' -k -c '200,301,302,303,307,308' .
