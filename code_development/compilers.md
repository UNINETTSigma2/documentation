# Compilers

## Compiling MPI Applications

### Intel MPI

The following table shows available Intel MPI compiler commands, the underlying Intel and GNU compilers, and ways to override underlying compilers with environment variables or command line options:

<table>
<thead>
<th align="left">Language</th>
<th align="left">Wrapper script</th>
<th align="left">Default compiler</th>
<th align="left">Environment variable</th>
<th align="left">Command line</th>
</thead>
<tbody>
<tr>
<td rowspan="2" >C</td>
<td><code class="code">mpiicc</code></td>
<td>`icc`</td>
<td rowspan="2">I_MPI_CC</td>
<td rowspan="2">`-cc`=&lt;compiler&gt;</td>
</tr>
<tr>
<td>`mpigcc`</td>
<td>`gcc`</td>
</tr>
<tr>
<td rowspan="2">C++</td>
<td>`mpiicpc`</td>
<td>`icpc`</td>
<td rowspan="2">I_MPI_CXX</td>
<td rowspan="2">`-cxx`=&lt;compiler&gt;</td>
</tr>
<tr>
<td>`mpigxx`</td>
<td>`g++`</td>
</tr>
<tr>
<td rowspan="2">Fortran 77 / Fortran 95</td>
<td>`mpiifort`</td>
<td>`ifort`</td>
<td rowspan="2">I_MPI_FC</td>
<td rowspan="2">`-fc`=&lt;compiler&gt;</td>
</tr>
<tr>
<td>`mpifc`</td>
<td>`gfortran`</td>
</tr>
</tbody>
</table>

Specify option `-show` with one of the compiler wrapper scripts to see the underlying compiler together with compiler options, link flags and libraries.
