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
<td rowspan="2" align="left" valign="middle">C</td>
<td><code class="code">mpiicc</code></td>
<td><code class="code">icc</code></td>
<td rowspan="2">I_MPI_CC</td>
<td rowspan="2"><class="code">-cc</code>=&lt;compiler&gt;</td>
</tr>
<tr>
<td><code class="code">mpigcc</code></td>
<td><code class="code">gcc</code></td>
</tr>
<tr>
<td rowspan="2">C++</td>
<td><code class="code">mpiicpc</code></td>
<td><code class="code">icpc</code></td>
<td rowspan="2">I_MPI_CXX</td>
<td rowspan="2"><code class="code">-cxx</code>=&lt;compiler&gt;</td>
</tr>
<tr>
<td><code class="code">mpigxx</code></td>
<td><code class="code">g++</code></td>
</tr>
<tr>
<td rowspan="2">Fortran 77 / Fortran 95</td>
<td><code class="code">mpiifort</code></td>
<td><code class="code">ifort</code></td>
<td rowspan="2">I_MPI_FC</td>
<td rowspan="2"><code class="code">-fc</code>=&lt;compiler&gt;</td>
</tr>
<tr>
<td><code class="code">mpifc</code></td>
<td><code class="code">gfortran</code></td>
</tr>
</tbody>
</table>

Specify option `-show` with one of the compiler wrapper scripts to see the underlying compiler together with compiler options, link flags and libraries.
