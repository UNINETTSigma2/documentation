# Licence and access policies

```{note}
This page is about software for academic use, thus the information here may not hold for users from research institutes and commercial companies!
```

Access to Software and Scientific Applications on NRIS installations is almost allways regulated by license and access agreements based on either usage (scale/time/academic or commercial), the user's affiliation to the machine-hosting institution or the users´s affiliation to an academic institution (or, of course, all subsets of the above). In general, license access restrictions typically fall within one of the following categories: 

1. Academic license - generally available to all academic users based in Norway.
   There may be application specific limitations and/or definitions, please allways check the license agreement(s) when starting using new software. 

2. Open-source license - users are allowed to freely use and distribute the
   library or tool with some restrictions. Some examples include the [GNU
   Public License](https://www.gnu.org/licenses/) and the [MIT
   License](https://mit-license.org/). A good overview of open source licenses
   can be found [at the Open Source
   Initiative](https://opensource.org/licenses).

3. Commercial license - may be granted to an individual, a project, a scientific group, a center or parts of/whole organization. See {ref}`the list below <license-list>` for available licenses at NRIS.
   
4. Free Proprietary Software - users are allowed to use the software if they have signed an agreement with the developer community/vendors. Example of such codes are the chemistry code ORCA and the physics code CASTEP. In these cases, if to install the users themselves have to provide access to installation files and tools, and new user groups needs to documente a valid agreement with the software providing entity to get access. Alternatively, an NRIS representative may make this agreement on behalf of the community - but the need for documenting access agreements is valid for this situation also.    

Note that it is allways the user's responsibility to make sure they adhere to the license
agreements. For some cases, NRIS has been delegated responsibility for limiting access to given codes unless users can prove their right to access these - for instance by documenting access to the license elsewhere. The necessary proof of access in these cases will inevitably vary, please check each case individually.   

## NRIS funding policy

Currently NRIS fund no scientific software, but for a number of cases users fund commercial software that runs on NRIS administered machines. A special case is the chemistry/material science code Gaussian, funded by the partner universities of NRIS (UiB, UiO, NTNU and UiT) as a joint agreement between them and Gaussian.inc - with Sigma2 the facilitator. Also, all of the BOTT partners holds MatLab licenses that are available on NRIS machines. 

```{note}
NRIS do still fund licenses for code development tools
(compilers, debuggers, profiling tools) (software that belongs to the field
tagged "Code Development").
```
All other software that demand license fees are thus funded by user communities themselves. 
However, there are a number of licenses that was purchased before the decision of not funding scientific software anymore which holds lifetime access. These are listed below.

Commercial software where NRIS users still have general access:

* **Amber** - the license is for release 11 of the code and is valid for the
  lifetime of the software. Currently this is not installed on any of our
  clusters. Please let us know if you need this.
* **Crystal** - the license is for release 14 of the code and is valid for the
  lifetime of the software. Currently this is not installed on any of our
  clusters. Please let us know if you need this.
* **Gaussian** - the license is paid for by the NRIS partner universities, thus users from UiB, UiO, NTNU and UiT automatically should get access to the code. Others who wants access would have to prove license access. 
* **NBO6/7** - the license is valid for the lifetime of the software, thus there will be no change until the release of major-version 8. 
* **Turbomole** - the license is valid for the lifetime of the software. We are allowed to use version 7.3 for 5 more years. Please let us know if you need this.

For more details, see the underlying {ref}`list <license-list>`. 

(license-list)=
## Commercial and Proprietary licenses at NRIS

| Software                    | Machines          | Available for whom                         | License type            | Field                      | Source of funding   |
|-----------------------------|-------------------|-------------------------------------------|-------------------------|----------------------------|---------------------|
| Abaqus                      | Betzy, Fram       | Members of license holding groups         | Group                   | Multiphysics/FEA           | Users               |
| AMS (ADF/BAND)					| Saga		| Members of license holding groups | Group | Chemistry/Material science | Users |
| Allinea MAP                 | Betzy, Fram, Saga | All users                                 | National HPC            | Code development           | National/Sigma2     |
| Allinea Performance Reports | Betzy, Fram, Saga | All users                                 | National HPC            | Code development           | National/Sigma2     |
| ANSYS                       | Fram              | Members of license holding groups         | Department/Group                   | Multi-physics              | Users               |
| ANSYS CFX                   | Fram              | Members of license holding groups         | Department/Group                  | CFD                        | Users               |
| ANSYS Fluent                | Betzy, Fram       | Members of license holding groups         | Department/Group                   | CFD                        | Users               |
| Gaussian                    | Fram, Saga              | UiB, UiO, NTNU, UiT users automatic  | Site for UiB, UiO, NTNU, UiT                   | Chemistry/Material science | Users |
| GaussView                   | Fram, Saga              | UiT users automatic | Site for UiT                   | Chemistry/Material science | Users |
| Intel Parallel Studio XE    | Betzy, Fram, Saga | All users                                 | National HPC            | Code development           | National/Sigma2     |
| MATLAB                      | Betzy, Fram, Saga | Members of institutions with site license | Site/Department         | Mathematics/Data analysis  | Users               |
| NBO6/7 | Fram, Saga | All academic users | National academic HPC | Chemistry/Material science | National/Sigma2     |
| STAR-CCM+                   | Betzy, Fram       | Members of license holding groups         | Group                   | Multi-physics/CFD          | Users               |
| TotalView                   | Betzy, Fram, Saga | All users                                 | National HPC            | Code development           | National/Sigma2     |
| Turbomole                   | Fram              | All users on given machine                | Computer Center  | Chemistry/Material science | National/Sigma2     |
| VASP                        | Fram, Saga        | Members of license holding groups         | Group                   | Chemistry/Material science | Users               |

**Table explanation:**

- License access limited on **group level**. Only members of the specific
  software group has access to the software.
- License access limited on **institutional level**. Either for all users from
  an institution (site) or for a limited subgroup of users from this
  institution (faculty/department/research-center/research-project).
- License access limited on **computing center level**. This means that all
  users have access to software on all machines administrated by this computing
  center.
  - A subset of this type is license access limited on **machine level**.
- License access limited on **national agreement level**. We currently have a
  couple of software packages with a multilateral agreement between the
  participants in the BOTT (Bergen-Oslo-Trondheim-Tromsø) collaboration and
  some software vendors. All users from all these four institutions will have
  access. We also have a couple of national agreements, which implies that all
  national users or all users with a certain access domestically will be
  allowed access.

**Users are responsible for having the correct credentials and agreements in
terms of license access**. For all software with access limitations on group-
and research-project level the credentials may have to be provided to NRIS
before being granted access to software installed on NRIS controlled machines.

## IPv4 and IPv6 addresses of Sigma2 HPC clusters

For a software to be able to communicate with its license server, they must be able to receive request(s) from one or more ip ranges below, depending on which machine you want to use a license.

| **Cluster name**  |   **IPv4 addresses**   |     **IPv6 addresses**    |------------------|----------------------|-----------------------------|
|     SAGA      |  158.36.42.32/28   | 2001:700:4a01:10::/64 |
                |  158.36.42.48/28   | 2001:700:4a01:21::/64 |
|     FRAM      |  158.39.114.64/27  | 2001:700:4a00:10::/64 |
|     BETZY     |  158.36.141.144/28 | 2001:700:4a01:25::/64 |
                |  158.36.154.0/28   | 2001:700:4a01:23::/64 |
                |  158.36.154.16/28  | 2001:700:4a01:24::/64 |
