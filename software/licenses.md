# Licence and access policies

Most applications on the HPC machines have their own license based on the usage
and on the user's affiliation of the user to the university hosting the
machines. In general, the application and tool licenses fall under these types:

1. Academic license - generally available to academic users based in Norway.
   There are other definitions that are specific to an application, therefore,
   check with the license agreement.

2. Open-source license - users are allowed to freely use and distribute the
   library or tool with some restrictions. Some examples include the [GNU
   Public License](https://www.gnu.org/licenses/) and the [MIT
   License](https://mit-license.org/). A good overview of open source licenses
   can be found [at the Open Source
   Initiative](https://opensource.org/licenses).

3. Commercial license - may be granted to an individual, project, or
   organization. See {ref}`the list below <license-list>` for available
   licenses at NRIS.

It is the user's responsibility to make sure they adhere to the license
agreements.

(license-list)=
## Commercial licenses at NRIS

| Software                    | Machines          | Available for whom                         | License type            | Field                      | Source of funding   |
|-----------------------------|-------------------|-------------------------------------------|-------------------------|----------------------------|---------------------|
| Abaqus                      | Betzy, Fram       | Members of license holding groups         | Group                   | Multiphysics/FEA           | Users               |
| Allinea MAP                 | Betzy, Fram, Saga | All users                                 | National HPC            | Code development           | National/Sigma2     |
| Allinea Performance Reports | Betzy, Fram, Saga | All users                                 | National HPC            | Code development           | National/Sigma2     |
| ANSYS                       | Fram              | Members of license holding groups         | Group                   | Multi-physics              | Users               |
| ANSYS CFX                   | Fram              | Members of license holding groups         | Group                   | CFD                        | Users               |
| ANSYS Fluent                | Betzy, Fram       | Members of license holding groups         | Group                   | CFD                        | Users               |
| Gaussian                    | Fram              | Only NTNU, UiB, UiO, and UiT users        | Site                    | Chemistry/Material science | NTNU, UiB, UiO, UiT |
| GaussView                   | Fram              | Only UiT users                            | Site                    | Chemistry/Material science | NTNU, UiB, UiO, UiT |
| Intel Parallel Studio XE    | Betzy, Fram, Saga | All users                                 | National HPC            | Code development           | National/Sigma2     |
| MATLAB                      | Betzy, Fram, Saga | Members of institutions with site license | Site/Department         | Mathematics/Data analysis  | Users               |
| STAR-CCM+                   | Betzy, Fram       | Members of license holding groups         | Group                   | Multi-physics/CFD          | Users               |
| TotalView                   | Betzy, Fram, Saga | All users                                 | National HPC            | Code development           | National/Sigma2     |
| Turbomole                   | Fram              | All users on given machine                | Computer Center license | Chemistry/Material science | National/Sigma2     |
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

## NRIS funding policy

Currently the cost for a number of licenses for application software is carried
by NRIS, as shown in the {ref}`list above<license-list>`. This will change
gradually, and from autum 2020 NRIS will not fund licenses for application
software (e.g. ADF/BAND, Crystal, Gaussian, GaussView, Molpro). For these
codes, and the corresponding research groups and research communities, NRIS
will offer consulting for pooling costs and for the maintenance of common
license certificates and servers. NRIS will also help these communities set up
coordination points to simplify paperwork and save costs.

**Note that NRIS will continue to fund licenses for code development tools
(compilers, debuggers, profiling tools) (software that belongs to the field
tagged "Code Development") - these base tools are not affected by the planned
change in funding policy.**

Access to and using of codes will be influenced the following way:

* **ADF/BAND** - the license is renewed every second year, and the software
  requires valid license to run. Software in the ADF/BAND software suite will
  cease to work on NRIS installations from September 1st 2020 unless
  alternatively funded.
* **Amber** - the license is for release 11 of the code and is valid for the
  lifetime of the software. Currently this is not installed on any of our
  clusters. Please let us know if you need this.
* **Crystal** - the license is for release 14 of the code and is valid for the
  lifetime of the software. There will be no change in access after September
  2020, but there will be no updates to newer versions unless alternatively
  funded.
* **Gaussian/GaussView** - the license is valid for the lifetime of the
  software, there will be no change in access after September 2020, but there
  will be no updates unless alternatively funded.
* **NBO6/7** - the license is valid for the lifetime of the software, there
  will be no change in access after September 2020, but there will be no
  updates unless alternatively funded.
* **Schrodinger** - license is renewed annually and the software requires a
  valid license to run. Most of the software in the Schrodinger suite, apart
  from the free releases of Maestro, PyMOL and Desmond, will cease to work on
  NRIS installations from November 2020 unless alternatively funded. Note that
  there is a gradual decrease of funding contribution from Sigma2, for
  2019/2020 there will be a 15% overall contribution (not more than 7k USD) and
  from 2020/2021 no contribuion at all. *Thus, if the user community decides to
  discontinue their contribution to the overall license costs, the renewal of
  the license may end already October 2019.*
* **Turbomole** - the license is valid for the lifetime of the software, paying
  an annual maintenance fee of about 10% of license cost. There will be no
  change in access to currently installed software after September 2020, but
  there will be no updates unless alternatively funded.

**Note also that even if there are alternative funding of some of the above
mentioned software, access policy may probably be altered since funding will
typically move from national/community to group level.**
