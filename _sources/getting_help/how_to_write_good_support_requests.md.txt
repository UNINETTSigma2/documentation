
# Writing good support requests

Writing descriptive and specific support requests helps the support team
understand your request quicker. Below is a list of good practices.


## Create a ticket

Send an email to [support@metacenter.no](mailto:support@metacenter.no) to create a support ticket. Tickets
are tracked and have higher visibility. Everyone in the support team can see
the tickets and respond appropriately.


## Create a ticket for each issue

Creating a ticket for separate issues ensures that each issue is given the
appropriate priority and can be tracked easily. Adding a new issue to a
resolved or unrelated issue diminishes its visibility.


## Give descriptive and specific subject line

The subject line should be descriptive and specific to the issue. "Problem on
Fram" is not descriptive enough and does not differentiate itself from other
issues.


## Specify the environment and intent

Describe the system environment such as which modules and build environment
were used. Details such as compilers and script commands are also important to
write in the support mail. The support team can then replicate the environment
and reproduce the issue.


## Tell us what has been done

Tell us what actually worked so far and what was attempted to solve the issue.
Often we get requests of the type "I cannot get X to run on two nodes". The
request does not mention whether either or both has ever worked or if this was
the first attempt.


## Create an example which reproduces the problem

Create an example that demonstrates the problem. Examples should be easy to set
up and run, otherwise, it is time consuming if the support team needs to
diagnose the issue with only a description. Make sure that we can run the
example. Note that the support team does not access read-protected files
without your permission.

Try to reduce the example so that the support team encounters the issue
quickly. It is easier to schedule and debug a problem which crashes after few
seconds compared to problem that happens after a few hours.


## Please send us full paths to examples

Instead of telling us that the example can be found in `~/myexample/` it is
much easier for us if you give us the full path, e.g.
`/home/myuser/myexample/`.
Use `pwd` to get the full path for your current folder.

The reason is that we don't know where `~` points to in your case. We have
hundreds of users and we do not remember usernames. For the staff `~` will
point to a different place (their home folder) and we will have to look up your
username and it's an extra step that we would prefer to avoid.


## Describe the original problem and intent (The XY problem)

Often we know the solution but we don't know the problem. Please read
<http://xyproblem.info> which happens when a user's original issue is masked
by a different problem.

In short (quoting from <http://xyproblem.info>):

-   User wants to do X.
-   User doesn't know how to do X, but thinks they can fumble their way
    to a solution if they can just manage to do Y.
-   User doesn't know how to do Y either.
-   User asks for help with Y.
-   Others try to help user with Y, but are confused because Y seems
    like a strange problem to want to solve.
-   After much interaction and wasted time, it finally becomes clear
    that the user really wants help with X, and that Y wasn't even a
    suitable solution for X.

To avoid the XY problem, if you struggle with Y but really what you are
after is X, please also tell us about X. Tell us what you really want to
achieve. Solving Y can take a long time. We have had cases where after
enormous effort on Y we realized that the user wanted X and that Y was
not the best way to achieve X.
