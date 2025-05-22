# Contributing to `ni-rf-data-recording-api` 

Contributions to `ni-rf-data-recording-api` are welcome from all!

`ni-rf-data-recording-api` is managed via [git](https://git-scm.com), with the canonical upstream
repository hosted on [GitHub](https://github.com/ni/<reponame>/).

`ni-rf-data-recording-api` follows a pull-request model for development.  If you wish to
contribute, you will need to create a GitHub account, fork this project, push a
branch with your changes to your project, and then submit a pull request.

Please remember to sign off your commits (e.g., by using `git commit -s` if you
are using the command line client). This amends your git commit message with a line
of the form `Signed-off-by: Name Lastname <name.lastmail@emailaddress.com>`. Please
include all authors of any given commit into the commit message with a
`Signed-off-by` line. This indicates that you have read and signed the Developer
Certificate of Origin (see below) and are able to legally submit your code to
this repository.

See [GitHub's official documentation](https://help.github.com/articles/using-pull-requests/) for more details.

# Getting Started

To contribute to this project, it is recommended that you follow these steps:
- Create a branch by forking this repository on GitHub.
- Ensure that you can run the fork on your system. If there are failures and you cannot resolve them, you can report issues through our [GitHub issues page](http://github.com/ni/ni-rf-data-recording-api/issues).
- Make your changes.
- Test the code after your changes.
- Send a GitHub Pull Request to the main repository's `main` branch. GitHub Pull Requests are the expected method of code collaboration on this project. Your Pull Request should have the following info:
    - **Summary**: Briefly summarize your changes and the motivation behind them in one or two sentences.
    - **Detailed Description**: Provide information about the nature of your changes.
    - **Affected Components**: List the core components of the system that are affected by your changes.
    - **Additional Testing/Validation**: Please specify whether there were additional (manual) tests done. Please also specify the configurations/systems used to run these tests.
    - **Checklist**: 
        - [ ] Revert Scripts without functional changes.
        - [ ] (Python) Reformat your code using Black.
        - [ ] (Python) Run Flake8 to check style guides.
        - [ ] Document your code.
        - [ ] Remove all debug code.
        - [ ] Functional changes are tested.
- Your request needs to be approved by a repository admin.
- If your Pull Request is accepted, it will be merged to the `main` branch.

# Testing

Tests are located in `src/tests` and are currently executed manually.

# Developer Certificate of Origin (DCO)

   Developer's Certificate of Origin 1.1

   By making a contribution to this project, I certify that:

   (a) The contribution was created in whole or in part by me and I
       have the right to submit it under the open source license
       indicated in the file; or

   (b) The contribution is based upon previous work that, to the best
       of my knowledge, is covered under an appropriate open source
       license and I have the right under that license to submit that
       work with modifications, whether created in whole or in part
       by me, under the same open source license (unless I am
       permitted to submit under a different license), as indicated
       in the file; or

   (c) The contribution was provided directly to me by some other
       person who certified (a), (b) or (c) and I have not modified
       it.

   (d) I understand and agree that this project and the contribution
       are public and that a record of the contribution (including all
       personal information I submit with it, including my sign-off) is
       maintained indefinitely and may be redistributed consistent with
       this project or the open source license(s) involved.

(taken from [developercertificate.org](https://developercertificate.org/))

See [LICENSE](https://github.com/ni/ni-rf-data-recording-api/blob/main/LICENSE)
for details about how `ni-rf-data-recording-api` is licensed.
