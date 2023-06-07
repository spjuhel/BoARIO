################
Contributing
################

.. _contributions:

Contributions are welcome, and they are greatly appreciated!
Every little bit helps, and credit will always be given.
If you are interested in contributing, please read all this section
before doing anything and in case of doubt, `contact the developer`_!

.. _contact the developer: pro@sjuhel.org

----------------------
Types of Contributions
----------------------

You can contribute in many ways:

Report Bugs
===========

Before reporting a bug, **Ensure the bug was not already reported** by searching on GitHub under `Issues <https://github.com/spjuhel/BoARIO/issues>`_.

If you are reporting a bug, please use the "Bug report template" `here <https://github.com/spjuhel/BoARIO/issues/new/choose>`_, and include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.


Fix Bugs
========

Look through the Github `issues <https://github.com/spjuhel/BoARIO/issues>`_ for bugs.
If you want to start working on a bug then please write short message on the issue tracker to prevent duplicate work.


Implement Features
==================

Look through the Github issues for features or add your own request 
using the Code feature request template `here <https://github.com/spjuhel/BoARIO/issues/new/choose>`_ 
and assign yourself.

Write Documentation
===================

BoARIO could always use more documentation, whether as part of the official documentation website, in docstrings, in Jupyter Notebook examples,
or even on the web in blog posts, articles, and such.

BoARIO uses `Sphinx <https://sphinx-doc.org>`_ for the user manual (that you are currently reading).


Submit Feedback
===============

The best way to send feedback is to file an issue at `here <https://github.com/spjuhel/BoARIO/issues/new/choose>`_

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions are welcome :)

-----------------------
Pull Request Guidelines
-----------------------

To update the documentation, fix bugs or add new features you need to create a Pull Request.
A PR is a change you make to your local copy of the code for us to review and potentially integrate into the code base.

To create a Pull Request you need to do these steps:

1. Create a Github account.
2. Fork the repository.
3. Clone your fork locally.
4. Go to the created BoARIO folder with :code:`cd boario`.
5. Create a new branch with :code:`git checkout -b <descriptive_branch_name>`.
6. Make your changes to the code or documentation.
7. Run :code:`git add .` to add all the changed files to the commit (to see what files will be added you can run :code:`git add . --dry-run`).
8. To commit the added files use :code:`git commit`. (This will open a command line editor to write a commit message. These should have a descriptive 80 line header, followed by an empty line, and then a description of what you did and why. To use your command line text editor of choice use (for example) :code:`export GIT_EDITOR=vim` before running :code:`git commit`).
9. Now you can push your changes to your Github copy of BoARIO by running :code:`git push origin <descriptive_branch_name>`.
10. If you now go to the webpage for your Github copy of BoARIO you should see a link in the sidebar called "Create Pull Request".
11. Now you need to choose your PR from the menu and click the "Create pull request" button. Be sure to change the pull request target branch to <descriptive_branch_name>!

If you want to create more pull requests, first run :code:`git checkout main` and then start at step 5. with a new branch name.

Feel free to ask questions about this if you want to contribute to BoARIO :)

------------------
Testing Guidelines
------------------

To ensure that you do not introduce bugs into BoARIO, you should test your code thouroughly.

From the base BoARIO folder you call :code:`pytest` to run all the tests, or choose one specific test:

.. code-block:: console

   $ pytest
   $ pytest tests/tests.py::test_log_input

If you introduce a new feature you should add a new test to the tests directory. See the folder for examples.