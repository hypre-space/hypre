<!--
Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
HYPRE Project Developers. See the top-level COPYRIGHT file for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)
-->

# Writing hypre documentation

The hypre documentation is written in reStructuredText and built through a
combination of Sphinx, doxygen, and breathe.  The User Manual source files are
in the directory `usr-manual` with top-level file `index.rst`.  The Reference
Manual is in `ref-manual`, but the actual content is in the hypre header files.

## Installing the utilities needed to build the documentation

Building the documentation requires a number of things to be installed.  To help
keep everything consistent and manageable, it is best to create a Python virtual
environment (venv) that contains all of the python packages that are required.
This venv can be turned on and off as needed.  The following will install the
venv in the directory `~/python-hypre`:

    mkdir ~/python-hypre
    cd ~/python-hypre
    python3 -m venv env

This creates a subdirectory called `env` that will contain the venv packages.
The following will install the various packages that are needed:

    cd ~/python-hypre
    source env/bin/activate

    pip install --upgrade pip

    pip install sphinx
    pip install breathe

    deactivate

Unfortunately, Sphinx uses a lot of latex packages, so it may be necessary to
install a pretty complete version of TexLive.  This installation takes a while,
but since we all use latex extensively, it's worth doing this for other reasons.
First, download the following and untar it somewhere:

    http://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
    tar xzf install-tl-unx.tar.gz

Now, `cd` into the untarred directory created above, type the following perl
command, then use the sequence of menu commands to change the install directory
to your home directory (here, it's set to `~/texlive/2019`) and install:

    perl install-tl
      D
      1
      ~/texlive/2019
      R
      I

Add `~/texlive/2019/bin/x86_64-linux` to your `PATH` and do `which pdflatex` to
verify that you did it correctly.

## Building the documentation

To build the documentation, first activate the virtual environment:

    source ~/python-hypre/env/bin/activate
        
Now, just type `make` in the `src/docs` directory to build the documentation.
When you are finished editing and building, turn off the virtual environment:

    deactivate

To view the output, open a browser and navigate to the following links to see
the user and reference manuals (adjust the path as needed):

    file:///home/falgout2/hypre/src/docs/usr-manual-html/index.html
    file:///home/falgout2/hypre/src/docs/ref-manual-html/index.html

Alternatively, run a (local) webserver:

    python3 -m http.server --directory usr-manual-html

and open http://localhost:8000 in a browser.

## Some useful links

Sphinx:

- http://www.sphinx-doc.org/en/stable/
- http://www.sphinx-doc.org/en/stable/examples.html
- https://alabaster.readthedocs.io/en/latest/index.html

reStructuredText:

- https://docutils.sourceforge.io/rst.html

Doxygen:

- http://www.doxygen.nl/manual/index.html

Breathe:

- https://breathe.readthedocs.io/en/latest/index.html

## Some notes on customization

After compilation, the CSS style files that control the HTML formatting will be
in the folder `usr-manual-html/_static`.  To override any of these settings, add
the appropriate lines to the file `usr-manual/_static/custom.css`.  Use the web
to get information on CSS.
