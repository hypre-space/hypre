The file configure.ac differs from the normal Babel configure.ac: it
has a few changes to support hypre.  Search for "hypre" to see them.

The file configure was generated from configure.ac with autoconf
version 2.59.  Hypre developers may use it as follows:
/usr/casc/babel/apps/autoconf-2.59/bin/autoconf configure.ac > configure

The configure script in this directory will be run by the configure
script in the main hypre directory and should not be run
independently.  That is to ensure that the two configure scripts use
the same compiler settings.

On CASC Linux machines, run configure with the arguments
 --with-babel --disable-fortran90
as the Fortran compiler doesn't support Fortran90.
Note that Python is also disabled even if you configure hypre with --enable-shared
because Numerical Python (NumPy) isn't installed!

On CASC Linux machines, make sure you have your Java environment set
up by doing source /usr/apps/java/default/setup.csh (or setup.sh if
you run bash)

  Some build system bugs which need to be fixed...
****** In subdirectory sidl there are 2 empty .c files.  I had to make
corresponding empty .o files by hand.  sidl_Resolve_IOR.o,
sidl_Scope_IOR.o
A "make clean" will delete these, so you have to restore them by hand.

***** sidl/*.h needs to be automatically copied into hypre/include -
      many header files are needed to build test drivers.  For now I'm
      doing it by hand.  Not a Babel runtime issue, but the same needs
      to be done for the Babel interface header files, babel/bHYPRE*/*.h

---- the rest of this file is from an older readme file, not recently
     tested ... 

On AIX MP systems, the configure script should be run under nopoe, a
script which forces everything to be run serially on the interactive
(login) node.  Use the provided version of nopoe, which is slightly
different from the system default version on LLNL machines.
Example: "nopoe configure --with-babel" instead of
"configure --with-babel".
Do not abuse nopoe for other purposes, such as production runs.

On frost.llnl.gov, configure cannot find an include necessary for Java
without being told the path, e.g. with the configure argument
 JNI_INCLUDES="-I /usr/java130/include"
That is taken care of in the top-level hypre configure script.
Other systems may need similar additions, depending on where java is.
