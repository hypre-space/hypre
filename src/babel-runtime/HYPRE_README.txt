As of November 14, 2005 (for the first time), this file is the only
part of the babel-runtime directory which differs from the Babel
group's runtime distribution.

If you need to generate a new "configure" file, use autoconf version
2.59 (along with associated programs which come with autoconf).  Hypre
developers with the right permissions may use it as follows:
  /usr/casc/babel/apps/autoconf-2.59/bin/autoconf configure.ac > configure
If you don't have access to that directory, download and build your
own copy of autoconf.

The configure script in this directory will be run by the configure
script in the main hypre directory and should not be run
independently.  That is to ensure that the two configure scripts use
the same compiler settings.

The "assert" macro is frequently invoked here.  This violates hypre
policy, which is to use hypre_assert instead (the difference is that
assert defaults to on, hypre_assert defaults to off; we have customers
who complain if asserts are on and don't want to have to turn them
off).  I don't have a solution.

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
