The file configure.ac has some changes to support hypre.  Search for
"hypre" to see the changes.  Also, section 0 of the file had to be
moved to after section 1.

The file configure was generated from configure.ac with autoconf
version 2.52.

"make" in the top-leve hypre directory works (It does "make install"
in its subdirectories.)  "make clean" and "make mostlyclean" will
work once hypre is updated to use GNU standard targets.

On frost.llnl.gov, configure cannot find an include necessary for Java
without being told the path, with the configure argument
 JNI_INCLUDES="-I /usr/java130/include"
If it doesn't have that information, it should (but doesn't) disable
Java from the languages supported by the Babel runtime.
Consequently ... I have changed the configure.ac file to disable
Java in all cases.  Fortran90 is also disabled, as it also won't build
properly on Frost.
