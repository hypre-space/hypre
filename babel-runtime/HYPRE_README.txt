The file configure.ac has some changes to support hypre.  Search for
"hypre" to see the changes.  Also, section 0 of the file had to be
moved to after section 1.

The file configure was generated from configure.ac with autoconf
version 2.52.

"make" in the top-leve hypre directory works (It does "make install"
in its subdirectories.)  "make clean" and "make mostlyclean" will
work once hypre is updated to use GNU standard targets.

