#!/bin/sh

# Small modifications to several hypre Makefiles needed to allow the use of the
# Visual Studio CL.exe compiler on Windows.  Note that this script should be run
# after ../configure, and should not be called more than once!

# Move *.obj to *.o after compiling an object file
sed -e s,' -c $<',' -c $<; mv -f $*.obj $*.o',g \
       config/Makefile.config > /tmp/Makefile.config
mv -f /tmp/Makefile.config config/Makefile.config

# Take care of the special compilation of lapack/dlamch.c
sed -e s,'-c dlamch.c','-c dlamch.c ; mv -f dlamch.obj dlamch.o',g \
       lapack/Makefile > /tmp/Makefile.lapack
mv -f /tmp/Makefile.lapack lapack/Makefile

# Take care of the special compilation of SuperLU/superlu_timer.c
sed -e s,' $<',' $<; mv -f $*.obj $*.o',g \
       FEI_mv/SuperLU/SRC/Makefile > /tmp/Makefile.SuperLU
mv -f /tmp/Makefile.SuperLU FEI_mv/SuperLU/SRC/Makefile
