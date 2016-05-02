############################################################################
#
#  Program:         SuperLU_DIST
#
#  Module:          make.inc
#
#  Purpose:         Top-level Definitions
#
#  Creation date:   February 4, 1999   version alpha
#
#  Modified:	    September 1, 1999  version 1.0
#                   March 15, 2003     version 2.0
#		    November 1, 2007   version 2.1
#
############################################################################
#
#  The machine (platform) identifier to append to the library names
#
PLAT		= _sp

#
#  The name of the libraries to be created/linked to
#
DSuperLUroot 	= $(HOME)/SuperLU_DIST_2.2
DSUPERLULIB   	= $(DSuperLUroot)/lib/libsuperlu_dist_2.2.a
#
BLASDEF	     	= -DUSE_VENDOR_BLAS
BLASLIB      	= -lessl
#MPILIB		= -L/usr/lpp/ppe.poe/lib -lmpi
#PERFLIB     	= -L/vol1/VAMPIR/lib -lVT
METISLIB	=
PARMETISLIB	=
LIBS            = $(DSUPERLULIB) $(BLASLIB) $(PARMETISLIB) $(METISLIB)

#
#  The archiver and the flag(s) to use when building archive (library)
#  If your system has no ranlib, set RANLIB = echo.
#
ARCH         	= ar
ARCHFLAGS    	= cr
RANLIB       	= ranlib

############################################################################
CC           	= mpcc
# CFLAGS should be set to be the C flags that include optimization
CFLAGS          = -D_SP -O3 -qarch=PWR3 -qalias=allptrs \
		  -DDEBUGlevel=0 -DPRNTlevel=0
#
# NOOPTS should be set to be the C flags that turn off any optimization
# This must be enforced to compile the two routines: slamch.c and dlamch.c.
NOOPTS		=
############################################################################
FORTRAN         = mpxlf90
FFLAGS          = -WF,-Dsp -O3 -Q -qstrict -qfixed -qinit=f90ptr -qarch=pwr3
############################################################################
LOADER	        = mpxlf90
#LOADOPTS	= -bmaxdata:0x80000000
LOADOPTS	= -bmaxdata:0x70000000
#
############################################################################
#  C preprocessor defs for compilation (-DNoChange, -DAdd_, or -DUpCase)
#
#  Need follow the convention of how C calls a Fortran routine.
#
CDEFS        = -DNoChange

