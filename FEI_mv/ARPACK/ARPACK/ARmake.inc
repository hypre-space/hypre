############################################################################
#
#  Program:         ARPACK
#
#  Module:          ARmake.inc
#
#  Purpose:         Top-level Definitions
#
#  Creation date:   February 22, 1996
#
#  Modified:
#
#  Send bug reports, comments or suggestions to arpack@caam.rice.edu
#
############################################################################
#
#  The complete path to the top level of ARPACK directory, usually
#  $(HOME)/ARPACK
#
home = $(HOME)/HYPRE/GPS/linear_solvers-v2.5.1R3/FEI_mv/ARPACK/ARPACK

#
#  The platform identifier to suffix to the end of library names
#
PLAT = alpha

#
#  The directories to find the various pieces of ARPACK
#
BLASdir      = $(home)/BLAS
LAPACKdir    = $(home)/LAPACK
UTILdir      = $(home)/UTIL
SRCdir       = $(home)/SRC

DIRS   = $(BLASdir) $(LAPACKdir) $(UTILdir) $(SRCdir)
# Comment out the previous line and uncomment the following
# if you already have the BLAS and LAPACK installed on your system.
# NOTE: ARPACK assumes the use of LAPACK version 2 codes.
#DIRS   = $(LAPACKdir) $(UTILdir) $(SRCdir)

#
#  The name of the libraries to be created/linked to
#
ARPACKLIB  = $(home)/libarpack_$(PLAT).a
LAPACKLIB = 
BLASLIB = -ldxml 

ALIBS =  $(ARPACKLIB) $(LAPACKLIB) $(BLASLIB) 

#
#  Make our own suffixes' list.
#
.SUFFIXES:
.SUFFIXES:	.f	.o
#.SUFFIXES:	.F	.f

#
#  Default command.
#
.DEFAULT:
	@$(ECHO) "Unknown target $@, try:  make help"

#
#  Command to build .o files from .f files.
#
.f.o:
	@$(ECHO) Making $@ from $<
	@$(FC) -c $(FFLAGS) $<
#.F.f:
#	@$(ECHO) Making $*.f from $<
#	@$(CPP) -P -DSINGLE $(CPPFLAGS) $< $*.f
#	@$(ECHO) Making d$*.f from $<
#	@$(CPP) -P -DDOUBLE $(CPPFLAGS) $< d$*.f

#  Various compilation programs and flags.
#  You need to make sure these are correct for your system.
FC       = f77
FFLAGS	 = -O3

LDFLAGS  = 
CD       = cd

CHMOD    = chmod
CHFLAGS  = -f

COMPRESS = compress

CP       = cp

#  To find the path for cpp, try `man cpp', and it should list the path
#  at the top, under `Syntax'.  It is usually in /lib.
CPP	 = /lib/cpp
CPPFLAGS =

ECHO	 = echo

LN	 = ln
LNFLAGS	 = -s

MAKE	 = /bin/make

MKDIR	 = mkdir
MDFLAGS	 = -p

MV	 = mv
MVFLAGS	 = -f

RM	 = rm
RMFLAGS  = -f

SHELL	 = /bin/sh

TAR	 = tar

#  For System V based machine without ranlib, like Cray and SGI, 
#  use touch instead.
#RANLIB   = touch
RANLIB   = ranlib

#
#  This is the general help target.
#
help:
	@$(ECHO) "usage: make ?"
