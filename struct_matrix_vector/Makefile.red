#BHEADER***********************************************************************
# (c) 1997   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

CC = cicc
F77 = ci77

# CFLAGS = -O
CFLAGS = -O -DHYPRE_TIMING -DHYPRE_COMM_SIMPLE
# CFLAGS = -g -DHYPRE_TIMING -DHYPRE_COMM_SIMPLE

FFLAGS = -O

LFLAGS =\
 -L/usr/local/lib\
 -L.\
 -L../utilities\
 -lHYPRE_mv\
 -lHYPRE_timing\
 -lHYPRE_memory\
 -lmpi\
 -lm

include Makefile.generic

