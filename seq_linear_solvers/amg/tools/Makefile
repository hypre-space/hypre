#BHEADER***********************************************************************
# (c) 1996   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

.SUFFIXES:
.SUFFIXES: .c .f .o

CC="gcc"
F77="f77"

##################################################################
# All
##################################################################

all: convert_icf3d

##################################################################
# convert_icf3d
##################################################################

CONVERT_ICF3D_HDRS = general.h matrix.h io.h

CONVERT_ICF3D_OBJS = convert_icf3d.o matrix.o write.o

convert_icf3d: Makefile ${CONVERT_ICF3D_HDRS} ${CONVERT_ICF3D_OBJS}
	@${CC} -o $@ ${LFLAGS} ${CONVERT_ICF3D_OBJS}

##################################################################
# Generic rules
##################################################################

.c.o:
	@${CC} -o $@ -c ${CFLAGS} $<

.f.o:
	@${F77} -o $@ -c ${FFLAGS} $<

