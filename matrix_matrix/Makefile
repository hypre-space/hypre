#BHEADER***********************************************************************
# (c) 1997   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************


.SUFFIXES:
.SUFFIXES: .c .f .o

HEADERS =
 
HYPREFILES =\
 HYPRE_ConvertPETScMatrixToDistributedMatrix.c

FILES = $(HYPREFILES) $(LOCALFILES)

OBJS = ${FILES:.c=.o}

#include site-specific information
include ../bmakes/base

##################################################################
# Main rules
##################################################################

lib: $(PROTOS) ${OBJS}
	@echo  "Building $@ ... "
	@ar -ru libHYPRE_MatrixMatrix.a ${OBJS}
	@ranlib libHYPRE_MatrixMatrix.a

${OBJS}: ${HEADERS}

##################################################################
# Generic rules
##################################################################

.c.o:
	@echo "Making (c) " $@
	${CC} -o $@ -c ${CFLAGS} $<

#	@${CC} -o $@ -c ${CFLAGS} $<

.f.${AMG_ARCH}.o:
	@echo "Making (f) " $@
	@${F77} -o $@ -c ${FFLAGS} $<

##################################################################
# Miscellaneous rules
##################################################################

veryclean: clean
	@rm -f driver libHYPRE.a

clean:
	@rm -f *.o

