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

#Headers
HEADERS=\
 PETScMatPilutSolver.h
 
HYPREFILES =\
 HYPRE_PETScMatPilutSolver.c

LOCALFILES =

FILES = $(HYPREFILES) $(LOCALFILES)

#Automatically generate prototypes when necessary
HYPRE_protos.h: $(HYPREFILES)
	mkproto HYPRE*.c > HYPRE_protos.h

hypre_protos.h: $(FILES)
	mkproto hypre*.c > hypre_protos.h

internal_protos.h: $(FILES)
	mkproto *.c > internal_protos.h

PROTOS =\
 HYPRE_protos.h\
 hypre_protos.h\
 internal_protos.h

OBJS = ${FILES:.c=.o}


${OBJS}: ${HEADERS}

#include site-specific information
include ../../bmakes/base



##################################################################
# Main rules
##################################################################

driver: driver.o libHYPREPETScMatPilutSolver.a
	@echo  "Linking" $@ "... "
	${CC} -o driver driver.o ${LFLAGS}

link:
	rm driver
	make driver

lib: ${PROTOS} ${OBJS}
	@echo  "Building $@ ... "
	@ar -ru libHYPREPETScMatPilutSolver.a ${OBJS}
	@ranlib libHYPREPETScMatPilutSolver.a 

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
