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
HEADERS =\
 distributed_matrix.h
 
HYPREFILES =\
 HYPRE_distributed_matrix.c

LOCALFILES =\
 distributed_matrix.c\
 distributed_matrix_PETSc.c\
 distributed_matrix_ISIS.cc

FILES = $(HYPREFILES) $(LOCALFILES)

#Automatically generate prototypes when necessary
HYPRE_protos.h: $(HYPREFILES)
	mkproto HYPRE*.c > HYPRE_protos.h

hypre_protos.h: $(FILES)
	mkproto hypre*.c > hypre_protos.h

internal_protos.h: $(FILES)
	mkproto *.c > internal_protos.h

PROTOS=\
 HYPRE_protos.h\
 hypre_protos.h\
 internal_protos.h

OBJS = ${FILES:.c=.o}


${OBJS}: ${HEADERS}

#include site-specific information
include ../bmakes/base


##################################################################
# Main rules
##################################################################

driver: driver.o libHYPRE.a
	@echo  "Linking" $@ "... "
	${CC} -o driver driver.o ${LFLAGS}

link:
	rm driver
	make driver

lib: $(PROTOS) ${OBJS}
	@echo  "Building $@ ... "
	@ar -ru libHYPREDistributedMatrix.a ${OBJS}
	@ranlib libHYPREDistributedMatrix.a 

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

