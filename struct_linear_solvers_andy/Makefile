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

HEADERS =\
 struct_solver.h
 
LOCALFILES =\
 struct_solver.c\
 struct_solver_PETSc.c

HYPREFILES =\
 HYPRE_struct_solver.c

FILES = $(HYPREFILES) $(LOCALFILES)

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
	@ar -ru libHYPRE_StructSolver.a ${OBJS}
	@ranlib libHYPRE_StructSolver.a

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

