#BHEADER***********************************************************************
# (c) 1997   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************


include ../../config/Makefile.config


all: lib

install:all

#Headers
HEADERS=\
 PETScMatPilutSolver.h
 
HYPREFILES =\
 HYPRE_PETScMatPilutSolver.c

LOCALFILES =

FILES = $(HYPREFILES) $(LOCALFILES)

OBJS = ${FILES:.c=.o}


${OBJS}: ${HEADERS}

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
CDEFS = ${HYPRE_PETSCDEFS}

C_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I$(srcdir)/../..\
 -I$(srcdir)/../../utilities\
 -I$(srcdir)/../../distributed_matrix\
 -I$(srcdir)/../../distributed_ls/pilut \
 ${PETSCINCLUDE} \
 ${CINCLUDES} ${CDEFS}

CXX_COMPILE_FLAGS = \
 ${PETSCINCLUDE} \
 -I$(srcdir)\
 -I$(srcdir)/../..\
 -I$(srcdir)/../../utilities\
 ${CINCLUDES}\
 ${COPT}\
 ${CDEBUG} ${CDEFS}


##################################################################
# Main rules
##################################################################

driver: driver.o lib_HYPRE_PETScMatPilutSolver.a
	@echo  "Linking" $@ "... "
	${CC} -o driver driver.o ${LFLAGS}

link:
	rm driver
	$(MAKE) driver

lib: ${PROTOS} ${OBJS}
	@echo  "Building $@ ... "
	@${AR} lib_HYPRE_PETScMatPilutSolver.a ${OBJS}
	@${RANLIB} lib_HYPRE_PETScMatPilutSolver.a 


##################################################################
# Targets
##################################################################
clean:
	@rm -f *.o lib*.a

distclean: clean
