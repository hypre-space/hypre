# Generated automatically from Makefile.in by configure.
#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

.SUFFIXES:
.SUFFIXES: .c .f .o

srcdir = .
 
CC = gcc
 
CINCLUDES= -I/home/casc/include  -I/usr/local/mpi/mpich-1.1.1/include -I/usr/local/mpi/mpich-1.1.1/lib/solaris/ch_p4
COPT= 
CDEBUG = -g -Wall
CDEFS =  -DHYPRE_MEMORY_DMALLOC -DHYPRE_TIMING

CFLAGS =\
 -I../../utilities\
 -I../../seq_matrix_vector\
 -I../../parcsr_matrix_vector\
 ${CINCLUDES}\
 ${COPT}\
 ${CDEBUG}\
 ${CDEFS}
 
RANLIB= ranlib
 
MPILIBFLAGS =  -L/usr/local/mpi/mpich-1.1.1/lib/solaris/ch_p4  -lpmpich -lmpich -lsocket -lnsl -lpthread  
LIBFLAGS =   -lm
LDLIBFLAGS = -L/home/casc/lib   -lcegdb -ldmalloc
 
LFLAGS =\
 -L.\
 -L../../parcsr_matrix_vector\
 -L../../seq_matrix_vector\
 -L../../utilities\
 -lHYPRE_ls\
 -lHYPRE_parcsr_mv\
 -lHYPRE_mv\
 -lHYPRE_utilities\
 -lmpe\
 ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}
 

HEADERS =\
 headers.h\
 amg.h\
 pcg.h\
 hypre_ls.h\
 HYPRE_ls.h

FILES =\
 driver.c\
 amg.c\
 amgstats.c\
 amg_setup.c\
 amg_solve.c \
 coarsen.c\
 cycle.c\
 indepset.c\
 interp.c\
 laplace.c\
 random.c\
 rap.c \
 relax.c\
 transpose.c\
 HYPRE_amg.c

OBJS = ${FILES:.c=.o}

CC = gcc



##################################################################
# Targets
##################################################################

all: libHYPRE_ls.a driver

install: all

clean:
	@rm -f *.o

veryclean: clean
	@rm -f driver
	@rm -f libHYPRE_ls.a

##################################################################
# Rules
##################################################################

driver: driver.o libHYPRE_ls.a
	@echo  "Linking" $@ "... "
	${CC} -o driver driver.o ${LFLAGS}

libHYPRE_ls.a: ${OBJS}
	@echo  "Building $@ ... "
	ar -ru $@ ${OBJS}
	ranlib $@

${OBJS}: ${HEADERS}

##################################################################
# Generic rules
##################################################################

.c.o:
	@echo "Making (c) " $@
	@${CC} -o $@ -c ${CFLAGS} $<

