#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

.SUFFIXES:
.SUFFIXES: .c .o

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

CFLAGS =\
 -I/usr/local/include\
 -I/home/casc/include\
 -I/usr/local/mpi/mpich/include\
 -I../../utilities\
 -I../../seq_matrix_vector\
 -g -Wall -DHYPRE_SEQUENTIAL -DHYPRE_MEMORY_DMALLOC
#-g -Wall -DHYPRE_SEQUENTIAL

FFLAGS = -g

LFLAGS =\
 -L/usr/local/lib\
 -L/home/casc/lib\
 -L/usr/local/mpi/mpich/lib/solaris/ch_p4\
 -L.\
 -L../../seq_matrix_vector\
 -L../../utilities\
 -lHYPRE_ls\
 -lHYPRE_mv\
 -lHYPRE_utilities\
 -lcegdb\
 -ldmalloc\
 -lm

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

