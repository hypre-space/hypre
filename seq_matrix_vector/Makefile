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
 HYPRE_mv.h\
 csr_matrix.h\
 headers.h\
 mapped_matrix.h\
 multiblock_matrix.h\
 protos.h\
 vector.h

FILES =\
 HYPRE_csr_matrix.c\
 HYPRE_mapped_matrix.c\
 HYPRE_multiblock_matrix.c\
 csr_matrix.c\
 csr_matvec.c\
 mapped_matrix.c\
 multiblock_matrix.c\
 vector.c

OBJS = ${FILES:.c=.o}

CC = gcc

F77 = /home/casc/g77/bin/g77

CFLAGS =\
 -I/usr/local/include\
 -I/home/casc/include\
 -I/usr/local/mpi/mpich/include\
 -I..\
 -g -Wall -DHYPRE_MEMORY_DMALLOC

FFLAGS = -g

LFLAGS =\
 -L/usr/local/lib\
 -L/home/casc/lib\
 -L/home/casc/g77/lib\
 -L/usr/local/mpi/mpich/lib/solaris/ch_p4\
 -L.\
 -L../utilities\
 -lHYPRE_utilities\
 -lf2c\
 -lcegdb\
 -ldmalloc\
 -lm


##################################################################
# Main rules
##################################################################

libHYPRE_mv.a: ${OBJS}
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

.f.${AMG_ARCH}.o:
	@echo "Making (f) " $@
	@${F77} -o $@ -c ${FFLAGS} $<

##################################################################
# Miscellaneous rules
##################################################################

veryclean: clean
	@rm -f libHYPRE_mv.a
	@rm -f driver driver_internal

clean:
	@rm -f *.o

