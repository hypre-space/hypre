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
 box.h\
 communication.h\
 computation.h\
 general.h\
 headers.h\
 protos.h\
 sbox.h\
 struct_grid.h\
 struct_matrix.h\
 struct_stencil.h\
 struct_vector.h

FILES =\
 box.c\
 box_algebra.c\
 communication.c\
 computation.c\
 grow.c\
 neighbors.c\
 project.c\
 sbox.c\
 struct_grid.c\
 struct_io.c\
 struct_matrix.c\
 struct_stencil.c\
 struct_vector.c

OBJS = ${FILES:.c=.o}

CC = gcc
F77 = /home/casc/g77/bin/g77

CFLAGS =\
 -I/usr/local/include\
 -I/home/casc/include\
 -I//usr/local/mpi/mpich/include\
 -g

FFLAGS = -g

LFLAGS =\
 -L/usr/local/lib\
 -L/home/casc/lib\
 -L/home/casc/g77/lib\
 -L/usr/local/mpi/mpich/lib/solaris/ch_p4\
 -L.\
 -lZZZ\
 -lnsl -lsocket -lmpi\
 -lf2c\
 -lcegdb\
 -lmalloc\
 -lm


##################################################################
# Main rules
##################################################################

driver_internal: driver_internal.o libZZZ.a
	@echo  "Linking" $@ "... "
	${CC} -o driver_internal driver_internal.o ${LFLAGS}

libZZZ.a: ${OBJS}
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
	@rm -f libZZZ.a
	@rm -f driver driver_internal

clean:
	@rm -f *.o

