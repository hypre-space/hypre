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
 general.h\
 headers.h\
 protos.h\
 smg.h\
 ZZZ_ls.h

FILES =\
 cyclic_reduction.c\
 general.c\
 smg2_setup_rap.c\
 smg3_setup_rap.c\
 smg.c\
 smg_intadd.c\
 smg_relax.c\
 smg_residual.c\
 smg_restrict.c\
 smg_setup.c\
 smg_setup_interp.c\
 smg_setup_rap.c\
 smg_setup_restrict.c\
 smg_solve.c\
 ZZZ_struct_smg.c

OBJS = ${FILES:.c=.o}

CC = gcc
F77 = /home/casc/g77/bin/g77

CFLAGS =\
 -I/usr/local/include\
 -I/home/casc/include\
 -I/usr/local/mpi/mpich/include\
 -g -Wall -DZZZ_DEBUG -DZZZ_TIMING

FFLAGS = -g

LFLAGS =\
 -L/usr/local/lib\
 -L/home/casc/lib\
 -L/home/casc/g77/lib\
 -L/usr/local/mpi/mpich/lib/solaris/ch_p4\
 -L.\
 -L../struct_matrix_vector\
 -L../utilities\
 -lZZZ_ls\
 -lZZZ_mv\
 -lZZZ_timing\
 -lZZZ_memory\
 -lnsl -lsocket -lmpi\
 -lf2c\
 -lcegdb\
 -lmalloc\
 -lm


##################################################################
# Main rules
##################################################################

driver_internal: driver_internal.o libZZZ_ls.a
	@echo  "Linking" $@ "... "
	${CC} -o driver_internal driver_internal.o ${LFLAGS}

libZZZ_ls.a: ${OBJS}
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
	@rm -f libZZZ_ls.a
	@rm -f driver driver_internal

clean:
	@rm -f *.o

