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
 communication.h\
 headers.h\
 protos.h

FILES =\
 communication.c\
 par_csr_matrix.c\
 par_csr_matvec.c\
 par_vector.c

OBJS = ${FILES:.c=.o}

CC = gcc
F77 = /home/casc/g77/bin/g77

CFLAGS =\
 -I/usr/local/include\
 -I/home/casc/include\
 -I/usr/local/mpi/mpich/include\
 -g -Wall -DHYPRE_DEBUG -DHYPRE_TIMING -DHYPRE_MEMORY_DMALLOC

FFLAGS = -g

LFLAGS =\
 -L/usr/local/lib\
 -L/home/casc/lib\
 -L/home/casc/g77/lib\
 -L/usr/local/mpi/mpich/lib/solaris/ch_p4\
 -L.\
 -L../utilities\
 -L../seq_matrix_vector\
 -lHYPRE_memory\
 -lHYPRE_mv\
 -lnsl -lsocket -lmpi -lmpe\
 -lf2c\
 -lcegdb\
 -ldmalloc\
 -lm


##################################################################
# Main rules
##################################################################

test: driver.o libHYPRE_parmv.a 
	@echo  "Linking" $@ "... "
	${CC} -o test driver.o libHYPRE_parmv.a ${LFLAGS} 

test_matvec: driver_matvec.o libHYPRE_parmv.a
	@echo  "Linking" $@ "... "
	${CC} -o test_matvec driver_matvec.o libHYPRE_parmv.a ${LFLAGS} 

driver.o: driver.c
	${CC} -c driver.c ${CFLAGS}

driver_matvec.o: driver_matvec.c
	${CC} -c driver_matvec.c ${CFLAGS}

libHYPRE_parmv.a: ${OBJS}
	@echo  "Building $@ ... "
	ar -ru $@ ${OBJS}
	ranlib $@

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
	@rm -f libHYPRE_parmv.a
	@rm -f test test_matvec driver.o driver_matvec.o

clean:
	@rm -f *.o

