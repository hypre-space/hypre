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
 headers.h\
 protos.h

FILES =\
 par_rap.c

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
 -lHYPRE_memory\
 -lnsl -lsocket -lmpi -lmpe\
 -lf2c\
 -lcegdb\
 -ldmalloc\
 -lm


##################################################################
# Main rules
##################################################################

test_rap: driver_rap.o libCodes.a
	@echo  "Linking" $@ "... "
	${CC} -o test_rap driver_rap.o libCodes.a ${LFLAGS} 

driver_rap.o: driver_rap.c
	${CC} -c driver_rap.c ${CFLAGS}

libCodes.a: ${OBJS}
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
	@rm -f libCodes.a
	@rm -f test driver.o

clean:
	@rm -f *.o

