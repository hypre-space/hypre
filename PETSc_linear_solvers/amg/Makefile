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

HEADERS =\
 HYPRE_amg.h\
 headers.h

FILES =\
 SPamg.c\
 amg.c\
 amg_setup.c\
 coarsen.c\
 indepset.c\
 random.c\
 transpose.c\
 interp.c\
 rap.c \
 amgstats.c\
 amg_solve.c \
 cycle.c\
 relax.c

OBJS = ${FILES:.c=.o}

CC = gcc
F77 = /home/casc/g77/bin/g77

CFLAGS =\
 -I/usr/local/include\
 -I/home/casc/include\
 -I/usr/local/mpi/mpich/include\
 -g -Wall -DHYPRE_DEBUG 
#-g -Wall  

FFLAGS = -g

LFLAGS =\
 -L/usr/local/lib\
 -L/home/casc/lib\
 -L/home/casc/g77/lib\
 -L/usr/local/mpi/mpich/lib/solaris/ch_p4\
 -L.\
 -L../../seq_matrix_vector\
 -L../../utilities\
 -lHYPRE_ls\
 -lHYPRE_mv\
 -lHYPRE_memory\
 -lf2c\
 -lcegdb\
 -lmalloc\
 -lm



SPAMGFILES =  SPamg.o amg.o amg_setup.o coarsen.o  indepset.o \
               random.o transpose.o interp.o rap.o amgstats.o \
               amg_solve.o cycle.o relax.o

##################################################################
# Main rules
##################################################################

SPamg: ${SPAMGFILES} libHYPRE_ls.a
	@echo  "Linking" $@ "... "
	${CC} -o SPamg ${SPAMGFILES}  ${LFLAGS}

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

.f.${AMG_ARCH}.o:
	@echo "Making (f) " $@
	@${F77} -o $@ -c ${FFLAGS} $<

##################################################################
# Miscellaneous rules
##################################################################

veryclean: clean
	@rm -f libHYPRE_ls.a
	@rm -f SPamg

clean:
	@rm -f *.o

