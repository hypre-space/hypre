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
 amg.h

FILES =\
 HYPRE_amg.c\
 amg.c\
 amg_setup.c\
 amg_solve.c\
 coarsen.c\
 cycle.c\
 rap.c

OBJS = ${FILES:.c=.o}

CC = gcc
F77 = /home/casc/g77/bin/g77

CFLAGS =\
 -I/usr/local/include\
 -I/home/casc/include\
 -g -Wall -DHYPRE_DEBUG

FFLAGS = -g

LFLAGS =\
 -L/usr/local/lib\
 -L/home/casc/lib\
 -L/home/casc/g77/lib\
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


##################################################################
# Main rules
##################################################################

driver_internal: driver_internal.o libHYPRE_ls.a
	@echo  "Linking" $@ "... "
	${CC} -o driver_internal driver_internal.o ${LFLAGS}

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
	@rm -f driver driver_internal

clean:
	@rm -f *.o

