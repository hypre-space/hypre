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
 ../includes/HYPRE_seq_matrix.h CSR_matrix.h mapped_matrix.h multiblock_matrix.h
 
FILES =\
 HYPRE_CSR_matrix.c HYPRE_mapped_matrix.c HYPRE_multiblock_matrix.c \
 CSR_matrix.c mapped_matrix.c multiblock_matrix.c

OBJS = ${FILES:.c=.o}

CC = gcc

F77 = /home/casc/g77/bin/g77

CFLAGS =\
 -I/usr/local/include\
 -I/home/casc/include\
 -I../includes\
 -DHYPRE_MEMORY_CHECK\
 -g

FFLAGS = -g

LFLAGS =\
 -L/usr/local/lib\
 -L/home/casc/lib\
 -L/home/casc/g77/lib\
 -L.\
 -lf2c\
 -lcegdb\
 -lmalloc\
 -lm


##################################################################
# Main rules
##################################################################

${OBJS}: ${HEADERS}

lib: ${OBJS}
	$(AR) -r libHYPREseq_mat_vec.a $(OBJS)

##################################################################
# Generic rules
##################################################################

.c.o:
	@echo "Making (c) " $@
	${CC} -o $@ -c ${CFLAGS} $<

#	@${CC} -o $@ -c ${CFLAGS} $<

.f.${AMG_ARCH}.o:
	@echo "Making (f) " $@
	@${F77} -o $@ -c ${FFLAGS} $<

##################################################################
# Miscellaneous rules
##################################################################

veryclean: clean
	@rm -f driver libHYPRE.a

clean:
	@rm -f *.o

