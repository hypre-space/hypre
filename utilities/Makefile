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

CC = gcc
F77 = /home/casc/g77/bin/g77

CFLAGS =\
 -I/usr/local/include\
 -I/home/casc/include\
 -I/usr/local/mpi/mpich/include\
 -g -Wall\
 -DHYPRE_MEMORY_DMALLOC\
 -DHYPRE_TIMING -DUSE_MPI_TIMER

FFLAGS = -g


##################################################################
# Make all utilities
##################################################################

all: libHYPRE_memory.a libHYPRE_timing.a 

##################################################################
# Memory rules
##################################################################

MEMORY_HEADERS =\
 memory.h

MEMORY_FILES =\
 memory.c\
 memory_dmalloc.c

MEMORY_OBJS = ${MEMORY_FILES:.c=.o}

libHYPRE_memory.a: ${MEMORY_OBJS}
	@echo  "Building $@ ... "
	ar -ru $@ ${MEMORY_OBJS}
	ranlib $@
${MEMORY_OBJS}: ${MEMORY_HEADERS}

##################################################################
# Timing rules
##################################################################

TIMING_HEADERS =\
 timer.h\
 timing.h

TIMING_FILES =\
 timer.c\
 timing.c

TIMING_OBJS = ${TIMING_FILES:.c=.o}

libHYPRE_timing.a: ${TIMING_OBJS}
	@echo  "Building $@ ... "
	ar -ru $@ ${TIMING_OBJS}
	ranlib $@
${TIMING_OBJS}: ${TIMING_HEADERS}

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
	@rm -f libHYPRE_timing.a
	@rm -f libHYPRE_memory.a

clean:
	@rm -f *.o

