#BHEADER***********************************************************************
# (c) 1997   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

include ../config/Makefile.config

 
CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
 
C_COMPILE_FLAGS = \
 -I..\
 -I../utilities\
 -I../seq_mv\
 ${CINCLUDES}
 
MPILIBFLAGS = ${MPILIBDIRS} ${MPILIBS} ${MPIFLAGS}
LIBFLAGS = ${LIBDIRS} ${LIBS}
LDLIBFLAGS = ${LDLIBDIRS} ${LDLIBS}
 
LFLAGS =\
 -L.\
 -L../seq_mv\
 -L../utilities\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_utilities\
 ${MPILIBFLAGS} ${LIBFLAGS} ${LDLIBFLAGS}

HEADERS =\
 par_chord_matrix.h


FILES =\
 par_chord_mm.c\
 par_chord_matrix.c\
 compute_outprocessors.c

DRIVER_FILES =\
 driver.c

OBJS = ${FILES:.c=.o}
DRIVER_OBJS = ${DRIVER_FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libHYPRE_parcsr_mv.a
	@cp -f HYPRE_*.h parcsr_matrix_vector.h $(HYPRE_BUILD_DIR)/include/.
	@cp -f libHYPRE_*.a $(HYPRE_BUILD_DIR)/lib/.
 
driver: driver.o libHYPRE_parcsr_mv.a 
	@echo  "Linking" $@ "... "
	${CC} -o driver driver.o ${LFLAGS} 
 

install: all
	@cp -f HYPRE_*.h parcsr_matrix_vector.h $(HYPRE_INC_INSTALL)/.
	@cp -f libHYPRE_*.a $(HYPRE_LIB_INSTALL)/.
 
clean:
	@rm -f *.o lib*.a driver

veryclean: clean

##################################################################
# Rules
##################################################################

libHYPRE_parcsr_mv.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR}  $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}

${DRIVER_OBJS}: ${HEADERS}
