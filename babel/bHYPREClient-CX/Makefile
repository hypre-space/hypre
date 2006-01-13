#BHEADER***********************************************************************
# (c) 2006   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
#EHEADER***********************************************************************

include ../../config/Makefile.config
include babel.make

CINCLUDES=${INCLUDES} ${MPIINCLUDE}
CDEFS = 
 
BABEL_RUNTIME_INCLUDES=\
  -I$(HYPRE_SRC_TOP_DIR)/babel-runtime/sidl

C_COMPILE_FLAGS=\
 -I..\
 -I$(HYPRE_SRC_TOP_DIR)\
 -I$(HYPRE_SRC_TOP_DIR)/utilities\
 -I$(HYPRE_SRC_TOP_DIR)/seq_mv\
 -I$(HYPRE_SRC_TOP_DIR)/krylov\
 -I$(HYPRE_SRC_TOP_DIR)/IJ_mv\
 -I$(HYPRE_SRC_TOP_DIR)/parcsr_mv\
 -I$(HYPRE_SRC_TOP_DIR)/parcsr_ls\
 ${BABEL_RUNTIME_INCLUDES}\
 ${CINCLUDES}\
 ${CDEFS}

CXX_INCLUDES=${C_INCLUDES}
CXX_DEFS=${C_DEFS}
CXX_COMPILE_FLAGS=${C_COMPILE_FLAGS}
 
LIBFLAGS = ${LIBDIRS} ${LIBS}
LDLIBFLAGS = ${LDLIBDIRS} ${LDLIBS}
 
LFLAGS =\
 -L.\
 -L../seq_mv\
 -L../utilities\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_utilities\
 ${LIBFLAGS} ${LDLIBFLAGS}

HEADERS = ${STUBHDRS} ${IORHDRS} ${IMPLHDRS}

FILES = ${IMPLSRCS} ${IORSRCS} ${SKELSRCS} ${STUBSRCS} 

OBJS = ${FILES:.cc=.o}

##################################################################
# Targets
##################################################################

all: libbHYPREClient-CX.a
	cp -fp $(srcdir)/*.h ${HYPRE_BUILD_DIR}/include/.
	cp -fp $(srcdir)/*.hh ${HYPRE_BUILD_DIR}/include/.
	cp -fp libbHYPRE* ${HYPRE_BUILD_DIR}/lib/.
 
install: all
	cp -fp $(srcdir)/*.h ${HYPRE_INC_INSTALL}/.
	cp -fp $(srcdir)/*.hh ${HYPRE_INC_INSTALL}/.
	cp -fp libbHYPRE* ${HYPRE_LIB_INSTALL}/.
 
clean:
	rm -f *.o libbHYPRE*

distclean: clean

##################################################################
# Rules
##################################################################

libbHYPREClient-CX.a: ${OBJS}
	@echo  "Building $@ ... "
	${AR} $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}
