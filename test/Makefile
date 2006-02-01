#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}
CDEFS = -DHYPRE_TIMING -DHYPRE_FORTRAN
CXXDEFS = -DNOFEI -DHYPRE_TIMING -DMPICH_SKIP_MPICXX

C_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I${HYPRE_LINK_DIR}/include\
 ${CINCLUDES}\
 ${CDEFS}

CXX_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I$(srcdir)/../FEI_mv/fei-base\
 -I${HYPRE_LINK_DIR}/include\
 ${CINCLUDES}\
 ${CXXDEFS}

F77_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I${HYPRE_LINK_DIR}/include\
 ${CINCLUDES}
 
MPILIBFLAGS = ${MPILIBDIRS} ${MPILIBS} ${MPIFLAGS} 
LIBFLAGS = ${LIBDIRS} ${LIBS} -lstdc++
LDLIBFLAGS = ${LDLIBDIRS} ${LDLIBS} ${LIBM} ${LDFLAGS}
LAPACKLIBFLAGS = ${LAPACKLIBDIRS} ${LAPACKLIBS}
BLASLIBFLAGS = ${BLASLIBDIRS} ${BLASLIBS}

LFLAGS =\
 -L${HYPRE_LINK_DIR}/lib\
 -lHYPRE\
 ${LAPACKLIBFLAGS} \
 ${BLASLIBFLAGS} \
 ${MPILIBFLAGS} ${LDLIBFLAGS} ${LIBFLAGS} ${FLIBS}

BABELLIBFLAGS =\
 -L${HYPRE_LINK_DIR}/lib\
 -L${HYPRE_LINK_DIR}/../babel-runtime/sidl/.libs\
 -lbHYPRE\
 -lbHYPREClient-C\
 -lbHYPREClient-F\
 -lsidl -ldl -lxml2

###########################################
# Commpkg definitions
###########################################
COMMPKG_LFLAGS =\
 -L${HYPRE_LINK_DIR}/lib\
 -lHYPRE_parcsr_ls\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_utilities\
 ${LAPACKLIBFLAGS} \
 ${BLASLIBFLAGS} \
 ${MPILIBFLAGS} ${LDLIBFLAGS} ${LIBFLAGS} ${FLIBS}


###########################################
# FEI definitions
###########################################
FEICXX_LFLAGS =\
 -L${HYPRE_LINK_DIR}/lib\
 -lHYPRE\
 ${LAPACKLIBFLAGS} \
 ${BLASLIBFLAGS} \
 ${MPILIBFLAGS} ${LDLIBFLAGS} ${LIBFLAGS} ${FLIBS}

##################################################################
# Targets
##################################################################
HYPRE_STANDARD_DRIVERS =\
 ij.c\
 ij_mv.c\
 new_ij.c\
 sstruct.c\
 sstruct_fac.c\
 struct.c

HYPRE_BABEL_DRIVERS =\
 struct_b.c\
 sstruct_b.c\
 ij_b.c

HYPRE_FEI_DRIVERS_CXX =\
 fei.cxx

HYPRE_STANDARD_DRIVERS_F77 =\
 f77_ij.f\
 f77_ij_mv.f\
 f77_struct.f

HYPRE_BABEL_DRIVERS_CXX =

HYPRE_BABEL_DRIVERS_F77 =\
 f77_ij_b.f\
 f77_struct_b.f

HYPRE_DRIVERS = ${HYPRE_STANDARD_DRIVERS}
HYPRE_DRIVERS_F77 = ${HYPRE_STANDARD_DRIVERS_F77}

HYPRE_DRIVER_OBJS=${HYPRE_DRIVERS:.c=.o}
HYPRE_DRIVER_F77_OBJS=${HYPRE_DRIVERS_F77:.f=.o}
HYPRE_FEI_DRIVER_CXX_OBJS=${HYPRE_FEI_DRIVERS_CXX:.cxx=.o}

HYPRE_DRIVER_EXECS=${HYPRE_DRIVERS:.c=}
HYPRE_DRIVER_F77_EXECS=${HYPRE_DRIVERS_F77:.f=}
HYPRE_FEI_DRIVER_CXX_EXECS=${HYPRE_FEI_DRIVERS_CXX:.cxx=}

HYPRE_BABEL_DRIVER_OBJS=${HYPRE_BABEL_DRIVERS:.c=.o}
HYPRE_BABEL_DRIVER_CXX_OBJS=${HYPRE_BABEL_DRIVERS_CXX:.cxx=.o}
HYPRE_BABEL_DRIVER_F77_OBJS=${HYPRE_BABEL_DRIVERS_F77:.f=.o}

HYPRE_BABEL_DRIVER_EXECS=${HYPRE_BABEL_DRIVERS:.c=}
HYPRE_BABEL_DRIVER_CXX_EXECS=${HYPRE_BABEL_DRIVERS_CXX:.cxx=}
HYPRE_BABEL_DRIVER_F77_EXECS=${HYPRE_BABEL_DRIVERS_F77:.f=}

all: ${HYPRE_DRIVER_EXECS}

all77: ${HYPRE_DRIVER_F77_EXECS}

all++: ${HYPRE_FEI_DRIVER_CXX_EXECS}

babel: ${HYPRE_BABEL_DRIVER_EXECS} ${HYPRE_BABEL_DRIVER_F77_EXECS}

install:

clean:
	rm -f *.o
	rm -rf pchdir tca.map *inslog*

distclean: clean
	rm -f ${HYPRE_DRIVER_EXECS}
	rm -f ${HYPRE_DRIVER_F77_EXECS}
	rm -f ${HYPRE_BABEL_DRIVER_EXECS}
	rm -f ${HYPRE_BABEL_DRIVER_CXX_EXECS}
	rm -f ${HYPRE_BABEL_DRIVER_F77_EXECS}
	rm -f ${HYPRE_FEI_DRIVER_CXX_EXECS}
	rm -f TEST_fac/*.out*
	rm -f TEST_fac/*.err*
	rm -f TEST_ij/*.out*
	rm -f TEST_ij/*.err*
	rm -f TEST_ij/*.txt*
	rm -f TEST_ij/vectors.*
	rm -f TEST_sstruct/*.out*
	rm -f TEST_sstruct/*.err*
	rm -f TEST_struct/*.out*
	rm -f TEST_struct/*.err*
	rm -f ex*_for

##################################################################
# Rules
##################################################################

struct: struct.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $@.o ${LFLAGS}

fij_mv.o: fij_mv.f
	@echo  "Building" $@ "... "
	${F77} -c fij_mv.f  ${FFLAGS}

fparcsr_ls.o: fparcsr_ls.f
	@echo  "Building" $@ "... "
	${F77} -c fparcsr_ls.f  ${FFLAGS}

fparcsr_mv.o: fparcsr_mv.f
	@echo  "Building" $@ "... "
	${F77} -c fparcsr_mv.f ${FFLAGS}

fsstruct_ls.o: fsstruct_ls.f
	@echo  "Building" $@ "... "
	${F77} -c fsstruct_ls.f  ${FFLAGS}

fsstruct_mv.o: fsstruct_mv.f
	@echo  "Building" $@ "... "
	${F77} -c fsstruct_mv.f ${FFLAGS}

fstruct_ls.o: fstruct_ls.f
	@echo  "Building" $@ "... "
	${F77} -c fstruct_ls.f  ${FFLAGS}

fstruct_mv.o: fstruct_mv.f
	@echo  "Building" $@ "... "
	${F77} -c fstruct_mv.f ${FFLAGS}

ex1_for: fstruct_mv.o fstruct_ls.o ex1_for.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ fstruct_mv.o fstruct_ls.o $@.o ${FFLAGS} ${LFLAGS}

ex3_for: fstruct_mv.o fstruct_ls.o ex3_for.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ fstruct_mv.o fstruct_ls.o $@.o ${FFLAGS} ${LFLAGS}

ex5_for: fij_mv.o fparcsr_mv.o fparcsr_ls.o ex5_for.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ fij_mv.o fparcsr_mv.o fparcsr_ls.o $@.o ${FFLAGS} ${LFLAGS}

sstruct: sstruct.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $@.o ${LFLAGS}

sstruct_fac: sstruct_fac.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $@.o ${LFLAGS}

hypre_set_precond: hypre_set_precond.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $@.o ${LFLAGS}

test_ij: hypre_set_precond.o test_ij.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ hypre_set_precond.o $@.o ${LFLAGS}

ij: ij.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $@.o ${LFLAGS}

ij_mv: ij_mv.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $@.o ${LFLAGS}

new_ij: new_ij.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $@.o ${LFLAGS}

fei: fei.o
	@echo  "Building" $@ "... "
	${LINK_CXX} -o $@ $@.o ${FEICXX_LFLAGS}

f77_struct: f77_struct.o
	@echo  "Building" $@ "... "
	${F77} -o $@ $@.o ${FFLAGS} ${LFLAGS}

f77_ij: f77_ij.o
	@echo  "Building" $@ "... "
	${F77} -o $@ $@.o ${FFLAGS} ${LFLAGS}

f77_ij_mv: f77_ij_mv.o
	@echo  "Building" $@ "... "
	${F77} -o $@ $@.o ${FFLAGS} ${LFLAGS}

driver_commpkg: driver_commpkg.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $@.o ${COMMPKG_LFLAGS}

##################################################################
# Babel version of above
##################################################################
${HYPRE_BABEL_DRIVER_EXECS}: ${HYPRE_BABEL_DRIVER_OBJS}
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $@.o ${BABELLIBFLAGS} ${LFLAGS}

${HYPRE_BABEL_DRIVER_F77_EXECS}: ${HYPRE_BABEL_DRIVER_F77_OBJS}
	@echo  "Building" $@ "... "
	${F77} -o $@ $@.o ${BABELLIBFLAGS} ${FFLAGS} ${LFLAGS}
