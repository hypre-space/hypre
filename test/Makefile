#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision$
#EHEADER**********************************************************************


include ../config/Makefile.config

CINCLUDES = ${MPIINCLUDE}
CDEFS = -DHYPRE_TIMING -DHYPRE_FORTRAN
CXXDEFS = -DNOFEI -DHYPRE_TIMING -DMPICH_SKIP_MPICXX

C_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I${HYPRE_BUILD_DIR}/include\
 ${CINCLUDES}\
 ${CDEFS}

CXX_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I$(srcdir)/../FEI_mv/fei-base\
 -I${HYPRE_BUILD_DIR}/include\
 ${CINCLUDES}\
 ${CXXDEFS}

F77_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I$(srcdir)/..\
 -I${HYPRE_BUILD_DIR}/include\
 ${CINCLUDES}

MPILIBFLAGS = ${MPILIBDIRS} ${MPILIBS} ${MPIFLAGS} 
LAPACKLIBFLAGS = ${LAPACKLIBDIRS} ${LAPACKLIBS}
BLASLIBFLAGS = ${BLASLIBDIRS} ${BLASLIBS}
LIBFLAGS = ${LDFLAGS} ${LIBS}
# LIBFLAGS = ${LDFLAGS} ${LIBS} ${FLIBS}

LFLAGS =\
 -L${HYPRE_BUILD_DIR}/lib\
 -lHYPRE\
 ${MPILIBFLAGS}\
 ${LAPACKLIBFLAGS}\
 ${BLASLIBFLAGS}\
 ${LIBFLAGS}

BABELLIBFLAGS =\
 -L${HYPRE_BUILD_DIR}/lib\
 -lbHYPRE\
 -lbHYPREClient-C\
 -lbHYPREClient-F\
 -lbHYPRE\
 -lsidl -ldl -lxml2

##################################################################
# Targets
##################################################################

HYPRE_DRIVERS =\
 ij.c\
 new_ij.c\
 sstruct.c\
 struct.c\
 ams_driver.c\
 maxwell_unscaled.c\
 sstruct_fac.c\
 ij_mv.c\
 zboxloop.c

HYPRE_BABEL_DRIVERS =\
 struct_b.c\
 sstruct_b.c\
 ij_b.c

HYPRE_DRIVERS_CXX =\
 fei.cxx\
 cxx_ij.cxx\
 cxx_sstruct.cxx\
 cxx_struct.cxx

HYPRE_F77_EXAMPLES_DRIVERS =\
 ex1_for.c\
 ex3_for.c\
 ex5_for.c\
 ex6_for.c\
 ex7_for.c

HYPRE_DRIVERS_F77 =\
 f77_ij.f\
 f77_ij_mv.f\
 f77_struct.f

HYPRE_BABEL_DRIVERS_CXX =

HYPRE_BABEL_DRIVERS_F77 =\
 f77_ij_b.f\
 f77_struct_b.f


HYPRE_DRIVER_EXECS=${HYPRE_DRIVERS:.c=}
HYPRE_F77_EXAMPLES_DRIVER_EXECS=${HYPRE_F77_EXAMPLES_DRIVERS:.c=}
HYPRE_DRIVER_F77_EXECS=${HYPRE_DRIVERS_F77:.f=}
HYPRE_DRIVER_CXX_EXECS=${HYPRE_DRIVERS_CXX:.cxx=}

HYPRE_BABEL_DRIVER_OBJS=${HYPRE_BABEL_DRIVERS:.c=.o}
HYPRE_BABEL_DRIVER_CXX_OBJS=${HYPRE_BABEL_DRIVERS_CXX:.cxx=.o}
HYPRE_BABEL_DRIVER_F77_OBJS=${HYPRE_BABEL_DRIVERS_F77:.f=.o}

HYPRE_BABEL_DRIVER_EXECS=${HYPRE_BABEL_DRIVERS:.c=}
HYPRE_BABEL_DRIVER_CXX_EXECS=${HYPRE_BABEL_DRIVERS_CXX:.cxx=}
HYPRE_BABEL_DRIVER_F77_EXECS=${HYPRE_BABEL_DRIVERS_F77:.f=}

all: ${HYPRE_DRIVER_EXECS}

all77: ${HYPRE_DRIVER_F77_EXECS}

all++: ${HYPRE_DRIVER_CXX_EXECS}

babel: ${HYPRE_BABEL_DRIVER_EXECS} ${HYPRE_BABEL_DRIVER_F77_EXECS}

install:

clean:
	rm -f *.o
	rm -rf pchdir tca.map *inslog*

distclean: clean
	rm -f ${HYPRE_DRIVER_EXECS}
	rm -f ${HYPRE_F77_EXAMPLES_DRIVER_EXECS}
	rm -f ${HYPRE_DRIVER_F77_EXECS}
	rm -f ${HYPRE_BABEL_DRIVER_EXECS}
	rm -f ${HYPRE_BABEL_DRIVER_CXX_EXECS}
	rm -f ${HYPRE_BABEL_DRIVER_F77_EXECS}
	rm -f ${HYPRE_DRIVER_CXX_EXECS} cxx_*
	rm -f TEST_examples/*.out*
	rm -f TEST_examples/*.err*
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
	rm -f ex1 ex2 ex3 ex4 ex5 ex5b ex5b77 ex6 ex7 ex8 ex9

##################################################################
# Rules
##################################################################

zboxloop: zboxloop.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $@.o ${LFLAGS}

struct: struct.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $@.o ${LFLAGS}

fij_mv: fij_mv.f
	@echo  "Building" $@ "... "
	${F77} -c $@  ${FFLAGS}

fparcsr_ls: fparcsr_ls.f
	@echo  "Building" $@ "... "
	${F77} -c $@  ${FFLAGS}

fparcsr_mv: fparcsr_mv.f
	@echo  "Building" $@ "... "
	${F77} -c $@  ${FFLAGS}

fsstruct_ls: fsstruct_ls.f 
	@echo  "Building" $@ "... "
	${F77} -c $@  ${FFLAGS}

fsstruct_mv: fsstruct_mv.f
	@echo  "Building" $@ "... "
	${F77} -c $@  ${FFLAGS}

fstruct_ls: fstruct_ls.f
	@echo  "Building" $@ "... "
	${F77} -c $@  ${FFLAGS}

fstruct_mv: fstruct_mv.f
	@echo  "Building" $@ "... "
	${F77} -c $@  ${FFLAGS}

struct_for: fstruct_mv.o fstruct_ls.o struct_for.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ fstruct_mv.o fstruct_ls.o $@.o ${FFLAGS} ${LFLAGS}

ex1_for: fstruct_mv.o fstruct_ls.o ex1_for.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ fstruct_mv.o fstruct_ls.o $@.o ${FFLAGS} ${LFLAGS}

ex3_for: fstruct_mv.o fstruct_ls.o ex3_for.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ fstruct_mv.o fstruct_ls.o $@.o ${FFLAGS} ${LFLAGS}

ex5_for: fij_mv.o fparcsr_mv.o fparcsr_ls.o ex5_for.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ fij_mv.o fparcsr_mv.o fparcsr_ls.o $@.o ${FFLAGS} ${LFLAGS}

ex6_for: fstruct_mv.o fstruct_ls.o fsstruct_mv.o fsstruct_ls.o ex6_for.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ fstruct_mv.o fstruct_ls.o fsstruct_mv.o fsstruct_ls.o $@.o ${FFLAGS} ${LFLAGS}

ex7_for: fstruct_mv.o fstruct_ls.o fsstruct_mv.o fsstruct_ls.o ex7_for.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ fstruct_mv.o fstruct_ls.o fsstruct_mv.o fsstruct_ls.o $@.o ${FFLAGS} ${LFLAGS}

for_maxwell: fparcsr_mv.o fsstruct_mv.o fsstruct_ls.o for_maxwell.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ fparcsr_mv.o fsstruct_mv.o fsstruct_ls.o $@.o ${FFLAGS} ${LFLAGS}

sstruct: sstruct.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $@.o ${LFLAGS}

maxwell_unscaled: maxwell_unscaled.o
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
	${LINK_CXX} -o $@ $@.o ${LFLAGS}

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
	${LINK_CC} -o $@ $@.o ${LFLAGS}

ams_driver: ams_driver.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $@.o ${LFLAGS}

cxx_ij: cxx_ij.o
	@echo  "Building" $@ "... "
	${LINK_CXX} -o $@ $@.o ${LFLAGS}
cxx_ij.o: cxx_ij.cxx
cxx_ij.cxx: ij.c
	cp -fp ij.c cxx_ij.cxx

cxx_sstruct: cxx_sstruct.o
	@echo  "Building" $@ "... "
	${LINK_CXX} -o $@ $@.o ${LFLAGS}
cxx_sstruct.o: cxx_sstruct.cxx
cxx_sstruct.cxx: sstruct.c
	cp -fp sstruct.c cxx_sstruct.cxx

cxx_struct: cxx_struct.o
	@echo  "Building" $@ "... "
	${LINK_CXX} -o $@ $@.o ${LFLAGS}
cxx_struct.o: cxx_struct.cxx
cxx_struct.cxx: struct.c
	cp -fp struct.c cxx_struct.cxx

##################################################################
# Babel version of above
##################################################################
${HYPRE_BABEL_DRIVER_EXECS}: ${HYPRE_BABEL_DRIVER_OBJS}
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $@.o ${BABELLIBFLAGS} ${LFLAGS}

${HYPRE_BABEL_DRIVER_F77_EXECS}: ${HYPRE_BABEL_DRIVER_F77_OBJS}
	@echo  "Building" $@ "... "
	${F77} -o $@ $@.o ${BABELLIBFLAGS} ${FFLAGS} ${LFLAGS}
