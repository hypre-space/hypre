# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

default:all

include ../config/Makefile.config

CINCLUDES = ${INCLUDES} ${MPIINCLUDE}

CDEFS = -DHYPRE_TIMING -DHYPRE_FORTRAN
CXXDEFS = -DNOFEI -DHYPRE_TIMING -DMPICH_SKIP_MPICXX

C_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I${HYPRE_BUILD_DIR}/include\
 $(SUPERLU_INCLUDE)\
 $(DSUPERLU_INCLUDE)\
 ${CINCLUDES}\
 ${CDEFS}

CXX_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I$(srcdir)/../FEI_mv/fei-base\
 -I${HYPRE_BUILD_DIR}/include\
 $(SUPERLU_INCLUDE)\
 $(DSUPERLU_INCLUDE)\
 ${CINCLUDES}\
 ${CXXDEFS}

F77_COMPILE_FLAGS = \
 -I$(srcdir)\
 -I${HYPRE_BUILD_DIR}/include\
 ${CINCLUDES}

MPILIBFLAGS = ${MPILIBDIRS} ${MPILIBS} ${MPIFLAGS}
LAPACKLIBFLAGS = ${LAPACKLIBDIRS} ${LAPACKLIBS}
BLASLIBFLAGS = ${BLASLIBDIRS} ${BLASLIBS}
LIBFLAGS = ${LDFLAGS} ${LIBS}
LIBHYPRE = -L${HYPRE_BUILD_DIR}/lib -lHYPRE

ifeq ($(notdir $(firstword ${LINK_CC})), nvcc)
   XLINK = -Xlinker=-rpath,${HYPRE_BUILD_DIR}/lib
else
   XLINK = -Wl,-rpath,${HYPRE_BUILD_DIR}/lib
endif

LFLAGS =\
 $(LIBHYPRE)\
 ${XLINK}\
 ${DSUPERLU_LIBS}\
 ${SUPERLU_LIBS}\
 ${MPILIBFLAGS}\
 ${LAPACKLIBFLAGS}\
 ${BLASLIBFLAGS}\
 ${LIBFLAGS}

##################################################################
# Targets
##################################################################

HYPRE_DRIVERS =\
 ij.c\
 ij_assembly.c\
 sstruct.c\
 struct.c\
 ams_driver.c\
 maxwell_unscaled.c\
 struct_migrate.c\
 sstruct_fac.c\
 ij_mv.c\
 ij_mm.c\
 zboxloop.c

HYPRE_DRIVERS_CXX =\
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

HYPRE_DRIVER_EXECS=${HYPRE_DRIVERS:.c=}
HYPRE_F77_EXAMPLES_DRIVER_EXECS=${HYPRE_F77_EXAMPLES_DRIVERS:.c=}
HYPRE_DRIVER_F77_EXECS=${HYPRE_DRIVERS_F77:.f=}
HYPRE_DRIVER_CXX_EXECS=${HYPRE_DRIVERS_CXX:.cxx=}

all: ${HYPRE_DRIVER_EXECS}

all77: ${HYPRE_DRIVER_F77_EXECS}

all++: ${HYPRE_DRIVER_CXX_EXECS}

install:

clean:
	rm -f *.o *.obj *.csv
	rm -rf pchdir tca.map *inslog*

distclean: clean
	rm -f ${HYPRE_DRIVER_EXECS}
	rm -f ${HYPRE_F77_EXAMPLES_DRIVER_EXECS}
	rm -f ${HYPRE_DRIVER_F77_EXECS}
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
	rm -f ex1 ex2 ex3 ex4 ex5 ex5big ex5b ex5b77 ex6 ex7 ex8 ex9 ex15big ex18comp

##################################################################
# Rules
##################################################################

# C

ij: ij.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

ij_assembly: ij_assembly.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

sstruct: sstruct.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

struct: struct.obj
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

ams_driver: ams_driver.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

maxwell_unscaled: maxwell_unscaled.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

struct_migrate: struct_migrate.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

sstruct_fac: sstruct_fac.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

ij_mv: ij_mv.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

ij_device: ij_device.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

ij_mm: ij_mm.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

zboxloop: zboxloop.obj
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

struct_newboxloop: struct_newboxloop.o $(KOKKOS_LINK_DEPENDS)
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

test_mgr: test_mgr.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

test_mgr_gpu: test_mgr_gpu.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

# RDF: Keep these for now

hypre_set_precond: hypre_set_precond.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $@.o ${LFLAGS}

test_ij: hypre_set_precond.o test_ij.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ hypre_set_precond.o $@.o ${LFLAGS}

driver_commpkg: driver_commpkg.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $@.o ${LFLAGS}

# C++

fei: fei.o
	@echo  "Building" $@ "... "
	${LINK_CXX} -o $@ $@.o ${LFLAGS}

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

# Fortran

f77_ij: f77_ij.o
	@echo  "Building" $@ "... "
	${LINK_FC} -o $@ $@.o ${LFLAGS}

f77_ij_mv: f77_ij_mv.o
	@echo  "Building" $@ "... "
	${LINK_FC} -o $@ $@.o ${LFLAGS}

f77_struct: f77_struct.o
	@echo  "Building" $@ "... "
	${LINK_FC} -o $@ $@.o ${LFLAGS}

# RDF: Keep these for now

struct_for: fstruct_mv.o fstruct_ls.o struct_for.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ fstruct_mv.o fstruct_ls.o $@.o ${LFLAGS}

ex1_for: fstruct_mv.o fstruct_ls.o ex1_for.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ fstruct_mv.o fstruct_ls.o $@.o ${LFLAGS}

ex3_for: fstruct_mv.o fstruct_ls.o ex3_for.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ fstruct_mv.o fstruct_ls.o $@.o ${LFLAGS}

ex5_for: fij_mv.o fparcsr_mv.o fparcsr_ls.o ex5_for.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ fij_mv.o fparcsr_mv.o fparcsr_ls.o $@.o ${LFLAGS}

ex6_for: fstruct_mv.o fstruct_ls.o fsstruct_mv.o fsstruct_ls.o ex6_for.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ fstruct_mv.o fstruct_ls.o fsstruct_mv.o fsstruct_ls.o $@.o ${LFLAGS}

ex7_for: fstruct_mv.o fstruct_ls.o fsstruct_mv.o fsstruct_ls.o ex7_for.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ fstruct_mv.o fstruct_ls.o fsstruct_mv.o fsstruct_ls.o $@.o ${LFLAGS}

for_maxwell: fparcsr_mv.o fsstruct_mv.o fsstruct_ls.o for_maxwell.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ fparcsr_mv.o fsstruct_mv.o fsstruct_ls.o $@.o ${LFLAGS}

fij_mv: fij_mv.f
	@echo  "Building" $@ "... "
	${LINK_FC} -c $@

fparcsr_ls: fparcsr_ls.f
	@echo  "Building" $@ "... "
	${LINK_FC} -c $@

fparcsr_mv: fparcsr_mv.f
	@echo  "Building" $@ "... "
	${LINK_FC} -c $@

fsstruct_ls: fsstruct_ls.f
	@echo  "Building" $@ "... "
	${LINK_FC} -c $@

fsstruct_mv: fsstruct_mv.f
	@echo  "Building" $@ "... "
	${LINK_FC} -c $@

fstruct_ls: fstruct_ls.f
	@echo  "Building" $@ "... "
	${LINK_FC} -c $@

fstruct_mv: fstruct_mv.f
	@echo  "Building" $@ "... "
	${LINK_FC} -c $@
