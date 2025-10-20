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
 ij_mm.c\
 sstruct.c\
 sstructmat.c\
 struct.c\
 structmat.c\
 ams_driver.c\
 struct_migrate.c\
 zboxloop.c\

HYPRE_MP_DRIVERS =\
 ij_mp.c\
 test_mp.c\
 test_mp_pcg.c\
 test_mp_pcg_3d.c\
# struct_mp.c\

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
 f77_struct.f

HYPRE_DRIVER_EXECS=${HYPRE_DRIVERS:.c=}
ifdef MP_BUILD
   HYPRE_DRIVER_EXECS+=${HYPRE_MP_DRIVERS:.c=}
endif
HYPRE_F77_EXAMPLES_DRIVER_EXECS=${HYPRE_F77_EXAMPLES_DRIVERS:.c=}
HYPRE_DRIVER_F77_EXECS=${HYPRE_DRIVERS_F77:.f=}
HYPRE_DRIVER_CXX_EXECS=${HYPRE_DRIVERS_CXX:.cxx=}

ifeq (${MP_BUILD}, 1)
   IJ_HELPER_OBJS = ij_helpers.o_flt ij_helpers.o_dbl ij_helpers.o_ldbl ij_helpers.o
else
   IJ_HELPER_OBJS = ij_helpers.o
endif

all: ${HYPRE_DRIVER_EXECS}

all77: ${HYPRE_DRIVER_F77_EXECS}

all++: ${HYPRE_DRIVER_CXX_EXECS}

install:

clean:
	rm -f *.o *.o_flt *.o_dbl *.o_ldbl *.obj *.csv
	rm -rf pchdir tca.map *inslog*

distclean: clean
	rm -f ${HYPRE_DRIVER_EXECS}
	rm -f ${HYPRE_F77_EXAMPLES_DRIVER_EXECS}
	rm -f ${HYPRE_DRIVER_F77_EXECS}
	rm -f ${HYPRE_DRIVER_CXX_EXECS} cxx_*
	rm -f TEST_examples/*.out*
	rm -f TEST_examples/*.err*
	rm -f TEST_ij/*.out*
	rm -f TEST_ij/*.err*
	rm -f TEST_ij/*.txt*
	rm -f TEST_ij/vectors.*
	rm -f TEST_sstruct/*.out*
	rm -f TEST_sstruct/*.err*
	rm -f TEST_struct/*.out*
	rm -f TEST_struct/*.err*
	rm -f TEST_structmat/*.out*
	rm -f TEST_structmat/*.err*


##################################################################
# Rules
##################################################################

# C
ij: ${IJ_HELPER_OBJS} ij.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $^ ${LFLAGS}

ij_assembly: ${IJ_HELPER_OBJS} ij_assembly.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $^ ${LFLAGS}

ij_device: ${IJ_HELPER_OBJS} ij_device.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $^ ${LFLAGS}

ij_mm: ${IJ_HELPER_OBJS} ij_mm.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $^ ${LFLAGS}

# TODO (VPM): Update data structures on sstruct_helpers for GPUs
# sstruct: sstruct_helpers.o sstruct.o
# 	@echo  "Building" $@ "... "
# 	${LINK_CC} -o $@ $^ ${LFLAGS}

sstruct: sstruct.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $^ ${LFLAGS}

sstructmat: sstruct_helpers.o sstructmat.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $^ ${LFLAGS}

struct: struct.obj
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

structmat: structmat.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

ams_driver: ams_driver.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

struct_migrate: struct_migrate.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

zboxloop: zboxloop.obj
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

struct_newboxloop: struct_newboxloop.o $(KOKKOS_LINK_DEPENDS)
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

ij_mp: ${IJ_HELPER_OBJS} ij_mp.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $^ ${LFLAGS}

struct_mp: struct_mp.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

test_mp: test_mp.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

test_mp_pcg: test_mp_pcg.o
	@echo  "Building" $@ "... "
	${LINK_CC} -o $@ $< ${LFLAGS}

test_mp_pcg_3d: test_mp_pcg_3d.o
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
