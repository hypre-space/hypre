# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

########################################################################
# Compiler and external dependences
########################################################################
CC        = mpicc
F77       = mpif77
CXX       = mpicxx
F90       = mpif90
CUF       = mpixlcuf
LINK_CC   = ${CC}
LINK_CXX  = ${CXX}
LINK_FC   = ${F77}
LINK_CUF  = ${CUF}

XL_DIR=$(dir $(shell which xlc))..

HYPRE_DIR = ../hypre

########################################################################
# CUDA
########################################################################
ifeq ($(use_cuda), 1)
   CUDA_INCL = -I${CUDA_HOME}/include
   CUDA_LIBS = -L${CUDA_HOME}/lib64 -lcudart -lcublas -lcusparse -lcurand -lstdc++ -L$(XL_DIR)/xlC/16.1.1/lib -libmc++
   CUDA_ARCH = -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70
   NVCC_LDFLAGS = -ccbin=${CXX} ${CUDA_ARCH}
   COPTS_CUDA = -DHYPRE_EXAMPLE_USING_CUDA
   FOPTS_CUDA = -DHYPRE_EXAMPLE_USING_CUDA -qsuppress=cmpmsg
endif

ifeq ($(use_cuf), 1)
   CUDA_LIBS = -L${CUDA_HOME}/lib64 -lcudart -lcublas -lcusparse -lcurand -lstdc++ -L$(XL_DIR)/xlC/16.1.1/lib -libmc++
endif

########################################################################
# Device OMP
########################################################################
ifeq ($(use_domp), 1)
   DOMP_LIBS = -L${CUDA_HOME}/lib64 -lcudart -lcublas -lcusparse -lcurand -lstdc++ -L$(XL_DIR)/xlC/16.1.1/lib -libmc++
   COPTS_DOMP = -DHYPRE_EXAMPLE_USING_DEVICE_OMP
   FOPTS_DOMP = -qoffload -W@,-v -qsmp=omp -qinfo=omperrtrace -DHYPRE_EXAMPLE_USING_DEVICE_OMP
   LOPTS_DOMP = -qoffload -W@,-v -qsmp=omp
endif

########################################################################
# Compiling and linking options
########################################################################
CINCLUDES = -I$(HYPRE_DIR)/include $(CUDA_INCL)
#CDEFS = -DHYPRE_EXVIS
CDEFS =
COPTS = -g -Wall $(COPTS_CUDA) $(COPTS_DOMP)
FOPTS = -g $(FOPTS_CUDA) $(FOPTS_DOMP)
CFLAGS = $(COPTS) $(CINCLUDES) $(CDEFS)
FINCLUDES = $(CINCLUDES)
FFLAGS = $(FOPTS) $(FINCLUDES)
CUFFLAGS = -qcuda
CXXOPTS = $(COPTS) -Wno-deprecated
CXXINCLUDES = $(CINCLUDES) -I..
CXXDEFS = $(CDEFS)
IFLAGS_BXX =
CXXFLAGS  = $(CXXOPTS) $(CXXINCLUDES) $(CXXDEFS) $(IFLAGS_BXX)
IF90FLAGS =
F90FLAGS = $(FFLAGS) $(IF90FLAGS)

LINKOPTS = $(LOPTS_CUDA) $(LOPTS_DOMP)
LIBS = -L$(HYPRE_DIR)/lib -lHYPRE -lm $(CUDA_LIBS) $(DOMP_LIBS)
LFLAGS = $(LINKOPTS) $(LIBS)
LFLAGS_B =\
 -L${HYPRE_DIR}/lib\
 -lbHYPREClient-C\
 -lbHYPREClient-CX\
 -lbHYPREClient-F\
 -lbHYPRE\
 -lsidl -ldl -lxml2
LFLAGS77 = $(LFLAGS)
LFLAGS90 =

########################################################################
# Rules for compiling the source files
########################################################################
.SUFFIXES: .c .f .cuf .cxx

.c.o:
	$(CC) $(CFLAGS) -c $<
.f.o:
	$(F77) $(FFLAGS) -c $<
.f.mod:
	$(F77) $(FFLAGS) -c $<
.cuf.o:
	$(CUF) $(FFLAGS) $(CUFFLAGS) -c $<
.cxx.o:
	$(CXX) $(CXXFLAGS) -c $<

########################################################################
# List of all programs to be compiled
########################################################################
ALLPROGS = ex1 ex2 ex3 ex4 ex5 ex5f ex6 ex7 ex8 ex9 ex11 ex12 ex12f \
           ex13 ex14 ex15 ex16
BIGINTPROGS = ex5big ex15big
FORTRANPROGS = ex5f ex12f
MAXDIMPROGS = ex17 ex18
COMPLEXPROGS = ex18comp

all: $(ALLPROGS)

default: all

gpu: all

bigint: $(BIGINTPROGS)

fortran: $(FORTRANPROGS)

maxdim: $(MAXDIMPROGS)

complex: $(COMPLEXPROGS)

########################################################################
# Example 1
########################################################################
ex1: ex1.o
	$(LINK_CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 2
########################################################################
ex2: ex2.o
	$(LINK_CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 3
########################################################################
ex3: ex3.o
	$(LINK_CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 4
########################################################################
ex4: ex4.o
	$(LINK_CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 5
########################################################################
ex5: ex5.o
	$(LINK_CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 5 with 64-bit integers
########################################################################
ex5big: ex5big.o
	$(LINK_CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 5 Fortran 77
########################################################################
ifeq ($(use_cuda), 1)
ex5f_cptr.o : ex5f_cptr.f cudaf.mod
	$(F77) $(FFLAGS) -c $<
ex5f: ex5f_cptr.o cudaf.o
	$(LINK_FC) -o $@ $^ $(LFLAGS77)
else
ifeq ($(use_cuf), 1)
ex5f: ex5cuf.o
	$(LINK_CUF) -o $@ $^ $(LFLAGS77)
else
ex5f: ex5f.o
	$(LINK_FC) -o $@ $^ $(LFLAGS77)
endif
endif

########################################################################
# Example 6
########################################################################
ex6: ex6.o
	$(LINK_CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 7
########################################################################
ex7: ex7.o
	$(LINK_CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 8
########################################################################
ex8: ex8.o
	$(LINK_CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 9
########################################################################
ex9: ex9.o
	$(LINK_CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 10
########################################################################
ex10: ex10.o
	$(LINK_CXX) -o $@ $^ $(LFLAGS)

########################################################################
# Example 11
########################################################################
ex11: ex11.o
	$(LINK_CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 12
########################################################################
ex12: ex12.o
	$(LINK_CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 12 Fortran 77
########################################################################
ifeq ($(use_cuda), 1)
ex12f_cptr.o : ex12f_cptr.f cudaf.mod
	$(F77) $(FFLAGS) -c $<
ex12f: ex12f_cptr.o cudaf.o
	$(LINK_FC) -o $@ $^ $(LFLAGS77)
else
ifeq ($(use_cuf), 1)
ex12f: ex12cuf.o
	$(LINK_CUF) -o $@ $^ $(LFLAGS77)
else
ex12f: ex12f.o
	$(LINK_FC) -o $@ $^ $(LFLAGS77)
endif
endif

########################################################################
# Example 13
########################################################################
ex13: ex13.o
	$(LINK_CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 14
########################################################################
ex14: ex14.o
	$(LINK_CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 15
########################################################################
ex15: ex15.o
	$(LINK_CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 15 with 64-bit integers
########################################################################
ex15big: ex15big.o
	$(LINK_CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 16
########################################################################
ex16: ex16.o
	$(LINK_CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 17
########################################################################
ex17: ex17.o
	$(LINK_CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 18
########################################################################
ex18: ex18.o
	$(LINK_CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 18 (complex)
########################################################################
ex18comp: ex18comp.o
	$(LINK_CC) -o $@ $^ $(LFLAGS)

########################################################################
# Clean up
########################################################################
clean:
	rm -f $(ALLPROGS:=.o)
	rm -f $(BIGINTPROGS:=.o)
	rm -f $(FORTRANPROGS:=.o)
	rm -f $(MAXDIMPROGS:=.o)
	rm -f $(COMPLEXPROGS:=.o)
	rm -f cudaf.o cudaf.mod ex*.o
	cd vis; make clean
distclean: clean
	rm -f $(ALLPROGS) $(ALLPROGS:=*~)
	rm -f $(BIGINTPROGS) $(BIGINTPROGS:=*~)
	rm -f $(FORTRANLPROGS) $(FORTRANPROGS:=*~)
	rm -f $(MAXDIMPROGS) $(MAXDIMPROGS:=*~)
	rm -f $(COMPLEXPROGS) $(COMPLEXPROGS:=*~)
	rm -fr README*
