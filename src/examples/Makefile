# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

####################################################################
# Compiler and External Dependencies
####################################################################

# Include the main Makefile configuration
include ../config/Makefile.config

# CUDA fortran (cuf) compiler
ifeq ($(use_cuf), 1)
   CUF      = mpixlcuf
   LINK_CUF = ${CUF}
   CUFFLAGS = $(FFLAGS) -qcuda
endif

####################################################################
# Compiler and Linker Flags
####################################################################

# Compiler Definitions
ifeq ($(with_vis), 1)
   CDEFS = -DHYPRE_EXVIS
endif
CDEFS   += -DHYPRE_TIMING
CXXDEFS += $(CDEFS)
FDEFS   += -DHYPRE_FORTRAN

# Include Paths
CINCLUDES   = -I${HYPRE_BUILD_DIR}/include\
              $(INCLUDES)\
              $(MPIINCLUDE)\
              $(SUPERLU_INCLUDE)\
              $(DSUPERLU_INCLUDE)

CXXINCLUDES = -I${HYPRE_BUILD_DIR}/include\
              -I$(srcdir)/../FEI_mv/fei-base\
              $(INCLUDES)\
	          $(MPIINCLUDE)\
              $(SUPERLU_INCLUDE)\
              $(DSUPERLU_INCLUDE)

# Hypre library
LIBS = -L$(HYPRE_BUILD_DIR)/lib -lHYPRE -lm

# GPU Options
GPUOPTS =

# Set specific GPU options according to the backend
ifdef HYPRE_CUDA_INCLUDE
    GPUOPTS += -DHYPRE_EXAMPLE_USING_CUDA $(HYPRE_CUDA_INCLUDE)
    LIBS += $(HYPRE_CUDA_LIBS) -lstdc++
    use_cuda = 1

else ifdef HYPRE_HIP_INCLUDE
    GPUOPTS += -DHYPRE_EXAMPLE_USING_HIP -D__HIP_PLATFORM_AMD__ $(HYPRE_HIP_INCLUDE)
    LIBS += $(HYPRE_HIP_LIBS) -lstdc++

else ifdef HYPRE_SYCL_INCLUDE
    GPUOPTS += -DHYPRE_EXAMPLE_USING_SYCL $(HYPRE_SYCL_INCLUDE)
    LIBS += $(HYPRE_SYCL_LIBS)

else ifdef HYPRE_RAJA_INCLUDE
    LIBS += $(HYPRE_RAJA_LIB)

else ifdef HYPRE_KOKKOS_INCLUDE
    LIBS += $(HYPRE_KOKKOS_LIB)
endif

# Compiler Flags
CFLAGS    += $(GPUOPTS) $(CDEFS) $(CINCLUDES)
CXXFLAGS  += $(GPUOPTS) $(CXXDEFS) $(CXXINCLUDES)
FFLAGS    += $(GPUOPTS) $(FDEFS) $(CINCLUDES)

# Dependency libraries
LIBS  += $(CALIPER_LIBS)\
         $(HYPRE_UMPIRE_LIB)\
         $(HYPRE_MAGMA_LIB)
LFLAGS = $(LIBS)

####################################################################
# Target Definitions
####################################################################

# C Example Executables
C_EXAMPLES       = ex01 ex02 ex03 ex04 ex05 ex06 ex07 ex08\
                   ex09 ex11 ex12 ex13 ex14 ex15 ex16
FORTRAN_EXAMPLES = ex05f ex12f
BIGINT_EXAMPLES  = ex05big ex15big
MAXDIM_EXAMPLES  = ex17 ex18
COMPLEX_EXAMPLES = ex18comp

EXAMPLES = $(C_EXAMPLES) $(FORTRAN_EXAMPLES) $(BIGINT_EXAMPLES)\
           $(MAXDIM_EXAMPLES) $(COMPLEX_EXAMPLES)

####################################################################
# Build Targets
####################################################################

all: $(C_EXAMPLES)

default: all

gpu: $(C_EXAMPLES) $(FORTRAN_EXAMPLES)

bigint: $(BIGINT_EXAMPLES)

fortran: $(FORTRAN_EXAMPLES)

maxdim: $(MAXDIM_EXAMPLES)

complex: $(COMPLEX_EXAMPLES)

####################################################################
# Compilation Rules
####################################################################

# Compilation Rules for C Programs
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compilation Rules for C++ Programs
%.o: %.cxx
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compilation Rules for Fortran Programs
.f.mod:
	$(F77) $(FFLAGS) -c $<
%.mod: cudaf.f
	$(F77) $(FFLAGS) -c $<
.cuf.o:
	$(CUF) $(FFLAGS) $(CUFFLAGS) -c $<

####################################################################
# Linking Rules
####################################################################

# Linking C Executables
$(C_EXAMPLES): %: %.o
	@echo "Building $@ ..."
	$(LINK_CC) -o $@ $< $(LFLAGS)

$(BIGINT_EXAMPLES): %: %.o
	@echo "Building $@ ..."
	$(LINK_CC) -o $@ $< $(LFLAGS)

$(MAXDIM_EXAMPLES): %: %.o
	@echo "Building $@ ..."
	$(LINK_CC) -o $@ $< $(LFLAGS)

# Linking C++ Executables
$(CXX_EXAMPLES): %: cxx_%.o
	@echo "Building $@ ..."
	$(LINK_CXX) -o $@ $^ $(LFLAGS)

####################################################################
# Installation Target
####################################################################

install: all
	@echo "Installing examples..."
	@mkdir -p $(HYPRE_INSTALL_DIR)/bin
	@cp -f $(EXAMPLES) $(HYPRE_INSTALL_DIR)/bin/
	@chmod +x $(HYPRE_INSTALL_DIR)/bin/*
	@echo "Installation complete."

####################################################################
# Example-Specific Rules
####################################################################

# Example 7 Fortran with CUDA or CUF Support
ifeq ($(use_cuda), 1)
ex05f_cptr.o : ex05f_cptr.f cudaf.mod
	$(F77) $(FFLAGS) -c $<
ex05f: ex05f_cptr.o cudaf.o
	$(LINK_FC) -o $@ $^ $(LFLAGS)
else
ifeq ($(use_cuf), 1)
ex05f: ex05cuf.o
	$(LINK_CUF) -o $@ $^ $(LFLAGS)
else
ex05f: ex05f.o
	$(LINK_FC) -o $@ $^ $(LFLAGS)
endif
endif

# Example 12 Fortran with CUDA or CUF Support
ifeq ($(use_cuda), 1)
ex12f_cptr.o : ex12f_cptr.f cudaf.mod
	$(F77) $(FFLAGS) -c $<
ex12f: ex12f_cptr.o cudaf.o
	$(LINK_FC) -o $@ $^ $(LFLAGS)
else
ifeq ($(use_cuf), 1)
ex12f: ex12cuf.o
	$(LINK_CUF) -o $@ $^ $(LFLAGS)
else
ex12f: ex12f.o
	$(LINK_FC) -o $@ $^ $(LFLAGS)
endif
endif

# Add more specific rules as needed
# Ensure that dependencies like cudaf.mod are correctly handled

########################################################################
# Clean Targets
########################################################################
clean:
	@rm -fv $(EXAMPLES:=.o)
	@rm -fv $(BIGINT_EXAMPLES:=.o)
	@rm -fv $(FORTRAN_EXAMPLES:=.o)
	@rm -fv $(MAXDIM_EXAMPLES:=.o)
	@rm -fv $(COMPLEX_EXAMPLES:=.o)
	@rm -fv cudaf.o cudaf.mod ex*.o
	@cd vis; make clean

distclean: clean
	@rm -fv $(EXAMPLES) $(EXAMPLES:=*~)
	@rm -fv $(BIGINT_EXAMPLES) $(BIGINT_EXAMPLES:=*~)
	@rm -fv $(FORTRAN_EXAMPLES) $(FORTRAN_EXAMPLES:=*~)
	@rm -fv $(MAXDIM_EXAMPLES) $(MAXDIM_EXAMPLES:=*~)
	@rm -fv $(COMPLEX_EXAMPLES) $(COMPLEX_EXAMPLES:=*~)

.PHONY: clean distclean install
