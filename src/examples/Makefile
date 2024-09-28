# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

####################################################################
# Compiler and External Dependencies
####################################################################

# Include the main Makefile configuration
include ../config/Makefile.config

# Compiler Definitions
CC        = mpicc
F77       = mpif77
CXX       = mpicxx
F90       = mpif90
CUF       = mpixlcuf
LINK_CC   = $(CXX)
LINK_CXX  = $(CXX)
LINK_FC   = $(F77)
LINK_CUF  = $(CUF)

####################################################################
# Feature Detection and Flag Configuration
####################################################################

# Initialize Feature Flags
USE_CUDA = 0
USE_HIP  = 0
USE_SYCL = 0

# Feature Detection and Flag Configuration using elseif for mutual exclusivity
ifeq ($(strip $(HYPRE_WITH_CUDA)),ON)
    USE_CUDA = 1
else ifeq ($(strip $(HYPRE_WITH_HIP)),ON)
    USE_HIP  = 1
else ifeq ($(strip $(HYPRE_WITH_SYCL)),ON)
    USE_SYCL = 1
endif

####################################################################
# Compiler and Linker Flags
####################################################################

# Include Paths
CINCLUDES   = -I$(HYPRE_BUILD_DIR)/include\
              -I$(CALIPER_DIR)/include\
              -I$(HYPRE_CUDA_DIR)/include\
              -I$(HYPRE_HIP_DIR)/include\
              -I$(HYPRE_SYCL_DIR)/include\
              -I$(HYPRE_RAJA_DIR)/include\
              -I$(HYPRE_KOKKOS_DIR)/include\
              -I$(HYPRE_UMPIRE_DIR)/include\
              -I$(HYPRE_MAGMA_DIR)/include\
              -I$(HYPRE_NAP_DIR)/include
CXXINCLUDES = $(CINCLUDES)

# Compiler Definitions
CDEFS   = -DHYPRE_TIMING
CXXDEFS = $(CDEFS)
FDEFS   = -DHYPRE_FORTRAN

# Feature-Specific Compiler Options
COPTS =
FOPTS = 

ifeq ($(USE_CUDA),1)
    COPTS += -DHYPRE_EXAMPLE_USING_CUDA
    FOPTS += -DHYPRE_EXAMPLE_USING_CUDA
endif

ifeq ($(USE_HIP),1)
    COPTS += -DHYPRE_EXAMPLE_USING_HIP
    FOPTS += -DHYPRE_EXAMPLE_USING_HIP
endif

ifeq ($(USE_SYCL),1)
    COPTS += -DHYPRE_EXAMPLE_USING_SYCL
    FOPTS += -DHYPRE_EXAMPLE_USING_SYCL
endif

# Compiler Flags
CFLAGS    += $(COPTS) $(CINCLUDES) $(CDEFS)
CXXFLAGS  += $(COPTS) $(CXXINCLUDES) $(CXXDEFS)
FFLAGS    += $(FOPTS) $(CINCLUDES) $(FDEFS)

# Linker Flags
LIBS   = -L$(HYPRE_BUILD_DIR)/lib\
          -lHYPRE\
          -lm\
          $(CALIPER_LIBS)\
          $(HYPRE_CUDA_LIBS)\
          $(HYPRE_HIP_LIBS)\
          $(HYPRE_SYCL_LIBS)\
          $(HYPRE_RAJA_LIB)\
          $(HYPRE_KOKKOS_LIB)\
          $(HYPRE_UMPIRE_LIB)\
          $(HYPRE_MAGMA_LIB)
LFLAGS = $(LIBS)

####################################################################
# Target Definitions
####################################################################

# Example Executables
EXAMPLES =\
    ex12f\
    ex13\
    ex14\
    ex15\
    ex15big\
    ex16\
    ex17\
    ex18\
    ex18comp

# Fortran Examples
FORTRAN_EXAMPLES =\
    ex12f\
    ex1_for\
    ex3_for\
    ex5_for\
    ex6_for\
    ex7_for

# C++ Examples
CXX_EXAMPLES =\
    cxx_ij\
    cxx_sstruct\
    cxx_struct

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
%.o: %.f
	$(F77) $(FFLAGS) -c $< -o $@

####################################################################
# Linking Rules
####################################################################

# Linking C Executables
%: %.o
	@echo "Building $@ ..."
	$(LINK_CC) -o $@ $^ $(LFLAGS)

# Linking C++ Executables
cxx_%: cxx_%.o
	@echo "Building $@ ..."
	$(LINK_CXX) -o $@ $^ $(LFLAGS)

# Linking Fortran Executables
f77_%: %.o
	@echo "Building $@ ..."
	$(LINK_FC) -o $@ $^ $(LFLAGS)

####################################################################
# Build Targets
####################################################################

all: $(EXAMPLES) $(CXX_EXAMPLES) $(FORTRAN_EXAMPLES)

# Phony Targets
.PHONY: clean distclean install

####################################################################
# Clean Targets
####################################################################

clean:
	rm -f *.o $(EXAMPLES) $(CXX_EXAMPLES) $(FORTRAN_EXAMPLES)

distclean: clean
	rm -f *.mod cudaf.o

####################################################################
# Installation Target
####################################################################

install: all
	@echo "Installing examples..."
	@mkdir -p $(HYPRE_INSTALL_DIR)/bin
	@cp -f $(EXAMPLES) $(CXX_EXAMPLES) $(FORTRAN_EXAMPLES) $(HYPRE_INSTALL_DIR)/bin/
	@chmod +x $(HYPRE_INSTALL_DIR)/bin/*
	@echo "Installation complete."

####################################################################
# Example-Specific Rules
####################################################################

# Example 12 Fortran with CUF Support
ifeq ($(USE_CUF),1)
ex12f: ex12cuf.o
	@echo "Building $@ ..."
	$(LINK_CUF) -o $@ $^ $(LFLAGS)
else
ex12f: ex12f.o
	@echo "Building $@ ..."
	$(LINK_FC) -o $@ $^ $(LFLAGS)
endif

# Add more specific rules as needed
# Ensure that dependencies like cudaf.mod are correctly handled