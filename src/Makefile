# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

default: all

# Include all variables defined by configure
include config/Makefile.config

# Tolerance level for checkpar tests
# The value of the variable can be set when calling `make checkpar`
HYPRE_CHECKPAR_TOL ?= 1e-6

# These are the directories for internal blas, lapack and general utilities
HYPRE_BASIC_DIRS =\
 blas\
 lapack\
 utilities

#These are the directories for multivector
HYPRE_MULTIVEC_DIRS =\
 multivector

# These are the directories for the generic Krylov solvers
HYPRE_KRYLOV_DIRS =\
 krylov

#These are the directories for the IJ interface
HYPRE_IJ_DIRS =\
 seq_mv\
 seq_block_mv\
 parcsr_mv\
 parcsr_block_mv\
 distributed_matrix\
 IJ_mv\
 matrix_matrix\
 parcsr_ls

#These are the directories for the structured interface
HYPRE_STRUCT_DIRS =\
 struct_mv\
 struct_ls

#These are the directories for the semi-structured interface
HYPRE_SSTRUCT_DIRS =\
 sstruct_mv\
 sstruct_ls

#These are the directories for the distributed_ls codes
HYPRE_DISTRIBUTED_LS_DIRS = ${HYPRE_DISTRIBUTED_LS_DIR}

#These are the directories for the FEI
HYPRE_FEI_DIRS = ${HYPRE_FEI_SRC_DIR}

#This is the lib directory
HYPRE_LIBS_DIRS = lib

#This is the documentation directory
HYPRE_DOCS_DIRS = docs

#This is the test-driver directory
HYPRE_TEST_DIRS = test

#This is the examples directory
HYPRE_EXAMPLE_DIRS = examples

# These are directories that are officially in HYPRE
HYPRE_DIRS =\
 ${HYPRE_BASIC_DIRS}\
 ${HYPRE_MULTIVEC_DIRS}\
 ${HYPRE_KRYLOV_DIRS}\
 ${HYPRE_IJ_DIRS}\
 ${HYPRE_STRUCT_DIRS}\
 ${HYPRE_SSTRUCT_DIRS}\
 ${HYPRE_DISTRIBUTED_LS_DIRS}\
 ${HYPRE_FEI_DIRS}\
 ${HYPRE_LIBS_DIRS}

# These are directories that are not yet officially in HYPRE
HYPRE_EXTRA_DIRS =\
 ${HYPRE_DOCS_DIRS}\
 ${HYPRE_TEST_DIRS}\
 seq_ls/pamg

#################################################################
# Targets
#################################################################

all:
	@ \
	mkdir -p ${HYPRE_BUILD_DIR}/include; \
	mkdir -p ${HYPRE_BUILD_DIR}/lib; \
	cp -fR HYPRE_config.h ${HYPRE_BUILD_DIR}/include/.; \
	cp -fR $(srcdir)/HYPRE.h ${HYPRE_BUILD_DIR}/include/.; \
	cp -fR $(srcdir)/HYPREf.h ${HYPRE_BUILD_DIR}/include/.; \
	for i in ${HYPRE_DIRS}; \
	do \
	  echo "Making $$i ..."; \
	  (cd $$i && $(MAKE) $@) || exit 1; \
	  echo ""; \
	done

help:
	@echo "     "
	@echo "************************************************************"
	@echo " HYPRE Make System Targets"
	@echo "   (using GNU-standards)"
	@echo "     "
	@echo "all:"
	@echo "     default target in all directories"
	@echo "     compile the entire program"
	@echo "     does not rebuild documentation"
	@echo "     "
	@echo "help:"
	@echo "     prints details of each target"
	@echo "     "
	@echo "install:"
	@echo "     compile the program and copy executables, libraries, etc"
	@echo "        to the file names where they reside for actual use"
	@echo "     executes mkinstalldirs script to create directories needed"
	@echo "     "
	@echo "clean:"
	@echo "     deletes all files from the current directory that are normally"
	@echo "        created by building the program"
	@echo "     "
	@echo "distclean:"
	@echo "     deletes all files from the current directory that are"
	@echo "        created by configuring or building the program"
	@echo "     "
	@echo "tags:"
	@echo "     runs etags to create tags table"
	@echo "     file is named TAGS and is saved in current directory"
	@echo "     "
	@echo "test:"
	@echo "     depends on the all target to be completed"
	@echo "     removes existing temporary installation sub-directory"
	@echo "     creates a temporary installation sub-directory"
	@echo "     copies all libHYPRE* and *.h files to the temporary locations"
	@echo "     builds the test drivers; linking to the temporary installation"
	@echo "        directories to simulate how application codes will link to HYPRE"
	@echo "     "
	@echo "check:"
	@echo "     runs a small driver test to verify a working library"
	@echo "     use CHECKRUN=<mpirun routine> if needed combined with"
	@echo "     PARMS='-P px py pz' where px*py*pz must be number of"
	@echo "     processes set in CHECKRUN"
	@echo "     "
	@echo "     "
	@echo "checkpar:"
	@echo "     runs several regression test to verify a working library."
	@echo "     Use parameter CHECKRUN='<mpirun routine -n>' if needed."
	@echo "     If CHECKRUN is not set, 'mpirun -np' is used."
	@echo "     "
	@echo "     "
	@echo "************************************************************"

test: all
	@ \
	echo "Making test drivers ..."; \
	(cd test; $(MAKE) clean; $(MAKE) all)

check:
	@ \
	echo "Checking the library ..."; \
	set -e; \
	(cd test; $(MAKE) all); \
	(cd test; $(CHECKRUN) ./ij $(PARMS) 2> ij.err); \
	(cd test; $(CHECKRUN) ./struct $(PARMS) 2> struct.err); \
	(cd test; cp -f TEST_sstruct/sstruct.in.default .; $(CHECKRUN) ./sstruct $(PARMS) 2> sstruct.err); \
	(cd test; ls -l ij.err struct.err sstruct.err)

checkpar:
	@ \
        echo "Checking the library ..."; \
        set -e; \
        (cd test; $(MAKE) all); \
        echo "Testing IJ ..."; \
        (cd test; ./runtest.sh -atol $(HYPRE_CHECKPAR_TOL) -mpi "$(CHECKRUN)" TEST_ij/solvers.sh); \
        (cd test; ./checktest.sh); \
        (cd test; ./cleantest.sh); \
        echo "Testing Struct ..."; \
        (cd test; ./runtest.sh -atol $(HYPRE_CHECKPAR_TOL) -mpi "$(CHECKRUN)" TEST_struct/solvers.sh); \
        (cd test; ./checktest.sh); \
        (cd test; ./cleantest.sh); \
        echo "Testing SStruct ..."; \
        (cd test; ./runtest.sh -atol $(HYPRE_CHECKPAR_TOL) -mpi "$(CHECKRUN)" TEST_sstruct/solvers.sh); \
        (cd test; ./checktest.sh); \
        (cd test; ./cleantest.sh);

install: all
	@ \
	echo "Installing hypre ..."; \
	${HYPRE_SRC_TOP_DIR}/config/mkinstalldirs ${HYPRE_LIB_INSTALL} ${HYPRE_INC_INSTALL}; \
	HYPRE_PWD=`pwd`; \
	cd ${HYPRE_BUILD_DIR}/lib; HYPRE_FROMDIR=`pwd`; \
	cd $$HYPRE_PWD; \
	cd ${HYPRE_LIB_INSTALL};   HYPRE_TODIR=`pwd`; \
	if [ "$$HYPRE_FROMDIR" != "$$HYPRE_TODIR" ]; \
	then \
	  cp -fR $$HYPRE_FROMDIR/* $$HYPRE_TODIR; \
	fi; \
	cd ${HYPRE_BUILD_DIR}/include; HYPRE_FROMDIR=`pwd`; \
	cd $$HYPRE_PWD; \
	cd ${HYPRE_INC_INSTALL};       HYPRE_TODIR=`pwd`; \
	if [ "$$HYPRE_FROMDIR" != "$$HYPRE_TODIR" ]; \
	then \
	  cp -fR $$HYPRE_FROMDIR/* $$HYPRE_TODIR; \
	fi; \
	cd $$HYPRE_PWD; \
	chmod -R a+rX,u+w,go-w ${HYPRE_LIB_INSTALL}; \
	chmod -R a+rX,u+w,go-w ${HYPRE_INC_INSTALL}; \
	echo

clean:
	@ \
	rm -Rf hypre; \
	for i in ${HYPRE_DIRS} ${HYPRE_EXTRA_DIRS} ${HYPRE_EXAMPLE_DIRS}; \
	do \
	  if [ -f $$i/Makefile ]; \
	  then \
	    echo "Cleaning $$i ..."; \
	    (cd $$i && $(MAKE) $@) || exit 1; \
	  fi; \
	done
	rm -rf tca.map pchdir *inslog*

distclean:
	@ \
	rm -Rf hypre; \
	for i in ${HYPRE_DIRS} ${HYPRE_EXTRA_DIRS} ${HYPRE_EXAMPLE_DIRS} examples/docs; \
	do \
	  if [ -d $$i ]; \
	  then \
	    echo "Dist-Cleaning $$i ..."; \
	    (cd $$i && $(MAKE) $@) || exit 1; \
	  fi; \
	done
	rm -rf tca.map pchdir *inslog*
	rm -rf ./config/Makefile.config
	rm -rf ./TAGS
	rm -rf ./autom4te.cache
	rm -rf ./config.log
	rm -rf ./config.status
	rm -rf ./HYPRE_config.h

tags:
	find . -name "*.c" -o -name "*.C" -o -name "*.h" -o \
	-name "*.c??" -o -name "*.h??" -o -name "*.f" | etags -
