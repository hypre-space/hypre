#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 2.38 $
#EHEADER**********************************************************************


# Include all variables defined by configure
include config/Makefile.config

# These are the directories for internal blas, lapack and general utilities
HYPRE_BASIC_DIRS =\
 ${HYPRE_BLAS_SRC_DIR}\
 ${HYPRE_LAPACK_SRC_DIR}\
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
 parcsr_mv\
 parcsr_block_mv\
 distributed_matrix\
 IJ_mv\
 matrix_matrix\
 distributed_ls\
 parcsr_ls

#These are the directories for the structured interface
HYPRE_STRUCT_DIRS =\
 struct_mv\
 struct_ls

#These are the directories for the semi-structured interface
HYPRE_SSTRUCT_DIRS =\
 sstruct_mv\
 sstruct_ls

#These are the directories for the FEI
HYPRE_FEI_DIRS = ${HYPRE_FEI_SRC_DIR}

#This is the lib directory
HYPRE_LIBS_DIRS = lib

#This is the documentation directory
HYPRE_DOCS_DIRS = docs

#This is the test-driver directory
HYPRE_TEST_DIRS = test

# These are directories that are officially in HYPRE
HYPRE_DIRS =\
 ${HYPRE_BASIC_DIRS}\
 ${HYPRE_MULTIVEC_DIRS}\
 ${HYPRE_KRYLOV_DIRS}\
 ${HYPRE_IJ_DIRS}\
 ${HYPRE_STRUCT_DIRS}\
 ${HYPRE_SSTRUCT_DIRS}\
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
	cp -fpPR HYPRE_config.h ${HYPRE_BUILD_DIR}/include/.; \
	cp -fpPR $(srcdir)/HYPRE.h ${HYPRE_BUILD_DIR}/include/.; \
	cp -fpPR $(srcdir)/HYPREf.h ${HYPRE_BUILD_DIR}/include/.; \
	for i in ${HYPRE_DIRS} ${HYPRE_BABEL_DIRS} ${HYPRE_EXAMPLE_DIRS}; \
	do \
	  echo "Making $$i ..."; \
	  (cd $$i && $(MAKE) $@); \
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
	@echo "     use CHECKRUN=<mpirun routine> if needed"
	@echo "     "
	@echo "************************************************************"

test: all
	@ \
	echo "Making test drivers ..."; \
	(cd test; $(MAKE) clean; $(MAKE) all)

check:
	@ \
	echo "Checking the library ..."; \
	(cd test; $(MAKE) all); \
	(cd test; $(CHECKRUN) ij 2> ij.err); \
	(cd test; $(CHECKRUN) struct 2> struct.err); \
	(cd test; cp -f TEST_sstruct/sstruct.in.default .; $(CHECKRUN) sstruct 2> sstruct.err); \
	(cd test; ls -l ij.err struct.err sstruct.err)

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
	  cp -fpPR $$HYPRE_FROMDIR/* $$HYPRE_TODIR; \
	fi; \
	cd ${HYPRE_BUILD_DIR}/include; HYPRE_FROMDIR=`pwd`; \
	cd $$HYPRE_PWD; \
	cd ${HYPRE_INC_INSTALL};       HYPRE_TODIR=`pwd`; \
	if [ "$$HYPRE_FROMDIR" != "$$HYPRE_TODIR" ]; \
	then \
	  cp -fpPR $$HYPRE_FROMDIR/* $$HYPRE_TODIR; \
	fi; \
	cd $$HYPRE_PWD; \
	chmod -R a+rX,u+w,go-w ${HYPRE_LIB_INSTALL}; \
	chmod -R a+rX,u+w,go-w ${HYPRE_INC_INSTALL}; \
	echo

clean:
	@ \
	rm -Rf hypre; \
	for i in ${HYPRE_DIRS} ${HYPRE_EXTRA_DIRS} ${HYPRE_BABEL_DIRS} ${HYPRE_EXAMPLE_DIRS}; \
	do \
	  if [ -f $$i/Makefile ]; \
	  then \
	    echo "Cleaning $$i ..."; \
	    (cd $$i && $(MAKE) $@); \
	  fi; \
	done
	rm -rf tca.map pchdir *inslog*

distclean:
	@ \
	rm -Rf hypre; \
	for i in ${HYPRE_DIRS} ${HYPRE_EXTRA_DIRS} ${HYPRE_BABEL_DIRS} ${HYPRE_EXAMPLE_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    echo "Dist-Cleaning $$i ..."; \
	    (cd $$i && $(MAKE) $@); \
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
