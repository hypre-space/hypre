#BHEADER**********************************************************************
# Copyright (c) 2006   The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the HYPRE team. UCRL-CODE-222953.
# All rights reserved.
#
# This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
# Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
# disclaimer, contact information and the GNU Lesser General Public License.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free Software 
# Foundation) version 2.1 dated February 1999.
#
# HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
# WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
# $Revision$
#EHEADER**********************************************************************

include ../config/Makefile.config

# Python support.
# It is very important that you keep the Python modules up-to-date.
# They contain copies of hypre binaries taken from the libraries
# libHYPRE.so and libbHYPRE.so.
# When you first run make at the top level (linear_solvers directory),
# the libraries get built before the Python modules, so all is well.
# But if you later change something and rebuild in a subdirectory, the Python
# modules won't get automatically updated.  You need to cd to babel, and
# type "make" there too.

# The build process for the Python interface can be time- and space-consuming,
# so we don't want to do it unless requested.
ifeq ($(BUILD_PYTHON),1)
   BHYPRE_BABEL_PYTHON_DIR=bHYPREClient-P
endif

BHYPRE_BABEL_DIRS =\
 bHYPRE \
 bHYPREClient-C\
 bHYPREClient-F\
 bHYPREClient-CX\
 $(BHYPRE_BABEL_PYTHON_DIR)

HYPRE_BABEL_RUNTIME_SIDL_DIR = ${HYPRE_SRC_TOP_DIR}/babel-runtime/sidl

HYPRE_BABEL_RUNTIME_PYTHON_DIR = ${HYPRE_SRC_TOP_DIR}/babel-runtime/python

all:
	@ \
	for i in ${BHYPRE_BABEL_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    echo "Making $$i ..."; \
	    (cd $$i && $(MAKE) $@); \
	  fi; \
	done
	cp -fp ${HYPRE_BABEL_RUNTIME_SIDL_DIR}/*.h ${HYPRE_BUILD_DIR}/include
	cp -fp ${HYPRE_BABEL_RUNTIME_SIDL_DIR}/.libs/libs* ${HYPRE_BUILD_DIR}/lib
	cp -fp hypre_babel_exception_handler.h ${HYPRE_BUILD_DIR}/include

install:
	@ \
	for i in ${BHYPRE_BABEL_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    echo "Making $$i ..."; \
	    (cd $$i && $(MAKE) $@); \
	  fi; \
	done
	cp -fp ${HYPRE_BABEL_RUNTIME_SIDL_DIR}/*.h ${HYPRE_INC_INSTALL}
	cp -fp hypre_babel_exception_handler.h ${HYPRE_INC_INSTALL}

clean:
	@ \
	for i in ${BHYPRE_BABEL_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    (cd $$i && $(MAKE) $@); \
	  fi; \
	done

distclean:
	@ \
	for i in ${BHYPRE_BABEL_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    (cd $$i && $(MAKE) $@); \
	  fi; \
	done

