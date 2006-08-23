#BHEADER**********************************************************************
# Copyright (c) 2006   The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
# All rights reserved.
#
# This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
# Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
# disclaimer and the GNU Lesser General Public License.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
# $Revision$
#EHEADER**********************************************************************




include ../config/Makefile.config

BHYPRE_BABEL_DIRS =\
 bHYPRE \
 bHYPREClient-C\
 bHYPREClient-F\
 bHYPREClient-CX

# bHYPREClient-P

HYPRE_BABEL_RUNTIME_SIDL_DIR = ${HYPRE_SRC_TOP_DIR}/babel-runtime/sidl

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

