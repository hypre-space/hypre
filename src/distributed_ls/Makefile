# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../config/Makefile.config

HYPRE_DIST_LS_DIRS =\
 pilut\
 ParaSails\
 Euclid

all:
	@ \
	for i in ${HYPRE_DIST_LS_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    echo "Making $$i ..."; \
	    (cd $$i && $(MAKE) $@) || exit 1; \
	  fi; \
	done

install:
	@ \
	for i in ${HYPRE_DIST_LS_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    echo "Making $$i ..."; \
	    (cd $$i && $(MAKE) $@) || exit 1; \
	  fi; \
	done

clean:
	@ \
	for i in ${HYPRE_DIST_LS_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    (cd $$i && $(MAKE) $@) || exit 1; \
	  fi; \
	done

distclean:
	@ \
	for i in ${HYPRE_DIST_LS_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    (cd $$i && $(MAKE) $@) || exit 1; \
	  fi; \
	done
