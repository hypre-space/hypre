#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

include ../config/Makefile.config

HYPRE_PARCSR_ES_DIRS = LOBPCG

all:
	@ \
	for i in ${HYPRE_DIST_ES_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    echo "Making $$i ..."; \
	    (cd $$i&& $(MAKE)); \
	  fi; \
	done

install: all

clean:
	@ \
	for i in ${HYPRE_DIST_ES_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    (cd $$i && $(MAKE) $@); \
	  fi; \
	done

distclean:
	@ \
	for i in ${HYPRE_DIST_ES_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    (cd $$i && $(MAKE) $@); \
	  fi; \
	done
