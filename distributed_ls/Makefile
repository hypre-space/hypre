#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

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
	    (cd $$i && $(MAKE) $@); \
	  fi; \
	done

install:
	@ \
	for i in ${HYPRE_DIST_LS_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    echo "Making $$i ..."; \
	    (cd $$i && $(MAKE) $@); \
	  fi; \
	done

clean:
	@ \
	for i in ${HYPRE_DIST_LS_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    (cd $$i && $(MAKE) $@); \
	  fi; \
	done

distclean:
	@ \
	for i in ${HYPRE_DIST_LS_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    (cd $$i && $(MAKE) $@); \
	  fi; \
	done
