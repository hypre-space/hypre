#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

include ../config/Makefile.config

# These are directories that are not yet officially in HYPRE
HYPRE_EXTRA_DIRS =\
 ParILUT

all: 
	@ \
	for i in ${HYPRE_EXTRA_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    echo "Making $$i ..."; \
	    (cd $$i && $(MAKE) $@); \
	    echo ""; \
	  fi; \
	done; \
	echo ""

install: all
	@ \
	echo ""

clean:
	@ \
	for i in ${HYPRE_EXTRA_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    (cd $$i && $(MAKE) $@); \
	  fi; \
	done

distclean:
	@ \
	for i in ${HYPRE_EXTRA_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    (cd $$i && $(MAKE) $@); \
	  fi; \
	done
