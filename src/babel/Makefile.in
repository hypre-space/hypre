#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 2.26 $
#EHEADER***********************************************************************

include $(top_srcdir)/config/Makefile.config

HYPRE_BABEL_DIRS =\
 bHYPRE \
 bHYPREClient-C\
 bHYPREClient-F

all: 
	@ \
	for i in ${HYPRE_BABEL_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    echo "Making $$i ..."; \
	    (cd $$i && $(MAKE) install); \
	  fi; \
	done;

install: all

clean:
	@ \
	for i in ${HYPRE_BABEL_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    (cd $$i && $(MAKE) $@); \
	  fi; \
	done

veryclean:
	@ \
	for i in ${HYPRE_BABEL_DIRS}; \
	do \
	  if [ -d $$i ]; \
	  then \
	    (cd $$i && $(MAKE) $@); \
	  fi; \
	done

