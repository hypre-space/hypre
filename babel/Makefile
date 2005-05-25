#BHEADER***********************************************************************
# (c) 2000   The Regents of the University of California
#
# See the file COPYRIGHT for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

include ../config/Makefile.config

BHYPRE_BABEL_DIRS =\
 bHYPRE \
 bHYPREClient-C\
 bHYPREClient-F

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

