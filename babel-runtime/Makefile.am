##
## File:        Makefile.am 
## Package:     Babel makefiles
## Release:     $Name$
## Revision:    $Revision$
## Modified:    $Date$
## Description: automake makefile to build the runtime library
##
## Copyright (c) 2000-2002, The Regents of the University of Calfornia.
## Produced at the Lawrence Livermore National Laboratory.
## Written by the Components Team <components@llnl.gov>
## UCRL-CODE-2002-054
## All rights reserved.
## 
## This file is part of Babel. For more information, see
## http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
## for Our Notice and the LICENSE file for the GNU Lesser General Public
## License.
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU Lesser General Public License (as published by
## the Free Software Foundation) version 2.1 dated February 1999.
## 
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
## conditions of the GNU Lesser General Public License for more details.
## 
## You should have recieved a copy of the GNU Lesser General Public License
## along with this program; if not, write to the Free Software Foundation,
## Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

AUTOMAKE_OPTIONS = foreign no-dependencies no-installinfo	\
                   no-installman no-texinfo.tex 1.7

SUBDIRS = bin config sidl java

#
# Unfortuntately, Python uses its own makefile system so we have to 
# deal with it explicitly here in the parent directory instead of
# integrating it inside automake.
#

if SUPPORT_CYGWIN
SO = dll
else
SO = so
endif

pydir = $(exec_prefix)/python
PYTHONLIB = @PYTHONLIB@
PYTHONISHFILES = python/SIDLObjA.h python/SIDLObjA.c			\
		 python/SIDLPyArrays.h python/SIDLPyArrays.c		\
		 python/sidlsetup.py python/__init__.py

RUNTIME_ONLY_FILES=python/SIDL/__init__.py \
		python/SIDL/BaseClass_Module.c \
		python/SIDL/BaseException_Module.c \
		python/SIDL/BaseInterface_Module.c \
		python/SIDL/DLL_Module.c \
		python/SIDL/Loader_Module.c \
		python/SIDL_BaseClass_Module.h \
		python/SIDL_BaseException_Module.h \
		python/SIDL_BaseInterface_Module.h \
		python/SIDL_DLL_Module.h \
		python/SIDL_Loader_Module.h \
		python/SIDL_BaseClass_IOR.h python/SIDL_BaseClass_IOR.c \
		python/SIDL_BaseException_IOR.h python/SIDL_BaseException_IOR.c \
		python/SIDL_BaseInterface_IOR.h python/SIDL_BaseInterface_IOR.c \
		python/SIDL_DLL_IOR.h python/SIDL_DLL_IOR.c \
		python/SIDL_Loader_IOR.h python/SIDL_Module_IOR.c \
		python/setup.py

if BABEL_RUNTIME_ONLY
all-local-python: all-local-python2
clean-local-python: clean-local-python2
else
all-local-python: all-local-python1
clean-local-python: clean-local-python1
endif

dist-hook: sidl/libsidl.la dist-hook-python

if SUPPORT_PYTHON
all-local: all-local-python
clean-local: clean-local-python
install-data-am: install-data-am-python
uninstall-am: uninstall-data-am-python
else
all-local clean-local install-data-am uninstall-am:
	@echo "Python not supported"
endif

sidl/libsidl.la:
	@-cd sidl && $(MAKE) libsidl.la

all-local-python1: python/babel-stamp
python/babel-stamp: $(PYTHONISHFILES)
	@-if ! test -d python; then mkdir -p python || exit 1; fi
	@-if ! test -d python/SIDL; then mkdir -p python/SIDL || exit 1; fi
	@-if test "X$(srcdir)" != "X."; then				\
	  for file in $(PYTHONISHFILES); do				\
	    d=`dirname $$file`;						\
	    test -f $$d							\
	    || mkdir -p $$d						\
	    || exit 1;							\
	    echo cp -p $(srcdir)/$$file $$file;				\
	    cp -p $(srcdir)/$$file $$file;				\
	  done;								\
	fi
	@if test -d python; then					\
	  abspath=`cd $(srcdir) && pwd`;				\
	  cd python;							\
	  echo $(SHELL) ../../bin/babel --suppress-timestamp -!		\
	    --generate-sidl-stdlib -cpython				\
	    $$abspath/sidl/sidl.sidl;					\
	  $(SHELL) ../../bin/babel --suppress-timestamp	 -!		\
	    --generate-sidl-stdlib -cpython  				\
	    $$abspath/sidl/sidl.sidl;					\
	  $(PYTHON) sidlsetup.py --library-dirs=../sidl/.libs		\
	   --include-dirs=../sidl --include-dirs=../config  build_ext   \
	   --inplace;							\
	  $(PYTHON) setup.py --library-dirs=../sidl/.libs		\
	   --include-dirs=../sidl --include-dirs=../config build_ext 	\
	   --inplace;							\
	fi
	touch python/babel-stamp

all-local-python2: python/babel-stamp2
python/babel-stamp2:
	@-if ! test -d python; then mkdir -p python || exit 1; fi
	@-if ! test -d python/SIDL; then mkdir -p python/SIDL || exit 1; fi
	@-if test "X$(srcdir)" != "X."; then				\
	  for file in $(PYTHONISHFILES) $(RUNTIME_ONLY_FILES); do	\
	    d=`dirname $$file`;						\
	    test -f $$d							\
	    || mkdir -p $$d						\
	    || exit 1;							\
	    echo cp -p $(srcdir)/$$file $$file;				\
	    cp -p $(srcdir)/$$file $$file;				\
	  done;								\
	fi
	@if test -d python; then					\
	  cd python;							\
	  $(PYTHON) sidlsetup.py --library-dirs=../sidl/.libs		\
	   --include-dirs=$(srcdir)/../sidl --include-dirs=../config  build_ext   \
	   --inplace;							\
	  $(PYTHON) setup.py --library-dirs=../sidl/.libs		\
	   --include-dirs=$(srcdir)/../sidl --include-dirs=../config build_ext 	\
	   --inplace;							\
	fi
	touch python/babel-stamp2

clean-local-python1:
	@if test -d python; then					\
          cd python;                                                    \
          echo rm -f sedscript Makefile Makefile.pre Makefile.pre.in    \
            Setup config.c *setup.installed;                            \
          rm -f sedscript Makefile Makefile.pre Makefile.pre.in Setup   \
             config.c *setup.installed;         			\
          echo rm -f babel-stamp *.so *.dll *.a *.o *~;                 \
	  rm -f babel-stamp *.so *.dll *.a *.o *~;			\
	  if test -d SIDL; then						\
	    cd SIDL;							\
	    echo rm -f config.c *.so *.dll *.a *.o *~ *.c *.h __init__.*;\
	    rm -f config.c *.so *.dll *.a *.o *~ *.c *.h __init__.*;	\
	  fi;								\
	fi
	@-if test "X$(srcdir)" != "X."; then				\
	  echo rm -rf python;					\
	  rm -rf python;						\
	else \
	  echo rm -rf python/build; \
	  rm -rf python/build; \
	fi

clean-local-python2:
	@if test -d python; then					\
          cd python;                                                    \
          echo rm -f sedscript Makefile Makefile.pre Makefile.pre.in    \
            Setup config.c *setup.installed;                            \
          rm -f sedscript Makefile Makefile.pre Makefile.pre.in Setup   \
             config.c *setup.installed;         			\
          echo rm -f babel-stamp *.so *.dll *.a *.o *~;                 \
	  rm -f babel-stamp *.so *.dll *.a *.o *~;			\
	  if test -d SIDL; then						\
	    cd SIDL;							\
	    echo rm -f config.c *.so *.dll *.a *.o *~  ;		\
	    rm -f config.c *.so *.dll *.a *.o *~;			\
	  fi;								\
	fi
	@-if test "X$(srcdir)" != "X."; then				\
	  echo rm -rf python;					\
	  rm -rf python;						\
	else \
	  echo rm -rf python/build; \
	  rm -rf python/build; \
	fi

dist-hook-python: all-local-python
	@for file in $(PYTHONISHFILES); do				\
	  d=`dirname $(distdir)/$$file`;				\
	  test -f $$d							\
	  || mkdir -p $$d						\
	  || exit 1;							\
	  echo cp -p $(srcdir)/$$file $(DESTDIR)$(distdir)/$$file;	\
	  cp -p $(srcdir)/$$file $(DESTDIR)$(distdir)/$$file;		\
	done
	@for file in $(RUNTIME_ONLY_FILES); do				\
	  d=`dirname $(distdir)/$$file`;				\
	  test -f $$d							\
	  || mkdir -p $$d						\
	  || exit 1;							\
	  echo cp -p $$file $(DESTDIR)$(distdir)/$$file;		\
	  cp -p $$file $(DESTDIR)$(distdir)/$$file;			\
	done

install-data-am-python:
	@$(NORMAL_INSTALL)
	test -d python || exit 1
	cd python ;							\
	$(PYTHON) sidlsetup.py --library-dirs=../sidl/.libs 		\
	   --include-dirs=../sidl --include-dirs=../config install 	\
	   --prefix=$(DESTDIR)$(prefix) 				\
	   --exec-prefix=$(DESTDIR)$(exec_prefix)			\
	   --record=sidlsetup.installed;				\
	$(PYTHON) setup.py --library-dirs=../sidl/.libs 		\
	   --include-dirs=../sidl  --include-dirs=../config install  	\
	   --prefix=$(DESTDIR)$(prefix) 				\
	   --exec-prefix=$(DESTDIR)$(exec_prefix)			\
	   --record=setup.installed

uninstall-data-am-python:
	@$(NORMAL_UNINSTALL)
	@test -d python && test -f python/setup.installed &&		\
	 rm -f `cat python/setup.installed`
	@test -d python && test -f python/sidlsetup.installed &&	\
	 rm -f `cat python/sidlsetup.installed`
