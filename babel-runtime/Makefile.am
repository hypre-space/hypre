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

if WITH_SIDLX
SIDLX_SUBDIR=sidlx
else
SIDLX_SUBDIR=
endif

SUBDIRS = m4 bin config sidl java $(SIDLX_SUBDIR)

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
PYTHONISHFILES = python/sidlObjA.h python/sidlObjA.c			\
		 python/sidlPyArrays.h python/sidlPyArrays.c		\
		 python/sidlsetup.py python/__init__.py

RUNTIME_ONLY_FILES=python/sidl/__init__.py \
		python/sidl/BaseClass_Module.c \
		python/sidl/BaseException_Module.c \
		python/sidl/BaseInterface_Module.c \
		python/sidl/DLL_Module.c \
		python/sidl/Loader_Module.c \
		python/sidl/sidlException_Module.c \
		python/sidl_BaseClass_Module.h \
		python/sidl_BaseException_Module.h \
		python/sidl_BaseInterface_Module.h \
		python/sidl_DLL_Module.h \
		python/sidl_Loader_Module.h \
		python/sidl_SIDLException_Module.h \
		python/sidl_BaseClass_IOR.h python/sidl_BaseClass_IOR.c \
		python/sidl_BaseException_IOR.h python/sidl_BaseException_IOR.c \
		python/sidl_BaseInterface_IOR.h python/sidl_BaseInterface_IOR.c \
		python/sidl_DLL_IOR.h python/sidl_DLL_IOR.c \
		python/sidl_SIDLException_IOR.h python/sidl_SIDLException_IOR.c \
		python/sidl_Loader_IOR.h python/sidl_Module_IOR.c \
		python/setup.py

if BABEL_RUNTIME_ONLY
babelpythonsrc=$(srcdir)
all-local-python: all-local-python2
clean-local-python: clean-local-python2
else
babelpythonsrc=.
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
	@-if ! test -d python/sidl; then mkdir -p python/sidl || exit 1; fi
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
	  $(PYTHON) sidlsetup.py --library-dirs=../sidl/.libs \
	   --include-dirs=../sidl --include-dirs=../config  $(PYTHON_SETUP_ARGS) build_ext   \
	   --inplace --rpath=$(libdir);					\
	  $(PYTHON) setup.py --library-dirs=../sidl/.libs \
	   --include-dirs=../sidl --include-dirs=../config $(PYTHON_SETUP_ARGS) build_ext 	\
	   --inplace --rpath=$(libdir);							\
	fi
	touch python/babel-stamp

all-local-python2: python/babel-stamp2
python/babel-stamp2:
	@-if ! test -d python; then mkdir -p python || exit 1; fi
	@-if ! test -d python/sidl; then mkdir -p python/sidl || exit 1; fi
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
	  $(PYTHON) sidlsetup.py --library-dirs=../sidl/.libs \
	   --include-dirs=$(srcdir)/../sidl --include-dirs=../config $(PYTHON_SETUP_ARGS)  build_ext   \
	   --inplace --rpath=$(libdir);					\
	  $(PYTHON) setup.py --library-dirs=../sidl/.libs \
	   --include-dirs=$(srcdir)/../sidl --include-dirs=../config $(PYTHON_SETUP_ARGS)  build_ext 	\
	   --inplace --rpath=$(libdir);					\
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
	  echo rm -f sidl_*IOR.* sidl_*_Module.h setup.py babel.make ;	\
	  rm -f sidl_*IOR.* sidl_*_Module.h setup.py babel.make ;	\
	  if test -d sidl; then						\
	    cd sidl;							\
	    echo rm -f config.c *.so *.dll *.a *.o *~ *.c *.h __init__.*;\
	    rm -f config.c *.so *.dll *.a *.o *~ *.c *.h __init__.*;	\
	    echo rm -f babel.make ; 					\
	    rm -f babel.make ; 						\
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
	  echo rm -f sidl_*IOR.* sidl_*_Module.h setup.py babel.make ;	\
	  rm -f sidl*IOR.* sidl_*_Module.h setup.py babel.make ;	\
	  if test -d sidl; then						\
	    cd sidl;							\
	    echo rm -f config.c *.so *.dll *.a *.o *~ babel.make ;	\
	    rm -f config.c *.so *.dll *.a *.o *~ babel.make ;		\
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
	$(PYTHON) sidlsetup.py --library-dirs=../sidl/.libs \
	 --include-dirs=$(babelpythonsrc)/../sidl --include-dirs=../config      \
	 $(PYTHON_SETUP_ARGS) build_ext --rpath=$(libdir);		\
	$(PYTHON) setup.py --library-dirs=../sidl/.libs \
	 --include-dirs=$(babelpythonsrc)/../sidl --include-dirs=../config 	\
	 $(PYTHON_SETUP_ARGS) build_ext --rpath=$(libdir);		\
	$(PYTHON) sidlsetup.py --library-dirs=../sidl/.libs \
	   --include-dirs=../sidl --include-dirs=../config install 	\
	   $(PYTHON_SETUP_ARGS) 					\
	   --prefix=$(DESTDIR)$(prefix) 				\
	   --exec-prefix=$(DESTDIR)$(exec_prefix)			\
	   --record=sidlsetup.installed;		\
	$(PYTHON) setup.py --library-dirs=../sidl/.libs \
	   --include-dirs=../sidl  --include-dirs=../config install  	\
	   $(PYTHON_SETUP_ARGS) 					\
	   --prefix=$(DESTDIR)$(prefix) 				\
	   --exec-prefix=$(DESTDIR)$(exec_prefix)			\
	   --record=setup.installed

uninstall-data-am-python:
	@$(NORMAL_UNINSTALL)
	@test -d python && test -f python/setup.installed &&		\
	 rm -f `cat python/setup.installed`
	@test -d python && test -f python/sidlsetup.installed &&	\
	 rm -f `cat python/sidlsetup.installed`
