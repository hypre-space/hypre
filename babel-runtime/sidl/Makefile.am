##
## File:        Makefile.am 
## Package:     Babel makefiles
## Release:     $Name$
## Revision:    $Revision$
## Modified:    $Date$
## Description: automake makefile to build the sidl runtime library
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



INCLUDES=$(CHASM_CFLAGS) $(LIBXML2_CFLAGS)
VERSION=@VERSION@

data_DATA = sidl.sidl

if SUPPORT_FORTRAN77 
FORTRAN_SOURCES = sidlfortran.c sidlf90array.c
else
if SUPPORT_FORTRAN90
FORTRAN_SOURCES = sidlfortran.c sidlf90array.c
else
FORTRAN_EXCLUDED = sidlfortran.c sidlf90array.c
endif
endif

if SUPPORT_PYTHON
PYTHON_SOURCES = sidl_Python.c
else
PYTHON_EXCLUDED = sidl_Python.c
endif

if SUPPORT_JAVA
JAVA_SOURCES = sidl_Java.c
else
JAVA_EXCLUDED = sidl_Java.c
endif

SRCS_EXCLUDED = $(FORTRAN_EXCLUDED) $(PYTHON_EXCLUDED) $(JAVA_EXCLUDED)
SRCS_INCLUDED = $(FORTRAN_SOURCES) $(PYTHON_SOURCES) $(JAVA_SOURCES)

lib_LTLIBRARIES = libsidl.la 
include_HEADERS = $(HEADERS_INSTALL)

execincldir = $(exec_prefix)/include
nodist_execincl_HEADERS =  babel_config.h

libsidl_la_SOURCES = $(SOURCES_HAND) ltdl.c
libsidl_la_DEPENDENCIES = $(HEADERS_BABEL) $(OBJECTS_BABEL) $(OBJECTS_SPLICE)
libsidl_la_LIBADD = $(OBJECTS_BABEL) $(OBJECTS_SPLICE) $(LIBADD_DL) $(CHASM_LIBS) $(LIBXML2_LIB)
libsidl_la_LDFLAGS = -no-undefined -release $(VERSION)

AM_CPPFLAGS = $(JNI_INCLUDES)
EXTRA_DIST = $(data_DATA) $(RUNTIME_ONLY_FILES) ltdl.h ignore_me.h.in babel.make \
	libsidl.scl.in scl.dtd changeundef.pl

HEADERS_HAND = \
	sidlArray.h \
	sidl_Exception.h \
	sidlType.h \
	sidl_search_scl.h \
	sidlfortran.h \
	sidlf90array.h \
	sidl_Java.h \
	sidl_Python.h \
	sidl_String.h \
	sidl_bool_IOR.h \
	sidl_char_IOR.h \
	sidl_dcomplex_IOR.h \
	sidl_double_IOR.h \
	sidl_fcomplex_IOR.h \
	sidl_float_IOR.h \
	sidl_header.h \
	sidl_int_IOR.h \
	sidl_long_IOR.h \
	sidl_opaque_IOR.h \
	sidl_string_IOR.h \
	sidl_interface_IOR.h \
	sidl_ucxx.hh \
	sidl_cxx.hh 

HEADERS_SPLICE = \
	sidl_BaseClass_Impl.h \
	sidl_SIDLException_Impl.h \
	sidl_DLL_Impl.h \
	sidl_Loader_Impl.h \
	sidl_ClassInfoI_Impl.h

SOURCES_HAND = \
	$(FORTRAN_SOURCES) \
	$(JAVA_SOURCES) \
	$(PYTHON_SOURCES) \
	sidl_String.c \
	sidl_bool_IOR.c \
	sidl_search_scl.c \
	sidl_char_IOR.c \
	sidl_int_IOR.c \
	sidl_long_IOR.c \
	sidl_float_IOR.c \
	sidl_double_IOR.c \
	sidl_fcomplex_IOR.c \
	sidl_dcomplex_IOR.c \
	sidl_opaque_IOR.c \
	sidl_string_IOR.c \
	sidl_interface_IOR.c

SOURCES_SPLICE = \
	sidl_BaseClass_Impl.c \
	sidl_SIDLException_Impl.c \
	sidl_DLL_Impl.c \
	sidl_Loader_Impl.c \
	sidl_ClassInfoI_Impl.c

SOURCES_BABEL = \
	sidl_BaseClass_IOR.c \
	sidl_BaseClass_Skel.c \
	sidl_BaseClass_Stub.c \
	sidl_BaseException_IOR.c \
	sidl_BaseException_Stub.c \
	sidl_SIDLException_IOR.c \
	sidl_SIDLException_Skel.c \
	sidl_SIDLException_Stub.c \
	sidl_BaseInterface_IOR.c \
	sidl_BaseInterface_Stub.c \
	sidl_DLL_IOR.c \
	sidl_DLL_Skel.c \
	sidl_DLL_Stub.c \
	sidl_Loader_IOR.c \
	sidl_Loader_Skel.c \
	sidl_Loader_Stub.c \
	sidl_ClassInfo_IOR.c \
	sidl_ClassInfo_Stub.c \
	sidl_ClassInfoI_IOR.c \
	sidl_ClassInfoI_Stub.c \
	sidl_ClassInfoI_Skel.c \
	sidl_Resolve_Stub.c \
	sidl_Scope_Stub.c

HEADERS_BABEL = \
	sidl.h \
	sidl_IOR.h \
	sidl_BaseClass.h \
	sidl_BaseClass_IOR.h \
	sidl_BaseException.h \
	sidl_BaseException_IOR.h \
	sidl_SIDLException.h \
	sidl_SIDLException_IOR.h \
	sidl_BaseInterface.h \
	sidl_BaseInterface_IOR.h \
	sidl_DLL.h \
	sidl_DLL_IOR.h \
	sidl_Loader.h \
	sidl_Loader_IOR.h \
	sidl_ClassInfo_IOR.h \
	sidl_ClassInfoI_IOR.h \
	sidl_ClassInfo.h \
	sidl_ClassInfoI.h \
	sidl_Resolve_IOR.h \
	sidl_Resolve.h \
	sidl_Scope_IOR.h \
	sidl_Scope.h

HEADERS_INSTALL = \
	$(HEADERS_HAND) \
	sidl.h \
	sidl_BaseClass.h \
	sidl_BaseClass_IOR.h \
	sidl_BaseClass_Impl.h \
	sidl_BaseException.h \
	sidl_BaseException_IOR.h \
	sidl_SIDLException.h \
	sidl_SIDLException_IOR.h \
	sidl_BaseInterface.h \
	sidl_BaseInterface_IOR.h \
	sidl_DLL.h \
	sidl_DLL_IOR.h \
	sidl_Loader.h \
	sidl_Loader_IOR.h \
	sidl_ClassInfo.h \
	sidl_ClassInfo_IOR.h \
	sidl_ClassInfoI.h \
	sidl_ClassInfoI_IOR.h \
	sidl_Resolve_IOR.h \
	sidl_Resolve.h \
	sidl_Scope_IOR.h \
	sidl_Scope.h

OBJECTS_BABEL  = $(SOURCES_BABEL:.c=.lo)
OBJECTS_HAND   = $(SOURCES_HAND:.c=.lo)
OBJECTS_SPLICE = $(SOURCES_SPLICE:.c=.lo)
OBJECTS_INCLUDED =$(SOURCES_INCLUDED:.c=.lo)
FILES_COPY = $(HEADERS_SPLICE) $(SOURCES_SPLICE) $(HEADERS_HAND)

$(OBJECTS_INCLUDED) $(OBJECTS_BABEL) $(OBJECTS_HAND) $(OBJECTS_SPLICE): babel-compile
$(HEADERS_INSTALL) $(HEADERS_BABEL) $(SOURCES_BABEL): babel-stamp
RUNTIME_ONLY_FILES= $(SOURCES_HAND) $(HEADERS_HAND) \
			$(SOURCES_SPLICE) $(HEADERS_SPLICE) \
			$(SOURCES_BABEL) $(HEADERS_BABEL) $(SRCS_EXCLUDED)


if BABEL_RUNTIME_ONLY
babel-compile:
	touch babel-compile

babel-stamp:
	touch babel-stamp

clean-local:
	rm -f babel-stamp babel-compile stamp-h1 *.o *.lo
else
babel-compile: babel-stamp
	touch babel-compile
	$(MAKE) $(AM_MAKEFLAGS) $(OBJECTS_BABEL) $(OBJECTS_SPLICE)

babel-stamp: $(srcdir)/sidl.sidl 
	$(SHELL) $(top_srcdir)/bin/copy.sh $(srcdir) . $(HEADERS_HAND)
	sed -e 's,@@DIR@@,'`pwd`',g' < $(srcdir)/libsidl.scl.in > libsidl.scl
	$(SHELL) ../../bin/babel --suppress-timestamp --no-default-repository\
	  --generate-sidl-stdlib --vpath=$(srcdir) $(srcdir)/sidl.sidl 
	touch babel-stamp


clean-local:
	rm -f $(HEADERS_BABEL) $(OBJECTS_BABEL) $(SOURCES_BABEL)
	rm -f babel-stamp babel-compile stamp-h1 *.scl
	$(SHELL) $(top_srcdir)/bin/remove.sh $(srcdir) $(FILES_COPY)
	test "X$(srcdir)" = "X." || rm -f babel.make

endif

install-data-hook:
	sed -e 's,@@DIR@@,$(DESTDIR)$(libdir),g' < $(srcdir)/libsidl.scl.in \
		> libsidl_inst.scl
	$(INSTALL_DATA) libsidl_inst.scl $(DESTDIR)$(libdir)/libsidl.scl
	rm -f libsidl_inst.scl

dist-hook:
	touch $(distdir)/Makefile.am $(distdir)/Makefile.in
distclean-local:
	rm -f babel_config.h

AUTOHEADER = @AUTOHEADER@ \
	     && cd sidl \
	     && @PERL@ -p -i.bak changeundef.pl \
	          babel_config.h.in \
             && rm -f babel_config.h.in.bak

include babel.make
