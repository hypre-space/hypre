##
## File:        Makefile.am 
## Package:     Babel makefiles
## Release:     $Name$
## Revision:    $Revision$
## Modified:    $Date$
## Description: automake makefile to build the SIDL runtime library
##
## Copyright (c) 2000-2001, The Regents of the University of Calfornia.
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
                   no-installman no-texinfo.tex 1.5

VERSION=@VERSION@
data_DATA = sidl.sidl

if SUPPORT_FORTRAN77
FORTRAN_SOURCES = SIDLfortran.c SIDLfortarray.c
else
FORTRAN_EXCLUDED = SIDLfortran.c SIDLfortarry.c
endif

if SUPPORT_PYTHON
PYTHON_SOURCES = SIDL_Python.c
else
PYTHON_EXCLUDED = SIDL_Python.c
endif

if SUPPORT_JAVA
JAVA_SOURCES = SIDL_Java.c
else
JAVA_EXCLUDED = SIDL_Java.c
endif

SRCS_EXCLUDED = $(FORTRAN_EXCLUDED) $(PYTHON_EXCLUDED) $(JAVA_EXCLUDED)
SRCS_INCLUDED = $(FORTRAN_SOURCES) $(PYTHON_SOURCES) $(JAVA_SOURCES)

lib_LTLIBRARIES = libsidl.la
include_HEADERS = $(HEADERS_INSTALL) babel_config.h

libsidl_la_SOURCES = $(SOURCES_HAND) ltdl.c
libsidl_la_DEPENDENCIES = $(HEADERS_BABEL) $(OBJECTS_BABEL) $(OBJECTS_SPLICE)
libsidl_la_LIBADD = $(OBJECTS_BABEL) $(OBJECTS_SPLICE) $(LIBADD_DL)
libsidl_la_LDFLAGS = -no-undefined -release $(VERSION)
libsidl_la_INCLUDES = -Iconfig

AM_CPPFLAGS = -DHAVE_LTDL
EXTRA_DIST = $(data_DATA) $(RUNTIME_ONLY_FILES) ltdl.h ignore_me.h.in babel.make

HEADERS_HAND = \
	SIDLArray.h \
	SIDL_Exception.h \
	SIDLType.h \
	SIDLfortran.h \
	SIDL_Java.h \
	SIDL_Python.h \
	SIDL_String.h \
	SIDL_bool_IOR.h \
	SIDL_char_IOR.h \
	SIDL_dcomplex_IOR.h \
	SIDL_double_IOR.h \
	SIDL_fcomplex_IOR.h \
	SIDL_float_IOR.h \
	SIDL_header.h \
	SIDL_int_IOR.h \
	SIDL_long_IOR.h \
	SIDL_opaque_IOR.h \
	SIDL_string_IOR.h \
	SIDL_interface_IOR.h \
	SIDL_cxx.hh 

HEADERS_SPLICE = \
	SIDL_BaseClass_Impl.h \
	SIDL_BaseException_Impl.h \
	SIDL_DLL_Impl.h \
	SIDL_Loader_Impl.h

SOURCES_HAND = \
	$(FORTRAN_SOURCES) \
	$(JAVA_SOURCES) \
	$(PYTHON_SOURCES) \
	SIDL_String.c \
	SIDL_bool_IOR.c \
	SIDL_char_IOR.c \
	SIDL_int_IOR.c \
	SIDL_long_IOR.c \
	SIDL_float_IOR.c \
	SIDL_double_IOR.c \
	SIDL_fcomplex_IOR.c \
	SIDL_dcomplex_IOR.c \
	SIDL_opaque_IOR.c \
	SIDL_string_IOR.c \
	SIDL_interface_IOR.c

SOURCES_SPLICE = \
	SIDL_BaseClass_Impl.c \
	SIDL_BaseException_Impl.c \
	SIDL_DLL_Impl.c \
	SIDL_Loader_Impl.c

SOURCES_BABEL = \
	SIDL_BaseClass_IOR.c \
	SIDL_BaseClass_Skel.c \
	SIDL_BaseClass_Stub.c \
	SIDL_BaseException_IOR.c \
	SIDL_BaseException_Skel.c \
	SIDL_BaseException_Stub.c \
	SIDL_BaseInterface_IOR.c \
	SIDL_BaseInterface_Stub.c \
	SIDL_DLL_IOR.c \
	SIDL_DLL_Skel.c \
	SIDL_DLL_Stub.c \
	SIDL_Loader_IOR.c \
	SIDL_Loader_Skel.c \
	SIDL_Loader_Stub.c

HEADERS_BABEL = \
	SIDL.h \
	SIDL_IOR.h \
	SIDL_BaseClass.h \
	SIDL_BaseClass_IOR.h \
	SIDL_BaseException.h \
	SIDL_BaseException_IOR.h \
	SIDL_BaseInterface.h \
	SIDL_BaseInterface_IOR.h \
	SIDL_DLL.h \
	SIDL_DLL_IOR.h \
	SIDL_Loader.h \
	SIDL_Loader_IOR.h

HEADERS_INSTALL = \
	$(HEADERS_HAND) \
	SIDL.h \
	SIDL_BaseClass.h \
	SIDL_BaseClass_IOR.h \
	SIDL_BaseException.h \
	SIDL_BaseException_IOR.h \
	SIDL_BaseInterface.h \
	SIDL_BaseInterface_IOR.h \
	SIDL_DLL.h \
	SIDL_DLL_IOR.h \
	SIDL_Loader.h \
	SIDL_Loader_IOR.h

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
	$(RM) babel-stamp babel-compile stamp-h1 *.o *.lo
else
babel-compile: babel-stamp
	touch babel-compile
	$(MAKE) $(AM_MAKEFLAGS) $(OBJECTS_BABEL) $(OBJECTS_SPLICE)

babel-stamp: $(srcdir)/sidl.sidl
	$(SHELL) $(top_srcdir)/bin/copy.sh $(srcdir) . $(FILES_COPY)
	$(SHELL) ../../bin/babel --suppress-timestamp --no-default-repository	\
	  --generate-sidl-stdlib $(srcdir)/sidl.sidl
	touch babel-stamp

clean-local:
	rm -f $(HEADERS_BABEL) $(OBJECTS_BABEL) $(SOURCES_BABEL)
	rm -f babel_config.h babel-stamp babel-compile babel.make 
	$(SHELL) $(top_srcdir)/bin/remove.sh $(srcdir) $(FILES_COPY)
endif

AUTOHEADER = @AUTOHEADER@ \
	     && cd config \
	     && @PERL@ -p -i.bak \
	          -e "s/^\#undef (.*)$$/\#ifndef \1\n\#undef \1\n\#endif/" \
	          babel_config.h.in \
             && rm -f babel_config.h.in.bak

include babel.make
