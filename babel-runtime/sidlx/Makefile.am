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


INCLUDEDIR  = $(top_builddir)/sidl
INCLUDEDIR2 = .
LIBSIDL    = $(top_builddir)/sidl/libsidl.la

VERSION=@VERSION@

data_DATA = sidlx.sidl

lib_LTLIBRARIES = libsidlx.la 
libsidlx_la_SOURCES = force_cxx_linker.cc sidlx_common.c 
libsidlx_la_DEPENDENCIES = babel-stamp $(STUBHDRS) sidlx_common.h $(ALLOBJS)
libsidlx_la_LIBADD = $(ALLOBJS)
libsidlx_la_LDFLAGS = -no-undefined -release $(VERSION) -rpath `pwd`

EXTRA_DIST = $(data_DATA) $(IMPLSRCS) $(IMPLHDRS) babel.make 

IOROBJS  = $(IORSRCS:.c=.lo)
STUBOBJS = $(STUBSRCS:.cc=.lo)
SKELOBJS = $(SKELSRCS:.cc=.lo)
IMPLOBJS = $(IMPLSRCS:.cc=.lo)
ALLOBJS = $(STUBOBJS) $(IOROBJS) $(SKELOBJS) $(IMPLOBJS)


# $(STUBHDRS) $(IOROBJS) $(STUBOBJS) $(SKELOBJS) $(IMPLOBJS): babel-stamp

babel-stamp: $(srcdir)/sidlx.sidl
	$(SHELL) ../../bin/babel --vpath="$(srcdir)" --server=C \
	  --suppress-timestamp $(srcdir)/sidlx.sidl
	rm -f sidlx.scl
	echo "<?xml version=\"1.0\" ?>" > sidlx.scl
	echo "<scl>" >> sidlx.scl
	echo "  <library uri=\""`pwd`"/libsidlx.la\" scope=\"local\" resolution=\"lazy\" >" >> sidlx.scl
	grep __set_epv $(IORSRCS) /dev/null | awk 'BEGIN {FS=":"} { print $$1}' | sort -u | sed -e 's/_IOR.c//g' -e 's/_/./g' | awk ' { printf "    <class name=\"%s\" desc=\"ior/impl\" />\n", $$1 }' >> sidlx.scl
	echo "  </library>" >> sidlx.scl
	echo "</scl>" >> sidlx.scl
	touch babel-stamp
#	$(MAKE) $(AM_MAKEFLAGS) $(IOROBJS) $(STUBOBJS) $(SKELOBJS) $(IMPLOBJS)

clean-local:
	rm -f babel-stamp *.a *.so *.scl stamp-installcheck core 
	rm -f $(IORHDRS) $(IORSRCS) $(SKELSRCS) $(STUBHDRS) $(STUBSRCS) 
	test "X$(srcdir)" = "X." || rm -f $(IMPLSRCS) $(IMPLHDRS) babel.make 

include babel.make
