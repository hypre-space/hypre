##
## File:        Makefile.am 
## Package:     gov.llnl.babel
## Revision:    $Revision: 1.11 $
## Modified:    $Date: 2006/08/29 22:29:15 $
## Description: automake makefile for the sidl Java runtime support
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




JAR     = @JAR@
VERSION = @VERSION@
JARFILE = sidl-$(VERSION).jar

if BABEL_RUNTIME_ONLY
jardir=$(prefix)/lib
jar_DATA=$(JARFILE)
endif

#
# Compile the sidl jar file.  Recompile only if the jar file does not
# exist or if any of the java sources are newer than the jar file.
#

all-local: jarfile
$(JARFILE): jarfile

jarfile:
	@if test ! -f $(JARFILE); then					\
	  $(MAKE) $(AM_MAKEFLAGS) force-jarfile;			\
	else								\
	  newfiles=`find $(srcdir)/gov $(srcdir)/sidl 			\
	    -name SCCS -prune 						\
	    -o -name CVS -prune						\
	    -o -name .svn -prune					\
            -o -name "*.java" -newer $(JARFILE) -print`;		\
	  if test "x$$newfiles" != "x"; then                         	\
	    $(MAKE) $(AM_MAKEFLAGS) force-jarfile;			\
	  fi								\
	fi

JAVAC_FLAGS = -g -d . -source 1.4 -target 1.4

force-jarfile:
	@javafiles=`find $(srcdir)/gov $(srcdir)/sidl			\
	  -name SCCS -prune -o 						\
	  -name .svn -prune -o 						\
	  -name CVS -prune -o -name "*.java" -print`;			\
	echo "$(JAVAC) $(JAVAC_FLAGS) $$javafiles";			\
	if $(JAVAC) $(JAVAC_FLAGS) $$javafiles; then			\
	  classfiles=`find gov sidl -name '*.class' -print`;		\
	  echo "$(JAR) cf $(JARFILE) $$classfiles";			\
	  $(JAR) cf $(JARFILE) $$classfiles;				\
	fi

clean-local:
	@if test -d gov; then						\
	  echo rm -f `find gov sidl -name '*.class' -print`;		\
	  rm -f `find gov sidl -name '*.class' -print`;		\
	fi
	rm -f *.jar

dist-hook:
	dirs=`cd $(srcdir); find gov sidl -type d \( -name SCCS -prune -o  \
		-name CVS -prune -o -name .svn -prune -o -print \)`;	\
	for d in $$dirs; do						\
	  mkdir -p $(distdir)/$$d;					\
	done
	files=`cd $(srcdir); find gov sidl -name SCCS -prune -o 	\
		-name CVS -prune -o					\
		-name .svn -prune -o					\
		-name "*.j*" -print`;					\
	for f in $$files; do						\
	  echo cp $(srcdir)/$$f $(distdir)/$$f;				\
	  cp $(srcdir)/$$f $(distdir)/$$f;				\
	done
