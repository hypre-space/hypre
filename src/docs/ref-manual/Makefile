#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision$
#EHEADER**********************************************************************

PDFLATEX = pdflatex --interaction=nonstopmode

##################################################################
# Targets
##################################################################

# all: html pdf
all: html pdf

install:

clean:
	@/bin/rm -fr html latex

distclean: clean

##################################################################
# Rules
##################################################################

html:
	doxygen conf.doxygen

pdf: html
	@cd latex; ${PDFLATEX} refman; ${PDFLATEX} refman

##########

version.tex:
	@ \
	echo "\def\HYPREVersion{$(HYPRE_RELEASE_VERSION)}" > version.tex; \
	echo "\def\HYPREVersionDate{$(HYPRE_RELEASE_DATE)}" >> version.tex
