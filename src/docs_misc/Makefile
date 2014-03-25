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


include ../config/Makefile.config

.SUFFIXES:
.SUFFIXES: .c .dxx .html .tex

PDFLATEX = pdflatex --interaction=nonstopmode
# LATEX = latex -interaction=nonstopmode
# MAKEINDEX = makeindex -q
# DVIPS = dvips -q
# PSPDF = ps2pdf
# 
# ONLINE_OPTS =\
#  -split 2\
#  -show_section_numbers\
#  -bottom_navigation

##################################################################
# Files
##################################################################

DEV_FILES =\
 dev_QA.tex\
 dev_autotest.tex\
 dev_coding.tex\
 dev_design.tex\
 dev_documentation.tex\
 dev_error_handling.tex\
 dev_install.tex\
 dev_language_interop.tex\
 dev_makefile.tex\
 dev_reference.tex\
 dev_repository.tex\
 dev_requirements.tex\
 dev_babel.tex\
 dev_manual.tex

##################################################################
# Targets
##################################################################

all: pdf

install: all

clean:
	@/bin/rm -f *.aux *.bbl *.blg *.dvi *.log *.dlog *.toc
	@/bin/rm -f *.idx *.ilg *.ind *.brf *.out
	@/bin/rm -f vref_manual.* vbref_manual.* version.tex
	@/bin/rm -fr HYPRE_dev_manual
	@/bin/rm -f  HYPRE_dev_manual.p*

distclean: clean

##################################################################
# Rules
##################################################################

pdf: dev

dev: HYPRE_dev_manual.pdf

##########

HYPRE_dev_manual.pdf: version.tex ${DEV_FILES}
	${PDFLATEX} dev_manual
	bibtex dev_manual
	${PDFLATEX} dev_manual
	${PDFLATEX} dev_manual
	mv dev_manual.pdf HYPRE_dev_manual.pdf

##########

version.tex:
	@ \
	echo "\def\HYPREVersion{$(HYPRE_RELEASE_VERSION)}" > version.tex; \
	echo "\def\HYPREVersionDate{$(HYPRE_RELEASE_DATETIME)}" >> version.tex
