#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 2.25 $
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

USR_FILES =\
 usr_intro.tex\
 usr_starting.tex\
 usr_struct.tex\
 usr_sstruct.tex\
 usr_fei.tex\
 usr_ij.tex\
 usr_solvers.tex\
 usr_misc.tex\
 usr_babel.tex\
 usr_manual.tex

##################################################################
# Targets
##################################################################

# all: html pdf
all: pdf

install: all
	@ \
	test ! -d ${HYPRE_INSTALL_DIR}/docs && \
	    mkdir -m 755 -p ${HYPRE_INSTALL_DIR}/docs; \
	for dir in HYPRE_usr_manual HYPRE_ref_manual bHYPRE_ref_manual; do \
	  if test -d $$dir; then \
	    test ! -d ${HYPRE_INSTALL_DIR}/docs/$$dir && \
		mkdir -m 755 -p ${HYPRE_INSTALL_DIR}/docs/$$dir; \
	    for file in $$dir/*; do \
	      cp -fp $$file ${HYPRE_INSTALL_DIR}/docs/$$file; \
	    done; \
	  fi; \
	done; \
	for file in HYPRE_usr_manual.pdf \
		HYPRE_ref_manual.pdf \
		bHYPRE_ref_manual.pdf; do \
	  test -f $$file && cp -fp $$file ${HYPRE_INSTALL_DIR}/docs/$$file; \
	done

clean:
	@/bin/rm -f *.aux *.bbl *.blg *.dvi *.log *.dlog *.toc
	@/bin/rm -f *.idx *.ilg *.ind *.brf *.out
	@/bin/rm -f vref_manual.* vbref_manual.* version.tex
	@/bin/rm -fr HYPRE_usr_manual
	@/bin/rm -f  HYPRE_usr_manual.p*
	@/bin/rm -fr HYPRE_ref_manual
	@/bin/rm -f  HYPRE_ref_manual.tex HYPRE_ref_manual.p*
	@/bin/rm -fr bHYPRE_ref_manual
	@/bin/rm -f  bHYPRE_ref_manual.tex bHYPRE_ref_manual.p*

distclean: clean

##################################################################
# Rules
##################################################################

html: ref_html bref_html

pdf: usr ref_pdf bref_pdf

usr: HYPRE_usr_manual.pdf

ref: ref_html ref_pdf
ref_html: HYPRE_ref_manual/index.html
ref_pdf: HYPRE_ref_manual.pdf

bref: bref_html bref_ps bref_pdf
bref_html: bHYPRE_ref_manual/index.html
bref_pdf: bHYPRE_ref_manual.pdf

##########

HYPRE_usr_manual.pdf: version.tex ${USR_FILES}
	${PDFLATEX} usr_manual
	bibtex usr_manual
	${PDFLATEX} usr_manual
	${PDFLATEX} usr_manual
	mv usr_manual.pdf HYPRE_usr_manual.pdf

##########

HYPRE_ref_manual/index.html: vref_manual.dxx 
	mkdir -p HYPRE_ref_manual
	doc++ -d HYPRE_ref_manual -B hypre_wiw.html vref_manual.dxx 2>&1
	cp -f hypre_wiw.gif HYPRE_ref_manual

HYPRE_ref_manual.pdf: version.tex vref_manual.tex
	${PDFLATEX} vref_manual
	${PDFLATEX} vref_manual
	mv vref_manual.pdf HYPRE_ref_manual.pdf

vref_manual.tex: vref_manual.dxx
	doc++ -t -o vref_manual.tex vref_manual.dxx
	mv vref_manual.tex vref_manual.tmp
	sed 's/HYPRE\\_Int\([^a-zA-Z_0-9]\)/int\1/g' vref_manual.tmp > vref_manual.tex

vref_manual.dxx: ref_manual.dxx
	@ \
	sed -e '/HYPRECopyright/r copyright.txt' -e 's/HYPRECopyright//g' -e 's/HYPREVersion/$(HYPRE_RELEASE_VERSION)/g' ref_manual.dxx > vref_manual.dxx

##########

bHYPRE_ref_manual/index.html: vbref_manual.dxx 
	mkdir -p bHYPRE_ref_manual
	doc++ -d bHYPRE_ref_manual -B hypre_wiw.html vbref_manual.dxx 2>&1
	cp -f hypre_wiw.gif bHYPRE_ref_manual

bHYPRE_ref_manual.pdf: version.tex vbref_manual.tex
	${PDFLATEX} vbref_manual
	${PDFLATEX} vbref_manual
	mv vbref_manual.pdf bHYPRE_ref_manual.pdf

vbref_manual.tex: vbref_manual.dxx
	doc++ -t -o vbref_manual.tex vbref_manual.dxx

vbref_manual.dxx: bref_manual.dxx
	@ \
	sed -e '/HYPRECopyright/r copyright.txt' -e 's/HYPRECopyright//g' -e 's/HYPREVersion/$(HYPRE_RELEASE_VERSION)/g' bref_manual.dxx > vbref_manual.dxx

##########

version.tex:
	@ \
	echo "\def\HYPREVersion{$(HYPRE_RELEASE_VERSION)}" > version.tex; \
	echo "\def\HYPREVersionDate{$(HYPRE_RELEASE_DATE)}" >> version.tex
