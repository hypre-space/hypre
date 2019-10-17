# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

include ../config/Makefile.config

.SUFFIXES:
.SUFFIXES: .c .dxx .html .tex

##################################################################
# Targets
##################################################################

all: html pdf

install: all
	@ \
	test ! -d ${HYPRE_INSTALL_DIR}/docs && \
	    mkdir -m 755 -p ${HYPRE_INSTALL_DIR}/docs; \
	for dir in HYPRE_usr_manual HYPRE_ref_manual; do \
	  if test -d $$dir; then \
	    test ! -d ${HYPRE_INSTALL_DIR}/docs/$$dir && \
		mkdir -m 755 -p ${HYPRE_INSTALL_DIR}/docs/$$dir; \
	    for file in $$dir/*; do \
	      cp -fp $$file ${HYPRE_INSTALL_DIR}/docs/$$file; \
	    done; \
	  fi; \
	done; \
	for file in HYPRE_usr_manual.pdf HYPRE_ref_manual.pdf; do \
	  test -f $$file && cp -fp $$file ${HYPRE_INSTALL_DIR}/docs/$$file; \
	done

clean:
	@cd usr-manual; make clean
	@cd ref-manual; make clean
	@/bin/rm -fr usr-manual-html ref-manual-html
	@/bin/rm -fr usr-manual.pdf  ref-manual.pdf

distclean: clean

##################################################################
# Rules
##################################################################

html: usr-html ref-html

pdf: usr-pdf ref-pdf

usr: usr-html usr-pdf
usr-html:
	@cd usr-manual; make html
	@mv usr-manual/_build/html usr-manual-html
usr-pdf:
	@cd usr-manual; make latexpdf
	@mv usr-manual/_build/latex/hypre.pdf usr-manual.pdf

ref: ref-html ref-pdf
ref-html:
	@cd ref-manual; make html
	@mv ref-manual/html ref-manual-html
ref-pdf:
	@cd ref-manual; make pdf
	@mv ref-manual/latex/refman.pdf ref-manual.pdf

##########

vref_manual.tex: vref_manual.dxx
	doc++ -t -o vref_manual.tex vref_manual.dxx
	mv vref_manual.tex vref_manual.tmp
	sed 's/HYPRE\\_Int\([^a-zA-Z_0-9]\)/int\1/g' vref_manual.tmp > vref_manual.tex

vref_manual.dxx: ref_manual.dxx
	@ \
	sed -e '/HYPRECopyright/r copyright.txt' -e 's/HYPRECopyright//g' -e 's/HYPREVersion/$(HYPRE_RELEASE_VERSION)/g' ref_manual.dxx > vref_manual.dxx

##########

version.tex:
	@ \
	echo "\def\HYPREVersion{$(HYPRE_RELEASE_VERSION)}" > version.tex; \
	echo "\def\HYPREVersionDate{$(HYPRE_RELEASE_DATE)}" >> version.tex
