#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

SUFFIXES = .c .dxx .html .tex

EXTRA_DIST = \
 V1.5.0b.IJ_transition_info\
 autoconfig.txt\
 babel_transition_info.txt\
 block_structured.eps\
 bref_manual.dxx\
 coding_style.txt\
 concep_iface.eps\
 copyright.tex\
 dev_QA.tex\
 dev_autotest.tex\
 dev_coding.tex\
 dev_design.tex\
 dev_documentation.tex\
 dev_error_handling.tex\
 dev_fei.tex\
 dev_install.tex\
 dev_language_interop.tex\
 dev_makefile.tex\
 dev_manual.tex\
 dev_reference.tex\
 dev_repository.tex\
 dev_requirements.tex\
 dev_scm.tex\
 dev_todo\
 docxx.sty\
 fv_grid.eps\
 hypre.bib\
 hypre.perl\
 hypre.sty\
 hypre_wiw.eps\
 hypre_wiw.gif\
 hypre_wiw.html\
 indexing.txt\
 insert_row.eps\
 lobpcg_info.txt\
 loc_cmps.eps\
 loc_cmps_blk.eps\
 ref_manual.dxx\
 sample.emacs\
 sample_c_code.c\
 sample_f77_code.f\
 smg98.readme\
 usr_boomeramg.tex\
 usr_euclid.tex\
 usr_fei.tex\
 usr_ij.tex\
 usr_intro.tex\
 usr_manual.tex\
 usr_misc.tex\
 usr_parasails.tex\
 usr_pfmg.tex\
 usr_pilut.tex\
 usr_smg.tex\
 usr_solvers.tex\
 usr_sstruct.tex\
 usr_starting.tex\
 usr_struct.tex

LATEX = latex -interaction=nonstopmode
MAKEINDEX = makeindex -q
DVIPS = dvips -q -Ppdf
PSPDF = ps2pdf

ONLINE_OPTS =\
 -split 4\
 -show_section_numbers\
 -bottom_navigation\
 -local_icons

##################################################################
# Targets
##################################################################

all: html ps pdf

install: all
	@cp -fr HYPRE_usr_manual     $$HYPRE_INSTALL_DIR/docs/.
	@cp -f  HYPRE_usr_manual.ps  $$HYPRE_INSTALL_DIR/docs/.
	@cp -f  HYPRE_usr_manual.pdf $$HYPRE_INSTALL_DIR/docs/.
	@cp -fr HYPRE_ref_manual     $$HYPRE_INSTALL_DIR/docs/.
	@cp -f  HYPRE_ref_manual.ps  $$HYPRE_INSTALL_DIR/docs/.
	@cp -f  HYPRE_ref_manual.pdf $$HYPRE_INSTALL_DIR/docs/.
	@cp -fr bHYPRE_ref_manual     $$HYPRE_INSTALL_DIR/docs/.
	@cp -f  bHYPRE_ref_manual.ps  $$HYPRE_INSTALL_DIR/docs/.
	@cp -f  bHYPRE_ref_manual.pdf $$HYPRE_INSTALL_DIR/docs/.
	@cp -f  V[1-9]* $$HYPRE_INSTALL_DIR/docs/.

clean:
	@/bin/rm -f *.aux *.bbl *.blg *.dvi *.log *.dlog *.toc
	@/bin/rm -f *.idx *.ilg *.ind 
	@/bin/rm -f vref_manual.* vbref_manual.* version.tex

veryclean: clean
	@/bin/rm -fr HYPRE_usr_manual
	@/bin/rm -f  HYPRE_usr_manual.ps
	@/bin/rm -f  HYPRE_usr_manual.pdf
	@/bin/rm -fr HYPRE_dev_manual
	@/bin/rm -f  HYPRE_dev_manual.ps
	@/bin/rm -f  HYPRE_dev_manual.pdf
	@/bin/rm -fr HYPRE_ref_manual
	@/bin/rm -f  HYPRE_ref_manual.tex HYPRE_ref_manual.ps
	@/bin/rm -f  HYPRE_ref_manual.pdf
	@/bin/rm -fr bHYPRE_ref_manual
	@/bin/rm -f  bHYPRE_ref_manual.tex bHYPRE_ref_manual.ps
	@/bin/rm -f  bHYPRE_ref_manual.pdf

##################################################################
# Rules
##################################################################

html: usr_html dev_html ref_html bref_html

ps: usr_ps dev_ps ref_ps bref_ps

pdf: usr_pdf dev_pdf ref_pdf bref_pdf

usr: usr_html usr_ps usr_pdf
usr_html: HYPRE_usr_manual/usr_manual.html
usr_ps: HYPRE_usr_manual.ps
usr_pdf: HYPRE_usr_manual.pdf

dev: dev_html dev_ps def_pdf
dev_html: HYPRE_dev_manual/dev_manual.html
dev_ps: HYPRE_dev_manual.ps
dev_pdf: HYPRE_dev_manual.pdf

ref: ref_html ref_ps ref_pdf
ref_html: HYPRE_ref_manual/index.html
ref_ps: HYPRE_ref_manual.ps
ref_pdf: HYPRE_ref_manual.pdf

bref: bref_html bref_ps bref_pdf
bref_html: bHYPRE_ref_manual/index.html
bref_ps: bHYPRE_ref_manual.ps
bref_pdf: bHYPRE_ref_manual.pdf

##########

USR_FILES =\
 usr_intro.tex\
 usr_starting.tex\
 usr_struct.tex\
 usr_sstruct.tex\
 usr_fei.tex\
 usr_ij.tex\
 usr_solvers.tex\
 usr_parasails.tex\
 usr_pilut.tex\
 usr_boomeramg.tex\
 usr_misc.tex\
 usr_manual.tex

HYPRE_usr_manual/usr_manual.html: usr_manual.dvi
	mkdir -p HYPRE_usr_manual
	latex2html ${ONLINE_OPTS} -dir HYPRE_usr_manual usr_manual.tex 2>&1
	@cp -f hypre_wiw.gif HYPRE_usr_manual/.

HYPRE_usr_manual.ps: usr_manual.dvi
	${DVIPS} -o HYPRE_usr_manual.ps usr_manual

HYPRE_usr_manual.pdf: HYPRE_usr_manual.ps
	${PSPDF} HYPRE_usr_manual.ps

usr_manual.dvi: version.tex usr_manual.idx
	${LATEX} usr_manual
	${LATEX} usr_manual

usr_manual.idx: ${USR_FILES} 
	${LATEX} usr_manual
	bibtex usr_manual
	${MAKEINDEX} usr_manual.idx

##########

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
 dev_manual.tex

HYPRE_dev_manual/dev_manual.html: dev_manual.dvi
	mkdir -p HYPRE_dev_manual
	latex2html ${ONLINE_OPTS} -dir HYPRE_dev_manual dev_manual.tex 2>&1
	@cp -f hypre_wiw.gif HYPRE_dev_manual/.

HYPRE_dev_manual.ps: dev_manual.dvi
	${DVIPS} -o HYPRE_dev_manual.ps dev_manual

HYPRE_dev_manual.pdf: HYPRE_dev_manual.ps
	${PSPDF} HYPRE_dev_manual.ps

dev_manual.dvi: version.tex dev_manual.idx
	${LATEX} dev_manual
	${LATEX} dev_manual

dev_manual.idx: ${DEV_FILES}
	${LATEX} dev_manual
#	bibtex dev_manual
	${MAKEINDEX} dev_manual.idx

##########

HYPRE_ref_manual/index.html: vref_manual.dxx 
	mkdir -p HYPRE_ref_manual
	doc++ -d HYPRE_ref_manual -B hypre_wiw.html vref_manual.dxx 2>&1
	cp -f hypre_wiw.gif HYPRE_ref_manual

HYPRE_ref_manual.ps: vref_manual.dvi
	${DVIPS} -o HYPRE_ref_manual.ps vref_manual.dvi 		

HYPRE_ref_manual.pdf: HYPRE_ref_manual.ps
	${PSPDF} HYPRE_ref_manual.ps

vref_manual.dvi: version.tex vref_manual.tex
	${LATEX} vref_manual
	${LATEX} vref_manual

vref_manual.tex: vref_manual.dxx
	doc++ -t -o vref_manual.tex vref_manual.dxx

vref_manual.dxx: ref_manual.dxx
	@ \
	VNUM=`$(srcdir)/../utilities/version -number`; \
	VDATE=`$(srcdir)/../utilities/version -date`; \
	sed "s/HYPREVersion/$$VNUM/g" ref_manual.dxx > vref_manual.dxx

##########

bHYPRE_ref_manual/index.html: vbref_manual.dxx 
	mkdir -p bHYPRE_ref_manual
	doc++ -d bHYPRE_ref_manual -B hypre_wiw.html vbref_manual.dxx 2>&1
	cp -f hypre_wiw.gif bHYPRE_ref_manual

bHYPRE_ref_manual.ps: vbref_manual.dvi
	${DVIPS} -o bHYPRE_ref_manual.ps vbref_manual.dvi 		

bHYPRE_ref_manual.pdf: bHYPRE_ref_manual.ps
	${PSPDF} bHYPRE_ref_manual.ps

vbref_manual.dvi: version.tex vbref_manual.tex
	${LATEX} vbref_manual
	${LATEX} vbref_manual

vbref_manual.tex: vbref_manual.dxx
	doc++ -t -o vbref_manual.tex vbref_manual.dxx

vbref_manual.dxx: bref_manual.dxx
	@ \
	VNUM=`$(srcdir)/../utilities/version -number`; \
	VDATE=`$(srcdir)/../utilities/version -date`; \
	sed "s/HYPREVersion/$$VNUM/g" bref_manual.dxx > vbref_manual.dxx

##########

version.tex:
	@ \
	VNUM=`$(srcdir)/../utilities/version -number`; \
	VDATE=`$(srcdir)/../utilities/version -date`; \
	echo "\def\HYPREVersion{$$VNUM}" > version.tex; \
	echo "\def\HYPREVersionDate{$$VDATE}" >> version.tex

##################################################################
# Generic rules
##################################################################

