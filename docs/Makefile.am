## Process this file with automake to create Makefile.in

# hopefully these will be auto-generated with >automake-1.8.0
abs_top_srcdir = @abs_top_srcdir@
abs_srcdir = @abs_srcdir@

EXTRA_DIST = \
 $(srcdir)/V1.5.0b.IJ_transition_info\
 $(srcdir)/autoconfig.txt\
 $(srcdir)/babel_transition_info.txt\
 $(srcdir)/block_structured.eps\
 $(srcdir)/bref_manual.dxx.in\
 $(srcdir)/coding_style.txt\
 $(srcdir)/concep_iface.eps\
 $(srcdir)/copyright.tex\
 $(srcdir)/dev_QA.tex\
 $(srcdir)/dev_autotest.tex\
 $(srcdir)/dev_coding.tex\
 $(srcdir)/dev_design.tex\
 $(srcdir)/dev_documentation.tex\
 $(srcdir)/dev_error_handling.tex\
 $(srcdir)/dev_fei.tex\
 $(srcdir)/dev_install.tex\
 $(srcdir)/dev_language_interop.tex\
 $(srcdir)/dev_makefile.tex\
 $(srcdir)/dev_manual.tex\
 $(srcdir)/dev_reference.tex\
 $(srcdir)/dev_repository.tex\
 $(srcdir)/dev_requirements.tex\
 $(srcdir)/dev_scm.tex\
 $(srcdir)/dev_todo\
 $(srcdir)/docxx.sty\
 $(srcdir)/fv_grid.eps\
 $(srcdir)/hypre.bib\
 $(srcdir)/hypre.perl\
 $(srcdir)/hypre.sty\
 $(srcdir)/hypre_wiw.eps\
 $(srcdir)/hypre_wiw.gif\
 $(srcdir)/hypre_wiw.html\
 $(srcdir)/indexing.txt\
 $(srcdir)/insert_row.eps\
 $(srcdir)/lobpcg_info.txt\
 $(srcdir)/loc_cmps.eps\
 $(srcdir)/loc_cmps_blk.eps\
 $(srcdir)/ref_manual.dxx.in\
 $(srcdir)/sample.emacs\
 $(srcdir)/sample_c_code.c\
 $(srcdir)/sample_f77_code.f\
 $(srcdir)/smg98.readme\
 $(srcdir)/usr_boomeramg.tex\
 $(srcdir)/usr_euclid.tex\
 $(srcdir)/usr_fei.tex\
 $(srcdir)/usr_ij.tex\
 $(srcdir)/usr_intro.tex\
 $(srcdir)/usr_manual.tex\
 $(srcdir)/usr_misc.tex\
 $(srcdir)/usr_parasails.tex\
 $(srcdir)/usr_pfmg.tex\
 $(srcdir)/usr_pilut.tex\
 $(srcdir)/usr_smg.tex\
 $(srcdir)/usr_solvers.tex\
 $(srcdir)/usr_sstruct.tex\
 $(srcdir)/usr_starting.tex\
 $(srcdir)/usr_struct.tex

#DISTCLEANFILES = \
#docxx.sty\
#V1.5.0b.IJ_transition_info

LATEX = latex -interaction=nonstopmode
MAKEINDEX = makeindex -q
DVIPS = dvips -q -Ppdf
PSPDF = ps2pdf

ONLINE_OPTS =\
 -split 4\
 -show_section_numbers\
 -bottom_navigation\
 -local_icons

USR_FILES = \
 usr_intro.tex\
 usr_starting.tex\
 usr_struct.tex\
 usr_sstruct.tex\
 usr_fei.tex\
 usr_ij.tex\
 usr_solvers.tex\
 usr_parasails.tex\
 usr_pfmg.tex\
 usr_pilut.tex\
 usr_boomeramg.tex\
 usr_smg.tex\
 usr_euclid.tex\
 usr_misc.tex\
 usr_manual.tex

DEV_FILES = \
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
 dev_reference.tex\
 dev_repository.tex\
 dev_requirements.tex\
 dev_scm.tex\
 dev_manual.tex

##################################################################
# Targets
##################################################################

install-data-local: all
	@ \
	$(MAKE) $(AM_MAKEFLAGS); \
	test ! -d ${prefix}/docs && mkdir -m 755 ${prefix}/docs; \
	for file in HYPRE_usr_manual.ps HYPRE_usr_manual.pdf \
		HYPRE_ref_manual.ps HYPRE_ref_manual.pdf \
		bHYPRE_ref_manual.ps bHYPRE_ref_manual.pdf; do \
	  test -f $$file && $(INSTALL_DATA) $$file ${prefix}/docs/$$file; \
	done; \
	test -f $(srcdir)/V1.5.0b.IJ_transition_info && \
		test -f ${prefix}/docs/V1.5.0b.IJ_transition_info || \
		$(INSTALL_DATA) $(srcdir)/V1.5.0b.IJ_transition_info \
		${prefix}/docs/V1.5.0b.IJ_transition_info; \
	for dir in HYPRE_usr_manual HYPRE_ref_manual bHYPRE_ref_manual; do \
	  if test -d $$dir; then \
	    test ! -d ${prefix}/docs/$$dir && \
		mkdir -m 755 -p ${prefix}/docs/$$dir; \
	    for file in $$dir/*; do \
	      $(INSTALL_DATA) $$file ${prefix}/docs/$$file; \
	    done; \
	  fi; \
	done

clean-local:
	@/bin/rm -f *.aux *.bbl *.blg *.dvi *.log *.dlog *.toc
	@/bin/rm -f *.idx *.ilg *.ind 
	@/bin/rm -f *.h
	@/bin/rm -f vref_manual.* vbref_manual.* version.tex
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

BUILT_SOURCES = dvi ps pdf html

dvi-local: usr_manual.dvi dev_manual.dvi vref_manual.dvi vbref_manual.dvi

html-local: usr_html dev_html ref_html bref_html

ps-local: usr_ps dev_ps ref_ps bref_ps

pdf-local: usr_pdf dev_pdf ref_pdf bref_pdf

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

HYPRE_usr_manual/usr_manual.html: usr_manual.dvi
	@ \
	clean_usr_files=no; \
	for file in hypre.sty copyright.tex hypre_wiw.eps hypre.bib \
	  concep_iface.eps fv_grid.eps block_structured.eps \
	  hypre.perl ${USR_FILES}; do \
	  test -f $$file || test -f $(srcdir)/$$file && \
	  $(INSTALL_DATA) $(srcdir)/$$file . && clean_usr_files=yes; \
	done; \
	test ! -d HYPRE_usr_manual && mkdir -m 755 -p HYPRE_usr_manual; \
	latex2html ${ONLINE_OPTS} -dir HYPRE_usr_manual usr_manual.tex 2>&1; \
	$(INSTALL_DATA) $(srcdir)/hypre_wiw.gif HYPRE_usr_manual/.; \
	if test "$$clean_usr_files" = "yes"; then \
	  for file in hypre.sty copyright.tex hypre_wiw.eps hypre.bib \
	    concep_iface.eps fv_grid.eps block_structured.eps \
	    hypre.perl ${USR_FILES}; do \
	    test -f $$file && rm -f $$file; \
	  done; \
	fi

HYPRE_usr_manual.ps: usr_manual.dvi
	@ \
	clean_usr_files=no; \
	for file in hypre_wiw.eps concep_iface.eps \
		fv_grid.eps block_structured.eps; do \
	  test -f $$file || test -f $(srcdir)/$$file && \
		$(INSTALL_DATA) $(srcdir)/$$file . && clean_usr_files=yes; \
	done; \
	${DVIPS} -o HYPRE_usr_manual.ps usr_manual; \
	if test "$$clean_usr_files" = "yes"; then \
	  for file in hypre_wiw.eps concep_iface.eps \
		fv_grid.eps block_structured.eps; do \
	    test -f $$file && rm -f $$file; \
	  done; \
	fi

HYPRE_usr_manual.pdf: HYPRE_usr_manual.ps
	${PSPDF} HYPRE_usr_manual.ps

usr_manual.dvi: version.tex usr_manual.idx
	@ \
	clean_usr_files=no; \
	for file in hypre.sty copyright.tex hypre_wiw.eps hypre.bib \
	        concep_iface.eps fv_grid.eps block_structured.eps \
	        ${USR_FILES}; do \
	  test -f $$file || test -f $(srcdir)/$$file && \
	        $(INSTALL_DATA) $(srcdir)/$$file . && clean_usr_files=yes; \
	done; \
	${LATEX} usr_manual; \
	${LATEX} usr_manual; \
	if test "$$clean_usr_files" = "yes"; then \
	  for file in hypre.sty copyright.tex hypre_wiw.eps hypre.bib \
	        concep_iface.eps fv_grid.eps block_structured.eps \
	        ${USR_FILES}; do \
	    test -f $$file && rm -f $$file; \
	  done; \
	fi

usr_manual.idx: ${USR_FILES} 
	@ \
	clean_usr_files=no; \
	for file in hypre.sty copyright.tex hypre_wiw.eps hypre.bib \
	        concep_iface.eps fv_grid.eps block_structured.eps \
	        ${USR_FILES}; do \
	  test -f $$file || test -f $(srcdir)/$$file && \
	        $(INSTALL_DATA) $(srcdir)/$$file . && clean_usr_files=yes; \
	done; \
	${LATEX} usr_manual; \
	bibtex usr_manual; \
	${MAKEINDEX} usr_manual.idx; \
	if test "$$clean_usr_files" = "yes"; then \
	  for file in hypre.sty copyright.tex hypre_wiw.eps hypre.bib \
	        concep_iface.eps fv_grid.eps block_structured.eps \
	        ${USR_FILES}; do \
	    test -f $$file && rm -f $$file; \
	  done; \
	fi

HYPRE_dev_manual/dev_manual.html: dev_manual.dvi
	@ \
	clean_dev_files=no; \
	for file in hypre.sty copyright.tex hypre_wiw.eps hypre.perl \
		${DEV_FILES}; do \
	  test -f $$file ||  test -f $(srcdir)/$$file && \
	        $(INSTALL_DATA) $(srcdir)/$$file . && clean_dev_files=yes; \
	done; \
	test ! -d HYPRE_dev_manual && mkdir -m 755 -p HYPRE_dev_manual; \
	latex2html ${ONLINE_OPTS} -dir HYPRE_dev_manual dev_manual.tex 2>&1; \
	$(INSTALL_DATA) $(srcdir)/hypre_wiw.gif HYPRE_dev_manual/.; \
	if test "$$clean_dev_files" = "yes"; then \
	  for file in hypre.sty copyright.tex hypre_wiw.eps hypre.perl \
		${DEV_FILES}; do \
	    test -f $$file && rm -f $$file; \
	  done; \
	fi

HYPRE_dev_manual.ps: dev_manual.dvi
	@ \
	clean_dev_files=no; \
	for file in hypre_wiw.eps concep_iface.eps; do \
	  test -f $$file || test -f $(srcdir)/$$file && \
	        $(INSTALL_DATA) $(srcdir)/$$file . && clean_dev_files=yes; \
	done; \
	${DVIPS} -o HYPRE_dev_manual.ps dev_manual; \
	if test "$$clean_dev_files" = "yes"; then \
	  for file in hypre_wiw.eps concep_iface.eps; do \
	    test -f $$file && rm -f $$file; \
	  done; \
	fi

HYPRE_dev_manual.pdf: HYPRE_dev_manual.ps
	${PSPDF} HYPRE_dev_manual.ps

dev_manual.dvi: version.tex dev_manual.idx
	@ \
	clean_dev_files=no; \
	for file in hypre.sty copyright.tex hypre_wiw.eps ${DEV_FILES}; do \
	  test -f $$file ||  test -f $(srcdir)/$$file && \
	        $(INSTALL_DATA) $(srcdir)/$$file . && clean_dev_files=yes; \
	done; \
	${LATEX} dev_manual; \
	${LATEX} dev_manual; \
	if test "$$clean_dev_files" = "yes"; then \
	  for file in hypre.sty copyright.tex hypre_wiw.eps ${DEV_FILES}; do \
	    test -f $$file && rm -f $$file; \
	  done; \
	fi

dev_manual.idx: ${DEV_FILES}
	@ \
	clean_dev_files=no; \
	for file in hypre.sty copyright.tex hypre_wiw.eps $(DEV_FILES); do \
	  test -f $$file ||  test -f $(srcdir)/$$file && \
	        $(INSTALL_DATA) $(srcdir)/$$file . && clean_dev_files=yes; \
	done; \
	${LATEX} dev_manual; \
	${MAKEINDEX} dev_manual.idx; \
	if test "$$clean_dev_files" = "yes"; then \
	  for file in hypre.sty copyright.tex hypre_wiw.eps $(DEV_FILES); do \
	    test -f $$file && rm -f $$file; \
	  done; \
	fi

HYPRE_ref_manual/index.html: vref_manual.dxx 
	@ \
	test ! -d HYPRE_ref_manual && mkdir -m 755 -p HYPRE_ref_manual; \
	doc++ -d HYPRE_ref_manual -B $(abs_srcdir)/hypre_wiw.html vref_manual.dxx 2>&1; \
	$(INSTALL_DATA) $(abs_srcdir)/hypre_wiw.gif HYPRE_ref_manual

HYPRE_ref_manual.ps: vref_manual.dvi
	${DVIPS} -o HYPRE_ref_manual.ps vref_manual.dvi 		

HYPRE_ref_manual.pdf: HYPRE_ref_manual.ps
	${PSPDF} HYPRE_ref_manual.ps

vref_manual.dvi: version.tex vref_manual.tex
	@ \
	clean_inc_files=no; \
	test -f docxx.sty || test -f $(srcdir)/docxx.sty && \
	        $(INSTALL_DATA) $(srcdir)/docxx.sty . && clean_inc_files=yes; \
	${LATEX} vref_manual; \
	${LATEX} vref_manual; \
	if test "$$clean_inc_files" = "yes"; then \
	  test -f docxx.sty || rm -f docxx.sty; \
	fi

vref_manual.tex: vref_manual.dxx
	@ \
	clean_inc_files=no; \
	test -f docxx.sty || test -f $(srcdir)/docxx.sty && \
	        $(INSTALL_DATA) $(srcdir)/docxx.sty . && clean_inc_files=yes; \
	doc++ -t -o vref_manual.tex vref_manual.dxx; \
	if test "$$clean_inc_files" = "yes"; then \
	  test -f docxx.sty || rm -f docxx.sty; \
	fi

vref_manual.dxx: ref_manual.dxx
	@ \
	sed "s/HYPREVersion/$$PACKAGE_VERSION/g" ref_manual.dxx > vref_manual.dxx

##########

bHYPRE_ref_manual/index.html: vbref_manual.dxx 
	@ \
	test ! -d bHYPRE_ref_manual && mkdir -m 755 -p bHYPRE_ref_manual; \
	doc++ -d bHYPRE_ref_manual -B $(abs_srcdir)/hypre_wiw.html vbref_manual.dxx 2>&1; \
	$(INSTALL_DATA) $(abs_srcdir)/hypre_wiw.gif bHYPRE_ref_manual

bHYPRE_ref_manual.ps: vbref_manual.dvi
	${DVIPS} -o bHYPRE_ref_manual.ps vbref_manual.dvi 		

bHYPRE_ref_manual.pdf: bHYPRE_ref_manual.ps
	${PSPDF} bHYPRE_ref_manual.ps

vbref_manual.dvi: version.tex vbref_manual.tex
	${LATEX} vbref_manual
	${LATEX} vbref_manual

vbref_manual.tex: vbref_manual.dxx
	@ \
	doc++ -t -o vbref_manual.tex vbref_manual.dxx

vbref_manual.dxx: bref_manual.dxx
	@ \
	sed "s/HYPREVersion/$$PACKAGE_VERSION/g" bref_manual.dxx > vbref_manual.dxx

##########

version.tex:
	@ \
	echo "\def\HYPREVersion{$$PACKAGE_VERSION}" > version.tex; \
	echo "\def\HYPREVersionDate{$$PACKAGE_DATETIME}" >> version.tex

##################################################################
# Generic rules
##################################################################

#dist-hook:
#	@ \
#	$(MAKE) $(AM_MAKEFLAGS); \
#	test ! -d ${distdir} && mkdir -m 755 -p ${distdir}; \
#	for file in HYPRE_usr_manual.ps HYPRE_usr_manual.pdf \
#		HYPRE_ref_manual.ps HYPRE_ref_manual.pdf \
#		bHYPRE_ref_manual.ps bHYPRE_ref_manual.pdf \
#		$(srcdir)/V[1-9]*; do \
#	  test -f $$file && $(INSTALL_DATA) $$file ${distdir}/$$file; \
#	done; \
#	test -f $(srcdir)/V1.5.0b.IJ_transition_info && \
#		$(INSTALL_DATA) $(srcdir)/V1.5.0b.IJ_transition_info \
#		${distdir}/V1.5.0b.IJ_transition_info; \
#	for dir in HYPRE_usr_manual HYPRE_ref_manual bHYPRE_ref_manual; do \
#	  if test -d $$dir; then \
#	    test ! -d ${distdir}/$$dir && mkdir -m 755 -p ${distdir}/$$dir; \
#	    for file in $$dir/*; do \
#	      $(INSTALL_DATA) $$file ${distdir}/$$file; \
#	    done; \
#	  fi; \
#	done
	
