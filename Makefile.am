#
# Makefile.am --- automake input file for Hypre
#
## Process this file with automake to create Makefile.in

ACLOCAL_AMFLAGS = -I config

include_HEADERS = HYPRE.h HYPRE_config.h

EXTRA_DIST = CHANGELOG COPYRIGHT_and_DISCLAIMER README nopoe

MAINTAINERCLEANFILES = Makefile.in\
 aclocal.m4\
 configure

# this is really ulgy, but there are different directory structures
# babel seq_ls tools are not in the distribution
#if PARCHORDLSDIR
#PARCHORDLS = "parchord_ls"
#else !PARCHORDLSDIR
PARCHORDLS =
#endif !PARCHORDLSDIR
#if SEQLSDIR
#SEQLS = "seq_ls"
#else
SEQLS =
#endif
#if TOOLSDIR
#TOOLS = tools
#else !TOOLSDIR
TOOLS =
#endif !TOOLSDIR

#EXTRA_DIRS = $(BABELDIR) $(SEQLS) $(TOOLS) test
## We need `.' in SUBDIRS because we want `check' to build `.' before
## tests.
SUBDIRS = . config\
 blas utilities\
 krylov\
 struct_mv struct_ls\
 sstruct_mv sstruct_ls\
 seq_mv parcsr_mv distributed_matrix matrix_matrix\
 IJ_mv distributed_ls parcsr_ls parcsr_es\
 FEI_mv $(SEQLS) $(TOOLS) $(BABELDIR) lib

DIST_SUBDIRS = . config\
 blas utilities\
 krylov\
 struct_mv struct_ls\
 sstruct_mv sstruct_ls\
 seq_mv parcsr_mv distributed_matrix matrix_matrix\
 IJ_mv distributed_ls parcsr_ls parcsr_es\
 FEI_mv babel babel-runtime test lib docs

docs: docs/HYPRE_usr_manual.ps
	@ (cd docs && $(MAKE) $(AM_MAKEFLAGS) $@)

nofei:
	@ \
	for i in FEI_mv/fei-hypre FEI_mv/femli/lib; \
	do \
	  if [ -d $$i ]; \
	  then \
	    (cd $$i && $(MAKE) $(AM_MAKEFLAGS) clean); \
	    (cd $$i && $(MAKE) CPPFLAGS=-DNOFEI $(AM_MAKEFLAGS)); \
	    cp $$i/*.lo lib; \
	    cp $$i/.libs/*.o lib/.libs; \
	    cp $$i/*.o lib; \
	  fi; \
	done; \
	(cd lib && touch libHYPRE.* .libs/libHYPRE.*); \
	(cd lib && $(MAKE) $(AM_MAKEFLAGS) libHYPRE_LSI.la)


#dist-hook:
#	@ \
#	for i in $(PETSC) $(SEQLS) $(TOOLS); \
#	do \
#	  if [ -d $$i ]; \
#	  then \
#	    rm -Rf $$i; \
#	  fi; \
#	done
