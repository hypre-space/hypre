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
# seq_ls tools are not in the distribution, but could be added
# by uncommenting herre and in configure.ac
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

#EXTRA_DIRS = $(SEQLS) $(TOOLS) test
if BLASNEEDED
BLASDIR = blas
else !BLASNEEDED
BLASDIR = 
endif !BLASNEEDED

if LAPACKNEEDED
LAPACKDIR = lapack
else !LAPACKNEEDED
LAPACKDIR = 
endif !LAPACKNEEDED

if LIBDIRNEEDED
LIBDIR = lib
else !LIBDIRNEEDED
LIBDIR = 
endif !LIBDIRNEEDED

## We need `.' in SUBDIRS because we want `check' to build `.' before
## tests.
SUBDIRS = . config\
 $(BLASDIR)\
 $(LAPACKDIR)\
 utilities\
 krylov\
 struct_mv struct_ls\
 sstruct_mv sstruct_ls\
 seq_mv parcsr_mv distributed_matrix matrix_matrix\
 IJ_mv distributed_ls parcsr_ls parcsr_es\
 FEI_mv $(SEQLS) $(TOOLS) $(BABELDIR) $(LIBDIR)

HYPRE_SUBDIR_LIST = config\
 blas lapack utilities\
 krylov\
 struct_mv struct_ls\
 sstruct_mv sstruct_ls\
 seq_mv parcsr_mv distributed_matrix matrix_matrix\
 IJ_mv distributed_ls parcsr_ls parcsr_es\
 FEI_mv babel babel-runtime test lib 

DIST_SUBDIRS = \
config\
 blas lapack utilities\
 krylov\
 struct_mv struct_ls\
 sstruct_mv sstruct_ls\
 seq_mv parcsr_mv distributed_matrix matrix_matrix\
 IJ_mv distributed_ls parcsr_ls parcsr_es\
 FEI_mv babel babel-runtime test lib docs .

test-local:
	echo "no automated test suite available yet."
	echo "To compile test drivers enter: make tests"

tests:
	@ \
	$(MAKE) $(AM_MAKEFLAGS) install; \
	(cd test && $(MAKE) $(AM_MAKEFLAGS) all)

docs: dvi ps pdf html

dvi-local:
	(cd docs && $(MAKE) $(AM_MAKEFLAGS) dvi)

ps-local:
	(cd docs && $(MAKE) $(AM_MAKEFLAGS) ps)

pdf-local: 
	(cd docs && $(MAKE) $(AM_MAKEFLAGS) pdf)

html-local: 
	(cd docs && $(MAKE) $(AM_MAKEFLAGS) html)

nofei:
	@ \
	for i in FEI_mv/fei-hypre FEI_mv/femli/lib; \
	do \
	  if [ -d $$i ]; \
	  then \
	    (cd $$i && $(MAKE) $(AM_MAKEFLAGS) clean); \
	    (cd $$i && $(MAKE) CPPFLAGS=-DNOFEI $(AM_MAKEFLAGS)); \
	    for j1 in $$i/*.lo; \
	    do \
	      test -f $$j1 && cp -f $$j1 lib; \
	    done; \
	    for j2 in $$i/.libs/*.o; \
	    do \
	      test -f $$j2 && cp -f $$j2 lib/.libs; \
	    done; \
	    for j3 in $$i/*.o; \
	    do \
	      test -f $$j3 && cp -f $$j3 lib; \
	    done; \
	  fi; \
	done; \
	(cd lib && touch libHYPRE.* .libs/libHYPRE.*); \
	(cd lib && $(MAKE) $(AM_MAKEFLAGS) libHYPRE_LSI.la)


#dist-hook:
#	@ \
#	for i in $(HYPRE_SUBDIR_LIST); \
#	do \
#	  if [ -d $$i ]; \
#	  then \
#	    cp -fr $$i $(distdir)/src; \
#	  fi; \
#	done; \
#	if [ -d docs ]; \
#	then \
#	  cp -fr docs $(distdir); \
#	fi; \
#	for i in Makefile.am Makefile.in aclocal.m4 \
#		configure configure.ac nopoe tarch; \
#	do \
#	  if [ -f $$i ]; \
#	  then \
#	    cp -f $$i $(distdir); \
#	  fi; \
#	done
#	for i in COPYRIGHT_and_DISCLAIMER CHANGELOG README; \
#	do \
#	  if [ -f $$i ]; \
#	  then \
#	    cp -f $$i $(distdir); \
#	  fi; \
#	done
