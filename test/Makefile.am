#
# Makefile.am --- automake input file for Hypre test drivers
#
## Process this file with automake to create Makefile.in

# hopefully these will be auto-generated with >automake-1.8.0
abs_top_srcdir = @abs_top_srcdir@
abs_srcdir = @abs_srcdir@
abs_builddir = @abs_builddir@
EXTRA_bindir = $(bindir)
binPROGRAMS_INSTALL = $(INSTALL_PROGRAM)

SUBDIRS = \
 .\
 AUTOTEST\
 TEST_fei\
 TEST_ij\
 TEST_sstruct\
 TEST_struct\
 TEST_timing

# list of possible build objects, (no automake install rule generation) 
EXTRA_PROGRAMS = \
 ij\
 ij_es\
 ij_mv\
 new_ij\
 sstruct\
 struct\
 f77_ij\
 f77_ij_b\
 f77_ij_mv\
 f77_struct\
 fei\
 ij_b\
 ij.C++\
 new_ij.C++\
 ij_mv.C++\
 struct.C++\
 sstruct.C++
# f77_struct_b\
# struct_b

# default preprocessor flags for C, C++, and Fortran
AM_CPPFLAGS = \
 -DHYPRE_TIMING\
 -I$(includedir)

AM_CXXFLAGS = -DMPICH_SKIP_MPICXX $(AM_CPPFLAGS)

AM_FFLAGS = \
 -I$(includedir)

EXTRA_DIST = \
 runtest.sh

# ij, build 
ij_SOURCES = ij.c
ij_libraries = \
 -L$(libdir)\
 -lHYPRE_parcsr_ls\
 -lHYPRE_DistributedMatrixPilutSolver\
 -lHYPRE_ParaSails\
 -lHYPRE_Euclid\
 -lHYPRE_IJ_mv\
 -lHYPRE_MatrixMatrix\
 -lHYPRE_DistributedMatrix\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 $(LAPACKLIBFLAGS)\
 $(BLASLIBFLAGS)\
 $(LIBM)
ij_LDADD = $(ij_libraries)
#ij_LDFLAGS = -static
# ij, dependencies rules needed because automatic dependency checking is off
ij_dependencies = \
 $(includedir)/utilities.h\
 $(includedir)/HYPRE.h\
 $(includedir)/HYPRE_IJ_mv.h\
 $(includedir)/HYPRE_parcsr_mv.h\
 $(includedir)/HYPRE_parcsr_ls.h\
 $(includedir)/parcsr_mv.h\
 $(includedir)/krylov.h

# ij, build 
ij_es_SOURCES = ij_es.c
ij_es_libraries = \
 -L$(libdir)\
 -lHYPRE_parcsr_es\
 -lHYPRE_parcsr_ls\
 -lHYPRE_DistributedMatrixPilutSolver\
 -lHYPRE_ParaSails\
 -lHYPRE_Euclid\
 -lHYPRE_IJ_mv\
 -lHYPRE_MatrixMatrix\
 -lHYPRE_DistributedMatrix\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 $(LAPACKLIBFLAGS)\
 $(BLASLIBFLAGS)\
 $(LIBM)
ij_es_LDADD = \
 $(ij_libraries) \
 @FLIBS@
#ij_es_LDFLAGS = -static
# ij, dependencies rules needed because automatic dependency checking is off
ij_es_dependencies = \
 $(includedir)/utilities.h\
 $(includedir)/HYPRE.h\
 $(includedir)/HYPRE_IJ_mv.h\
 $(includedir)/HYPRE_parcsr_mv.h\
 $(includedir)/HYPRE_parcsr_ls.h\
 $(includedir)/HYPRE_lobpcg.h\
 $(includedir)/lobpcg.h\
 $(includedir)/parcsr_mv.h\
 $(includedir)/krylov.h

# ij_mv, build
ij_mv_SOURCES = ij_mv.c
ij_mv_CPPFLAGS = $(AM_CPPFLAGS)
ij_mv_libraries = $(ij_libraries)
ij_mv_LDADD = $(ij_mv_libraries)
ij_mv_dependencies = \
 $(includedir)/utilities.h\
 $(includedir)/HYPRE.h\
 $(includedir)/HYPRE_IJ_mv.h\
 $(includedir)/HYPRE_parcsr_mv.h\
 $(includedir)/HYPRE_parcsr_ls.h\
 $(includedir)/parcsr_mv.h\
 $(includedir)/krylov.h

# new_ij, build
new_ij_SOURCES = new_ij.c
new_ij_CPPFLAGS = $(AM_CPPFLAGS)
new_ij_LDADD = $(ij_libraries)

# sstruct, build
sstruct_SOURCES = sstruct.c
sstruct_CPPFLAGS = $(AM_CPPFLAGS)
sstruct_libraries = \
 -L$(libdir)\
 -lHYPRE_sstruct_ls\
 -lHYPRE_sstruct_mv\
 -lHYPRE_struct_ls\
 -lHYPRE_struct_mv\
 -lHYPRE_parcsr_ls\
 -lHYPRE_DistributedMatrixPilutSolver\
 -lHYPRE_ParaSails\
 -lHYPRE_Euclid\
 -lHYPRE_MatrixMatrix\
 -lHYPRE_DistributedMatrix\
 -lHYPRE_IJ_mv\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 $(LAPACKLIBFLAGS)\
 $(BLASLIBFLAGS)\
 $(LIBM)
sstruct_LDADD = $(sstruct_libraries)
sstruct_dependencies = \
 $(includedir)/utilities.h\
 $(includedir)/HYPRE_sstruct_ls.h\
 $(includedir)/krylov.h

# struct, build
struct_SOURCES = struct.c
struct_CPPFLAGS = $(AM_CPPFLAGS)
struct_libraries = \
 -L$(libdir)\
 -lHYPRE_struct_ls\
 -lHYPRE_struct_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 $(LIBM)
struct_LDADD = $(struct_libraries)
struct_dependencies = \
 $(includedir)/utilities.h\
 $(includedir)/HYPRE_struct_ls.h\
 $(includedir)/struct_mv.h\
 $(includedir)/krylov.h

# optiontional target, fei++
# fei
fei_SOURCES = fei.cxx
fei_CXXFLAGS = \
 -DNOFEI\
 -DHYPRE_TIMING\
 -DMPICH_SKIP_MPICXX\
 -I$(includedir)
fei_LDADD = \
 -L$(libdir)\
 -lHYPRE_LSI\
 -lHYPRE_superlu\
 $(LAPACKLIBFLAGS)\
 $(BLASLIBFLAGS)\
 $(LIBM)
#fei_LDFLAGS = -static
fei_dependencies = \
 $(includedir)/utilities.h\
 $(includedir)/HYPRE.h\
 $(includedir)/HYPRE_IJ_mv.h\
 $(includedir)/HYPRE_parcsr_mv.h\
 $(includedir)/HYPRE_parcsr_ls.h\
 $(includedir)/HYPRE_LinSysCore.h

# optiontional targets, all77
# f77_ij
f77_ij_SOURCES = f77_ij.f
f77_ij_FFLAGS = $(AM_FFLAGS)
f77_ij_LDADD = \
 $(ij_libraries)\
 @FLIBS@
#f77_ij_LDFLAGS = -static

# f77_ij_mv
f77_ij_mv_SOURCES = f77_ij_mv.f
f77_ij_mv_FFLAGS = $(AM_FFLAGS)
f77_ij_mv_LDADD = \
 $(ij_mv_libraries)\
 @FLIBS@
#f77_ij_mv_LDFLAGS = -static

# f77_struct
f77_struct_SOURCES = f77_struct.f
f77_struct_FFLAGS = $(AM_FFLAGS)
f77_struct_LDADD = \
 $(struct_libraries)\
 @FLIBS@
#f77_struct_LDFLAGS = -static

# optiontional targets, babel
# ij_b
ij_b_SOURCES = ij_b.c
ij_b_CPPFLAGS = $(AM_CPPFLAGS)
ij_b_libraries = \
 $(ij_libraries)\
 -lbHYPREClient-C\
 -lbHYPREClient-F\
 -lbHYPRE\
 -lsidl
ij_b_LDADD = $(ij_b_libraries)
#ij_b_LDFLAGS = -static
ij_b_dependencies = \
 $(ij_dependencies)\
 $(includedir)/bHYPRE.h\
 $(includedir)/bHYPRE_IJParCSRMatrix_Impl.h\
 $(includedir)/bHYPRE_IJParCSRVector_Impl.h\
 $(includedir)/bHYPRE_ParCSRDiagScale_Impl.h

# f77_ij_b
f77_ij_b_SOURCES = f77_ij_b.f
f77_ij_b_FFLAGS = $(AM_FFLAGS)
f77_ij_b_LDADD = \
 $(ij_b_libraries)\
 @FLIBS@
#f77_ij_b_LDFLAGS = -static

# struct_b
#EXTRA_struct_b_SOURCES = struct_b.c
#struct_b_CPPFLAGS = $(AM_CPPFLAGS)
#struct_b_LDADD = $(struct_libraries)\
# -lbHYPREClient-C\
# -lbHYPREClient-F\
# -lbHYPRE\
# -lsidl
##struct_b_LDFLAGS = -static
#struct_b_dependencies = \
#$(struct_dependencies)\
#$(includedir)/HYPRE_struct_ls.h\
#$(includedir)/Hypre_Box_Stub.h\
#$(includedir)/Hypre_StructStencil_Stub.h\
#$(includedir)/Hypre_StructGrid_Stub.h\
#$(includedir)/Hypre_StructMatrix_Stub.h\
#$(includedir)/Hypre_StructMatrixBuilder_Stub.h\
#$(includedir)/Hypre_StructVector_Stub.h\
#$(includedir)/Hypre_StructVectorBuilder_Stub.h\
#$(includedir)/Hypre_MPI_Com_Stub.h\
#$(includedir)/Hypre_StructJacobi_Stub.h\
#$(includedir)/Hypre_StructSMG_Stub.h\
#$(includedir)/Hypre_PCG_Stub.h

# f77_struct_b
#EXTRA_f77_struct_b_SOURCES = f77_struct_b.f
#f77_struct_b_FFLAGS = $(AM_FFLAGS)
#f77_struct_b_LDADD = \
# $(struct_libraries)\
# -lbHYPREClient-C\
# -lbHYPREClient-F\
# -lbHYPRE\
# -lsidl\
# @FLIBS@
##f77_struct_b_LDFLAGS = -static

# optiontional targets, all++
# Note, automake conversion of +, and . to _ 
# sources are temp (no distribute), add to clean-local target
# ij.C++
nodist_ij_C___SOURCES = ij.C++.cxx
ij_C___CXXFLAGS = $(AM_CXXFLAGS)
ij_C___LDADD = $(ij_libraries)

# ij_mv.C++
nodist_ij_mv_C___SOURCES = ij_mv.C++.cxx
ij_mv_C___CXXFLAGS = $(AM_CXXFLAGS)
ij_mv_C___LDADD = $(ij_mv_libraries)

# new_ij.C++
nodist_new_ij_C___SOURCES = new_ij.C++.cxx
new_ij_C___CXXFLAGS = $(default_cxxxlags)
new_ij_C___LDADD = $(ij_libraries)

# sstruct.C++
nodist_sstruct_C___SOURCES = sstruct.C++.cxx
sstruct_C___CXXFLAGS = $(AM_CXXFLAGS)
sstruct_C___LDADD = $(sstruct_libraries)

# struct.C++
nodist_struct_C___SOURCES = struct.C++.cxx
struct_C___CXXFLAGS = $(AM_CXXFLAGS)
struct_C___LDADD = $(struct_libraries)

all-local: ij$(EXEEXT) ij_es$(EXEEXT) ij_mv$(EXEEXT) new_ij$(EXEEXT) \
		struct$(EXEEXT) sstruct$(EXEEXT)

fei++: fei$(EXEEXT)

all++: ij.C++$(EXEEXT) ij_mv.C++$(EXEEXT) \
		struct.C++$(EXEEXT) sstruct.C++$(EXEEXT)

all77: f77_ij$(EXEEXT) f77_ij_mv$(EXEEXT) f77_struct$(EXEEXT)

babel: f77_ij_b$(EXEEXT) ij_b$(EXEEXT) # f77_struct_b$(EXEEXT) struct_b$(EXEEXT)

# some simple dependency checking
$(ij_OBJECTS): $(ij_dependencies)
$(ij_es_OBJECTS): $(ij_es_dependencies)
$(ij_mv_OBJECTS): $(ij_mv_dependencies)
$(new_ij_OBJECTS): $(ij_dependencies)
$(sstruct_OBJECTS): $(sstruct_dependencies)
$(struct_OBJECTS): $(struct_dependencies)
$(fei_OBJECTS): $(fei_dependencies)
$(f77_ij_OBJECTS): $(ij_dependencies)
$(f77_ij_mv_OBJECTS): $(ij_mv_dependencies)
$(f77_struct_OBJECTS): $(struct_dependencies)
$(ij_b_OBJECTS): $(ij_b_dependencies)
$(f77_ij_b_OBJECTS): $(ij_b_dependencies)
#$(struct_b_OBJECTS): $(struct_b_dependencies)
#$(f77_struct_b_OBJECTS): $(struct_b_dependencies)
$(ij_C___OBJECTS): $(ij_dependencies)
$(ij_mv_C___OBJECTS): $(ij_mv_dependencies)
$(new_ij_C___OBJECTS): $(ij_dependencies)
$(sstruct_C___OBJECTS): $(sstruct_dependencies)
$(struct_C___OBJECTS): $(struct_dependencies)

# rule to compile C files with C++
ij.C++.cxx:
	cp -f $(srcdir)/ij.c ij.C++.cxx

ij_mv.C++.cxx:
	cp -f $(srcdir)/ij_mv.c ij_mv.C++.cxx

sstruct.C++.cxx:
	cp -f $(srcdir)/sstruct.c sstruct.C++.cxx

struct.C++.cxx:
	cp -f $(srcdir)/struct.c struct.C++.cxx

clean-local:
	@ \
	for i in $(EXTRA_PROGRAMS) *.C++* driver.out.* test.* ; \
	do \
	  if [ -e $$i ] ; \
	  then \
	    rm -f $$i; \
	  fi; \
	done

# install, put programs in to ./TEST_* directories
install-exec-hook: all
	@ \
	for i in fei ij sstruct struct ; \
	do \
	if test -x $$i$(EXEEXT) ; \
	  then \
	    if test -d $(abs_builddir)/TEST_$$i ; \
	    then \
	      echo $(INSTALL_PROGRAM_ENV) $(LIBTOOL) --mode=install $(binPROGRAMS_INSTALL) $$i$(EXEEXT) $(abs_builddir)/TEST_$$i/$$i$(EXEEXT); \
	      $(INSTALL_PROGRAM_ENV) $(LIBTOOL) --mode=install $(binPROGRAMS_INSTALL) $$i$(EXEEXT) $(abs_builddir)/TEST_$$i/$$i$(EXEEXT); \
	    fi; \
	  fi; \
	done

uninstall-hook:
	@ \
	for i in fei ij sstruct struct ; \
	do \
	    if test -d $(abs_builddir)/TEST_$$i ; \
	    then \
	      if test -x $(abs_builddir)/TEST_$$i/$$i$(EXEEXT) ; \
	      then \
		rm -f $(abs_builddir)/TEST_$$i/$$i$(EXEEXT); \
	      fi; \
	    fi; \
	done
