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

AM_CXXFLAGS = -DMPICH_SKIP_MPICXX

AM_FFLAGS = \
 -I$(includedir)

EXTRA_DIST = \
 funcs.sh\
 runtest.sh

# ij, build 
ij_SOURCES = ij.c
ij_LDADD = \
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
# automake will reorder
$(ij_OBJECTS): $(ij_dependencies)

# ij_mv, build
ij_mv_SOURCES = ij_mv.c
ij_mv_LDADD = \
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
ij_mv_dependencies = \
 $(includedir)/utilities.h\
 $(includedir)/HYPRE.h\
 $(includedir)/HYPRE_IJ_mv.h\
 $(includedir)/HYPRE_parcsr_mv.h\
 $(includedir)/HYPRE_parcsr_ls.h\
 $(includedir)/krylov.h
$(ij_mv_OBJECTS): $(ij_mv_dependencies)

# new_ij, build
new_ij_SOURCES = new_ij.c
new_ij_LDADD = \
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
new_ij_dependencies = \
 $(includedir)/utilities.h\
 $(includedir)/HYPRE.h\
 $(includedir)/HYPRE_IJ_mv.h\
 $(includedir)/HYPRE_parcsr_mv.h\
 $(includedir)/HYPRE_parcsr_ls.h\
 $(includedir)/parcsr_mv.h\
 $(includedir)/krylov.h
$(new_ij_OBJECTS): $(new_ij_dependencies)

# sstruct, build
sstruct_SOURCES = sstruct.c
sstruct_LDADD = \
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
sstruct_dependencies = \
 $(includedir)/utilities.h\
 $(includedir)/HYPRE_sstruct_ls.h\
 $(includedir)/krylov.h
$(sstruct_OBJECTS): $(sstruct_dependencies)

# struct, build
struct_SOURCES = struct.c
struct_LDADD = \
 -L$(libdir)\
 -lHYPRE_struct_ls\
 -lHYPRE_struct_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 $(LIBM)
struct_dependencies = \
 $(includedir)/utilities.h\
 $(includedir)/HYPRE_struct_ls.h\
 $(includedir)/struct_mv.h\
 $(includedir)/krylov.h
$(struct_OBJECTS): $(struct_dependencies)

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
$(fei_OBJECTS): $(fei_dependencies)

# optiontional targets, all77
# f77_ij
f77_ij_SOURCES = f77_ij.f
f77_ij_LDADD = \
 -L$(libdir)\
 -lHYPRE_parcsr_ls\
 -lHYPRE_DistributedMatrixPilutSolver \
 -lHYPRE_ParaSails \
 -lHYPRE_Euclid \
 -lHYPRE_MatrixMatrix \
 -lHYPRE_DistributedMatrix \
 -lHYPRE_IJ_mv\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 $(LAPACKLIBFLAGS)\
 $(BLASLIBFLAGS)\
 @FLIBS@\
 $(LIBM)
#f77_ij_LDFLAGS = -static
f77_ij_dependencies = \
 $(includedir)/utilities.h\
 $(includedir)/HYPRE.h\
 $(includedir)/HYPRE_IJ_mv.h\
 $(includedir)/HYPRE_parcsr_mv.h\
 $(includedir)/HYPRE_parcsr_ls.h\
 $(includedir)/parcsr_mv.h\
 $(includedir)/krylov.h
$(f77_ij_OBJECTS): $(f77_ij_dependencies)

# f77_ij_mv
f77_ij_mv_SOURCES = f77_ij_mv.f
f77_ij_mv_LDADD = \
 -L$(libdir)\
 -lHYPRE_parcsr_ls\
 -lHYPRE_DistributedMatrixPilutSolver \
 -lHYPRE_ParaSails \
 -lHYPRE_Euclid \
 -lHYPRE_MatrixMatrix \
 -lHYPRE_DistributedMatrix \
 -lHYPRE_IJ_mv\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 $(LAPACKLIBFLAGS)\
 $(BLASLIBFLAGS)\
 @FLIBS@\
 $(LIBM)
#f77_ij_mv_LDFLAGS = -static
f77_ij_mv_dependencies = \
 $(includedir)/utilities.h\
 $(includedir)/HYPRE.h\
 $(includedir)/HYPRE_IJ_mv.h\
 $(includedir)/HYPRE_parcsr_mv.h\
 $(includedir)/HYPRE_parcsr_ls.h\
 $(includedir)/krylov.h
$(f77_ij_mv_OBJECTS): $(f77_ij_mv_dependencies)

# f77_struct
f77_struct_SOURCES = f77_struct.f
f77_struct_LDADD = \
 -L$(libdir)\
 -lHYPRE_struct_ls\
 -lHYPRE_struct_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 @FLIBS@\
 $(LIBM)
#f77_struct_LDFLAGS = -static
f77_struct_dependencies = \
 $(includedir)/utilities.h\
 $(includedir)/HYPRE_struct_ls.h\
 $(includedir)/struct_mv.h\
 $(includedir)/krylov.h
$(f77_struct_OBJECTS): $(f77_struct_dependencies)

# optiontional targets, babel
# f77_ij_b
f77_ij_b_SOURCES = f77_ij_b.f
f77_ij_b_LDADD = \
 @BABELLIBFLAGS@\
 -L$(libdir)\
 -lHYPRE_parcsr_ls\
 -lHYPRE_DistributedMatrixPilutSolver \
 -lHYPRE_ParaSails \
 -lHYPRE_Euclid \
 -lHYPRE_MatrixMatrix \
 -lHYPRE_DistributedMatrix \
 -lHYPRE_IJ_mv\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 $(LAPACKLIBFLAGS)\
 $(BLASLIBFLAGS)\
 @FLIBS@\
 $(LIBM)
#f77_ij_b_LDFLAGS = -static
# ij_b
ij_b_SOURCES = ij_b.c
ij_b_LDADD = \
 @BABELLIBFLAGS@\
 -L$(libdir)\
 -lHYPRE_parcsr_ls\
 -lHYPRE_DistributedMatrixPilutSolver \
 -lHYPRE_ParaSails \
 -lHYPRE_Euclid \
 -lHYPRE_MatrixMatrix \
 -lHYPRE_DistributedMatrix \
 -lHYPRE_IJ_mv\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 $(LAPACKLIBFLAGS)\
 $(BLASLIBFLAGS)\
 $(LIBM)
#ij_b_LDFLAGS = -static
ij_b_dependencies = \
 $(includedir)/utilities.h\
 $(includedir)/HYPRE.h\
 $(includedir)/HYPRE_IJ_mv.h\
 $(includedir)/HYPRE_parcsr_mv.h\
 $(includedir)/HYPRE_parcsr_ls.h\
 $(includedir)/parcsr_mv.h\
 $(includedir)/krylov.h\
 $(includedir)/bHYPRE.h\
 $(includedir)/bHYPRE_IJParCSRMatrix_Impl.h\
 $(includedir)/bHYPRE_IJParCSRVector_Impl.h\
 $(includedir)/bHYPRE_ParCSRDiagScale_Impl.h
$(ij_b_OBJECTS): $(ij_b_dependencies)
#EXTRA_f77_struct_b_SOURCES = f77_struct_b.f
#f77_struct_b_LDADD = \
# -L$(libdir)\
# -lHYPRE_struct_ls\
# -lHYPRE_struct_mv\
# -lHYPRE_krylov\
# -lHYPRE_utilities\
# @FLIBS@
##f77_struct_b_LDFLAGS = -static
#f77_struct_b_dependencies = \
#$(includedir)/utilities.h\
#$(includedir)/HYPRE_struct_ls.h\
#$(includedir)/struct_mv.h\
#$(includedir)/krylov.h
#$(f77_struct_b_OBJECTS): $(f77_struct_b_dependencies)
#EXTRA_struct_b_SOURCES = struct_b.c
#struct_b_LDADD = \
# -L$(libdir)\
# -lHYPRE_struct_ls\
# -lHYPRE_struct_mv\
# -lHYPRE_krylov\
# -lHYPRE_utilities\
# $(LIBM)
##struct_b_LDFLAGS = -static
#struct_b_dependencies = \
#$(includedir)/utilities.h\
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
#$(struct_b_OBJECTS): $(struct_b_dependencies)

# optiontional targets, all++
# Note, automake conversion of + to _, and sources are temp (no distribute)
# ij.C++
nodist_ij_C___SOURCES = ij.C++.cxx
ij_C___LDADD = \
 -L$(libdir)\
 -lHYPRE_parcsr_ls\
 -lHYPRE_DistributedMatrixPilutSolver \
 -lHYPRE_ParaSails \
 -lHYPRE_Euclid \
 -lHYPRE_IJ_mv\
 -lHYPRE_MatrixMatrix \
 -lHYPRE_DistributedMatrix \
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 $(LAPACKLIBFLAGS)\
 $(BLASLIBFLAGS)\
 $(LIBM)
ij_C___dependencies = \
 $(includedir)/utilities.h\
 $(includedir)/HYPRE.h\
 $(includedir)/HYPRE_IJ_mv.h\
 $(includedir)/HYPRE_parcsr_mv.h\
 $(includedir)/HYPRE_parcsr_ls.h\
 $(includedir)/parcsr_mv.h\
 $(includedir)/krylov.h
$(ij_C___OBJECTS): $(ij_C___dependencies)

# ij_mv.C++
nodist_ij_mv_C___SOURCES = ij_mv.C++.cxx
ij_mv_C___LDADD = \
 -L$(libdir)\
 -lHYPRE_parcsr_ls\
 -lHYPRE_DistributedMatrixPilutSolver \
 -lHYPRE_ParaSails \
 -lHYPRE_Euclid \
 -lHYPRE_IJ_mv\
 -lHYPRE_MatrixMatrix \
 -lHYPRE_DistributedMatrix \
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 $(LAPACKLIBFLAGS)\
 $(BLASLIBFLAGS)\
 $(LIBM)
ij_mv_C___dependencies = \
 $(includedir)/utilities.h\
 $(includedir)/HYPRE.h\
 $(includedir)/HYPRE_IJ_mv.h\
 $(includedir)/HYPRE_parcsr_mv.h\
 $(includedir)/HYPRE_parcsr_ls.h\
 $(includedir)/krylov.h
$(ij_mv_C___OBJECTS): $(ij_mv_C___dependencies)

# new_ij.C++
nodist_new_ij_C___SOURCES = new_ij.C++.cxx
new_ij_C___LDADD = \
 -L$(libdir)\
 -lHYPRE_parcsr_ls\
 -lHYPRE_DistributedMatrixPilutSolver \
 -lHYPRE_ParaSails \
 -lHYPRE_Euclid \
 -lHYPRE_IJ_mv\
 -lHYPRE_MatrixMatrix \
 -lHYPRE_DistributedMatrix \
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 $(LAPACKLIBFLAGS)\
 $(BLASLIBFLAGS)\
 $(LIBM)
new_ij_C___dependencies = \
 $(includedir)/utilities.h\
 $(includedir)/HYPRE.h\
 $(includedir)/HYPRE_IJ_mv.h\
 $(includedir)/HYPRE_parcsr_mv.h\
 $(includedir)/HYPRE_parcsr_ls.h\
 $(includedir)/parcsr_mv.h\
 $(includedir)/krylov.h
$(new_ij_C___OBJECTS): $(new_ij_C___dependencies)

# sstruct.C++
nodist_sstruct_C___SOURCES = sstruct.C++.cxx
sstruct_C___LDADD = \
 -L$(libdir)\
 -lHYPRE_sstruct_ls\
 -lHYPRE_sstruct_mv\
 -lHYPRE_struct_ls\
 -lHYPRE_struct_mv\
 -lHYPRE_parcsr_ls\
 -lHYPRE_DistributedMatrixPilutSolver \
 -lHYPRE_ParaSails \
 -lHYPRE_Euclid \
 -lHYPRE_MatrixMatrix \
 -lHYPRE_DistributedMatrix \
 -lHYPRE_IJ_mv\
 -lHYPRE_parcsr_mv\
 -lHYPRE_seq_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 $(LAPACKLIBFLAGS)\
 $(BLASLIBFLAGS)\
 $(LIBM)
sstruct_C___dependencies = \
 $(includedir)/utilities.h\
 $(includedir)/HYPRE_sstruct_ls.h\
 $(includedir)/krylov.h
$(sstruct_C___OBJECTS): $(sstruct_C___dependencies)

# struct.C++
nodist_struct_C___SOURCES = struct.C++.cxx
struct_C___LDADD = \
 -L$(libdir)\
 -lHYPRE_struct_ls\
 -lHYPRE_struct_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 $(LIBM)
struct_C___dependencies = \
 $(includedir)/utilities.h\
 $(includedir)/HYPRE_struct_ls.h\
 $(includedir)/struct_mv.h\
 $(includedir)/krylov.h
$(struct_C___OBJECTS): $(struct_C___dependencies)

all: ij ij_mv new_ij struct sstruct

fei++: fei

all++: ij.C++ ij_mv.C++ struct.C++ sstruct.C++

all77: f77_ij f77_ij_mv f77_struct

babel: f77_ij_b ij_b # f77_struct_b struct_b

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
	for i in ij ij_mv new_ij sstruct struct \
		fei f77_ij f77_ij_mv f77_struct \
		f77_ij_b ij_b *.C++* driver.out.* ; \
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
