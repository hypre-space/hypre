#
# Makefile.am --- automake input file for Hypre
#
## Process this file with automake to create Makefile.in

#SUFFIXES = .C++.cxx .C++.o

SUBDIRS = AUTOTEST\
 TEST_fei\
 TEST_ij\
 TEST_sstruct\
 TEST_struct\
 TEST_timing

bin_PROGRAMS = \
 ij\
 ij_mv\
 new_ij\
 sstruct\
 struct

AM_CPPFLAGS = \
 -DHYPRE_TIMING\
 -I$(includedir)

AM_FFLAGS = \
 -I$(includedir)

ij_SOURCES = ij.c
ij_mv_SOURCES = ij_mv.c
new_ij_SOURCES = new_ij.c
sstruct_SOURCES = sstruct.c
struct_SOURCES = struct.c
ij_LDADD = \
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
ij_mv_LDADD = \
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
new_ij_LDADD = \
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
sstruct_LDADD = \
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
struct_LDADD = \
 -L$(libdir)\
 -lHYPRE_struct_ls\
 -lHYPRE_struct_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 $(LIBM)

#ij_LDFLAGS = -no-undefined -rpath $(libdir)
#ij_mv_LDFLAGS = -no-undefined -rpath $(libdir)
#new_ij_LDFLAGS = -no-undefined -rpath $(libdir)
#sstruct_LDFLAGS = -no-undefined -rpath $(libdir)
#struct_LDFLAGS = -no-undefined -rpath $(libdir)

EXTRA_bindir = $(bindir)

EXTRA_PROGRAMS = \
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

f77_ij_SOURCES = f77_ij.f
f77_ij_b_SOURCES = f77_ij_b.f
f77_ij_mv_SOURCES = f77_ij_mv.f
f77_struct_SOURCES = f77_struct.f
#EXTRA_f77_struct_b_SOURCES = f77_struct_b.f
fei_SOURCES = fei.cxx
fei_CPPFLAGS = \
 -DNOFEI\
 -DHYPRE_TIMING\
 -I$(includedir)
ij_b_SOURCES = ij_b.c
#EXTRA_struct_b_SOURCES = struct_b.c
nodist_ij_C___SOURCES = ij.C++.cxx
nodist_new_ij_C___SOURCES = new_ij.C++.cxx
nodist_ij_mv_C___SOURCES = ij_mv.C++.cxx
nodist_struct_C___SOURCES = struct.C++.cxx
nodist_sstruct_C___SOURCES = sstruct.C++.cxx
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
f77_struct_LDADD = \
 -L$(libdir)\
 -lHYPRE_struct_ls\
 -lHYPRE_struct_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 @FLIBS@\
 $(LIBM)
#f77_struct_b_LDADD = \
# -L$(libdir)\
# -lHYPRE_struct_ls\
# -lHYPRE_struct_mv\
# -lHYPRE_krylov\
# -lHYPRE_utilities\
# @FLIBS@
fei_LDADD = \
 -L$(libdir)\
 -lHYPRE_LSI\
 -lHYPRE_superlu\
 $(LAPACKLIBFLAGS)\
 $(BLASLIBFLAGS)\
 $(LIBM)
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
#struct_b_LDADD = \
# -L$(libdir)\
# -lHYPRE_struct_ls\
# -lHYPRE_struct_mv\
# -lHYPRE_krylov\
# -lHYPRE_utilities\
# $(LIBM)
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
struct_C___LDADD = \
 -L$(libdir)\
 -lHYPRE_struct_ls\
 -lHYPRE_struct_mv\
 -lHYPRE_krylov\
 -lHYPRE_utilities\
 $(LIBM)
#f77_ij_LDFLAGS = -static
#f77_ij_b_LDFLAGS = -static
#f77_ij_mv_LDFLAGS = -static
#f77_struct_LDFLAGS = -static
##f77_struct_b_LDFLAGS = -static
#fei_LDFLAGS = -static
#ij_b_LDFLAGS = -static
##struct_b_LDFLAGS = -static

EXTRA_DIST = \
 funcs.sh\
 runtest.sh

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

##.c.C++.o:
##	cp -f $< ${@:.o=.cxx}
##	${CXX} -o $@ -c ${CXXFLAGS} ${@:.o=.cxx}
##	rm -f ${@:.o=.cxx}

clean-local:
	@ \
	for i in fei f77_ij f77_ij_mv f77_struct \
		f77_ij_b ij_b *.C++* driver.out.* ; \
	do \
	  if [ -e $$i ]; \
	  then \
	    rm -f $$i; \
	  fi; \
	done
