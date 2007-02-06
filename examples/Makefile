########################################################################
# Compiler and external dependences
########################################################################
CC        = mpicc
F77       = mpif77
CXX       = mpiCC
F90       = mpif90
HYPRE_DIR = ../hypre

########################################################################
# Compiling and linking options
########################################################################
COPTS     = -g -pedantic -Wall
CINCLUDES = -I$(HYPRE_DIR)/include
CDEFS     = -DHAVE_CONFIG_H -DHYPRE_TIMING
CFLAGS    = $(COPTS) $(CINCLUDES) $(CDEFS)
FOPTS     = -g
FINCLUDES = $(CINCLUDES)
FDEFS     = $(CDEFS)
FFLAGS    = $(FOPTS) $(FINCLUDES) $(FDEFS)
CXXOPTS   = $(COPTS)
CXXINCLUDES = $(CINCLUDES) -I..
CXXDEFS   = $(CDEFS)
IFLAGS_BXX = -I../babel-runtime/sidl
CXXFLAGS  = $(CXXOPTS) $(CXXINCLUDES) $(CXXDEFS) $(IFLAGS_BXX)
IF90FLAGS = -I../babel/bHYPREClient-F90
F90FLAGS = $(FFLAGS) $(IF90FLAGS)


LINKOPTS  = $(COPTS)
LIBS      = -L$(HYPRE_DIR)/lib -lHYPRE -lg2c -lm
LFLAGS    = $(LINKOPTS) $(LIBS) -lstdc++
LFLAGS_B =\
 -L${HYPRE_DIR}/lib\
 -L${HYPRE_DIR}/../babel-runtime/sidl/.libs\
 -lbHYPREClient-C\
 -lbHYPREClient-CX\
 -lbHYPREClient-F\
 -lbHYPRE\
 -lsidl -ldl -lxml2
LFLAGS90 =

########################################################################
# Rules for compiling the source files
########################################################################
.SUFFIXES: .c .f .cxx .f90

.c.o:
	$(CC) $(CFLAGS) -c $<
.f.o:
	$(F77) $(FFLAGS) -c $<
.cxx.o:
	$(CXX) $(CXXFLAGS) -c $<

# This Fortran 90 build code is adapted from the sample in the Babel Users' Guide,
# which explains the separate preprocessing steps as a workaround to the 31-
# character name limit in Fortran 90.
.f90.o:
	$(CC) -E -traditional -P -o $*.tmp -x c $(IF90FLAGS) $<
	sed -e 's/^#pragma.*$$//' < $*.tmp > $*-pp.f90
	$(F90) -c -o $@ $(F90FLAGS) $*-pp.f90
	rm -f $*.tmp
# ... we can't delete the $*-pp.f90 files because they are the source code which
# debuggers need to look at

########################################################################
# List of all programs to be compiled
########################################################################
ALLPROGS = ex1 ex2 ex3 ex4 ex5 ex6 ex7 ex8 ex9 ex10
BABELPROGS = ex5b ex5b77 ex5bxx ex6b ex6b77
# ... ex5bp and ex5b90 are not in BABELPROGS because they require a
# software environment which many people haven't set up.

all: $(ALLPROGS)

babel: $(BABELPROGS)

########################################################################
# Example 1
########################################################################
ex1: ex1.o
	$(CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 2
########################################################################
ex2: ex2.o
	$(CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 3
########################################################################
ex3: ex3.o
	$(CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 4
########################################################################
ex4: ex4.o
	$(CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 5
########################################################################
ex5: ex5.o
	$(CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 5 Babel C
########################################################################
ex5b: ex5b.o
	$(CC) -o $@ $^ $(LFLAGS_B) $(LFLAGS)

########################################################################
# Example 5 Babel Fortran 77
########################################################################
ex5b77: ex5b77.o
	$(F77) -o $@ $^ $(LFLAGS_B) $(LFLAGS)

########################################################################
# Example 5 Babel Fortran 90
########################################################################
ex5b90: ex5b90.o
	$(F90) -o $@ $^ $(LFLAGS_B) $(LFLAGS) $(LFLAGS90)

########################################################################
# Example 5 Babel C++
########################################################################
ex5bxx: ex5bxx.o
	$(CXX) -o $@ $^ $(LFLAGS_B) $(LFLAGS)

########################################################################
# Example 6
########################################################################
ex6: ex6.o
	$(CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 6 Babel C
########################################################################
ex6b: ex6b.o
	$(CC) -o $@ $^ $(LFLAGS_B) $(LFLAGS)

########################################################################
# Example 6 Babel Fortran 77
########################################################################
ex6b77: ex6b77.o
	$(F77) -o $@ $^ $(LFLAGS_B) $(LFLAGS)

########################################################################
# Example 7
########################################################################
ex7: ex7.o
	$(CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 8
########################################################################
ex8: ex8.o
	$(CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 9
########################################################################
ex9: ex9.o
	$(CC) -o $@ $^ $(LFLAGS)

########################################################################
# Example 10
########################################################################
ex10: ex10.o
	$(CXX) -o $@ $^ $(LFLAGS)

########################################################################
# Clean up
########################################################################
clean:
	rm -f $(ALLPROGS:=.o)
	rm -f $(BABELPROGS:=.o)
veryclean distclean: clean
	rm -f $(ALLPROGS) $(ALLPROGS:=*~)
	rm -f $(BABELPROGS) $(BABELPROGS:=*~)
	rm -rf *.html README_files
	rm -rf $(BABELPROGS:=*-pp.f90)
