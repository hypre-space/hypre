#!/bin/csh

set ext = ""
set wl = "with-lapack"
set nol = "without-lapack"
set mpirun = "mpimon"

if ( $1 == "without-MPI" ) then
#
set compiler = cc
set compiler_options1 = "-DHAVE_CONFIG_H"
set compiler_options2 = "-O2 -g"
set compiler_options3 = "-DHYPRE_MODE -DOPTIMIZED_DH"
set install = "/usr/bin/install -c"
if ( $2 != $wl ) then
echo ./configure --without-MPI --without-lapack
set lapack_libraries = "-L../../hypre/lib -lHYPRE_lapack -lHYPRE_blas -lm"
else
echo ./configure --without-MPI --with-lapack
set lapack_libraries = "-L/usr/local/lib -lblas -llapack -lg2c -lm"
endif
#
else if ( $1 == "cygwin" ) then
#
echo ./configure --without-MPI
set compiler = cc
set compiler_options1 = "-DHAVE_CONFIG_H"
set compiler_options2 = "-O2"
set compiler_options3 = "-DHYPRE_MODE -DOPTIMIZED_DH"
set install = "/usr/bin/install -c"
set lapack_libraries = "-L../../hypre/lib -lHYPRE_lapack -lHYPRE_blas -lm"
set ext = ".exe"
set mpirun = "cygwin"
#
else if ( $1 == "blue" ) then
#
if ( $2 == "strict" ) then
#
echo ./nopoe ./configure --with-strict-checking 
set compiler = xlc
set compiler_options1 = "-DHAVE_CONFIG_H"
set compiler_options2 = ""
set compiler_options3 = "-g -qinfo=dcl:eff:pro:rea:ret:use"
set install = "/usr/local/gnu/bin/install -c"
set mpirun = "poe_strict"
set lapack_libraries = "-lessl -lm"
#
else
#
set compiler = mpcc
set compiler_options1 = "-DHAVE_CONFIG_H"
set compiler_options2 = "-O3 -qstrict"
set compiler_options3 = "-DHYPRE_MODE -DOPTIMIZED_DH"
set install = "/usr/local/gnu/bin/install -c"
set mpirun = "poe"
if ( $2 == $wl ) then
echo ./nopoe ./configure --with-blas=\"-L/usr/local/lib -lblas -lxlf90\" 
set lapack_libraries = "-llapack -lxlf90 -L/usr/local/lib -lblas -lm" 
else if ( $2 == $nol ) then
echo ./nopoe ./configue --with-blas=no --with-lapack=no
set lapack_libraries = "-L../../hypre/lib -lHYPRE_lapack -lHYPRE_blas -lm"
else
echo ./nopoe ./configure
set lapack_libraries = "-lessl -lm"
endif
#
endif
#
else if ( $1 == "mpich" ) then
#
set compiler = mpicc
set compiler_options1 = "-DHAVE_CONFIG_H"
set compiler_options2 = "-O"
set compiler_options3 = "-DHYPRE_MODE -DOPTIMIZED_DH"
set install = "../../config/install-sh -c"
set mpirun = "mpirun"
if ( $2 == $wl ) then
echo ./configure --with-lapack --with-blas
set lapack_libraries = "-L/usr/local/lib -lblas -llapack -lg2c -lm"
else if ( $2 == $nol ) then
echo ./configue --without-lapack --without-blas
set lapack_libraries = "-L../../hypre/lib -lHYPRE_lapack -lHYPRE_blas -lm"
else
echo ./configure
set lapack_libraries = "-lcxml -lm"
endif
#
else if ( $1 == "scali" ) then
set compiler = "gcc -O2 -fomit-frame-pointer -D_REENTRANT -I/opt/scali/include"
set compiler_options1 = "-DHAVE_CONFIG_H"
set compiler_options2 = "-O2"
set compiler_options3 = "-DHYPRE_MODE -DOPTIMIZED_DH"
set install = "/usr/bin/install -C"
#
if ( $2 != $nol ) then
echo ./configure
set lapack_libraries = "-L/usr/local/lib -lblas -llapack -lg2c -lm"
else
echo ./configue --without-lapack --without-blas 
set lapack_libraries = "-L../../hypre/lib -lHYPRE_lapack -lHYPRE_blas -lm"
endif
#
else
#
set compiler = mpicc
set compiler_options1 = "-DHAVE_CONFIG_H"
set compiler_options2 = "-O2"
set compiler_options3 = "-DHYPRE_MODE -DOPTIMIZED_DH"
set install = "/usr/bin/install -C"
#
if ( $# < 1 || $1 != $nol ) then
echo ./configure
set lapack_libraries = "-L/usr/local/lib -lblas -llapack -lg2c -lm"
else
echo ./configue --without-lapack --without-blas 
set lapack_libraries = "-L../../hypre/lib -lHYPRE_lapack -lHYPRE_blas -lm"
#
endif
#
endif

echo using $compiler as a compiler...
echo using $lapack_libraries

if ( $1 == "scali" ) then
set builder = mpicc
else
set builder = $compiler
endif

set fail = 0

echo building fortran_matrix...
cd ./utilities
$compiler $compiler_options1 -I. -I../../blas -I../../utilities $compiler_options2 -c fortran_matrix.c
rm -f libHYPRE_fortran_matrix.a
ar cru libHYPRE_fortran_matrix.a fortran_matrix.o
ranlib libHYPRE_fortran_matrix.a
mkdir -p -- . ../hypre/lib
 $install -m 644 libHYPRE_fortran_matrix.a ../hypre/lib/libHYPRE_fortran_matrix.a
 ranlib ../hypre/lib/libHYPRE_fortran_matrix.a
mkdir -p -- . ../hypre/include
 $install -m 644 fortran_matrix.h ../hypre/include/fortran_matrix.h

cd ../multivector

echo building interpreter...
mkdir -p -- . ../hypre/include
 $install -m 644 HYPRE_interpreter.h ../hypre/include/HYPRE_interpreter.h

echo building multivector...
$compiler $compiler_options1 -I. -I.. -I../.. -I../../utilities $compiler_options2 -c temp_multivector.c
$compiler $compiler_options1 -I. -I.. -I../.. -I../../utilities $compiler_options2 -c multivector.c
rm -f libHYPRE_multivector.a
ar cru libHYPRE_multivector.a multivector.o temp_multivector.o
ranlib libHYPRE_multivector.a
mkdir -p -- . ../hypre/lib
 $install -m 644 libHYPRE_multivector.a ../hypre/lib/libHYPRE_multivector.a
 ranlib ../hypre/lib/libHYPRE_multivector.a
mkdir -p -- . ../hypre/include
 $install -m 644 multivector.h ../hypre/include/multivector.h

cd ../krylov

echo bulding lobpcg...
$compiler $compiler_options1 -I. -I../multivector -I../utilities -I../.. -I../../utilities $compiler_options2 -c lobpcg.c
$compiler $compiler_options1 -I. -I../multivector -I../utilities -I../.. -I../../krylov -I../../utilities   $compiler_options2 -c HYPRE_lobpcg.c
rm -f libHYPRE_lobpcg.a
ar cru libHYPRE_lobpcg.a HYPRE_lobpcg.o lobpcg.o
ranlib libHYPRE_lobpcg.a
mkdir -p -- . ../hypre/lib
 $install  -m 644 libHYPRE_lobpcg.a ../hypre/lib/libHYPRE_lobpcg.a
 ranlib ../hypre/lib/libHYPRE_lobpcg.a
mkdir -p -- . ../hypre/include
 $install -m 644 lobpcg.h ../hypre/include/lobpcg.h
 $install -m 644 HYPRE_lobpcg.h ../hypre/include/HYPRE_lobpcg.h

echo building parcsr_int...
cd ../parcsr_ls
$compiler $compiler_options1 $compiler_options3 -I. -I../multivector -I../.. -I../../utilities -I../../krylov -I../../parcsr_ls -I../../parcsr_mv -I../../IJ_mv -I../../seq_mv $compiler_options2 -c HYPRE_parcsr_int.c
rm -f libHYPRE_parcsr_int.a
ar cru libHYPRE_parcsr_int.a HYPRE_parcsr_int.o
ranlib libHYPRE_parcsr_int.a
mkdir -p -- . ../hypre/lib
 $install -m 644 libHYPRE_parcsr_int.a ../hypre/lib/libHYPRE_parcsr_int.a
 ranlib ../hypre/lib/libHYPRE_parcsr_int.a
mkdir -p -- . ../hypre/include
 $install -m 644 HYPRE_parcsr_int.h ../hypre/include/HYPRE_parcsr_int.h

echo building struct_int...
cd ../struct_ls
$compiler $compiler_options1 $compiler_options3 -I. -I../multivector -I../.. -I../../utilities -I../../krylov -I../../struct_ls -I../../struct_mv -I../../IJ_mv -I../../seq_mv $compiler_options2 -c HYPRE_struct_int.c
rm -f libHYPRE_struct_int.a
ar cru libHYPRE_struct_int.a HYPRE_struct_int.o
ranlib libHYPRE_struct_int.a
mkdir -p -- . ../hypre/lib
 $install -m 644 libHYPRE_struct_int.a ../hypre/lib/libHYPRE_struct_int.a
 ranlib ../hypre/lib/libHYPRE_struct_int.a
mkdir -p -- . ../hypre/include
 $install -m 644 HYPRE_struct_int.h ../hypre/include/HYPRE_struct_int.h

echo building sstruct_int...
cd ../sstruct_ls
$compiler $compiler_options1 $compiler_options3 -I. -I../multivector -I../.. -I../../utilities -I../../krylov -I../../sstruct_ls -I../../sstruct_mv -I../../struct_ls -I../../struct_mv -I../../parcsr_ls -I../../parcsr_mv -I../../IJ_mv -I../../seq_mv $compiler_options2 -c HYPRE_sstruct_int.c
rm -f libHYPRE_sstruct_int.a
ar cru libHYPRE_sstruct_int.a HYPRE_sstruct_int.o
ranlib libHYPRE_sstruct_int.a
mkdir -p -- . ../hypre/lib
 $install -m 644 libHYPRE_sstruct_int.a ../hypre/lib/libHYPRE_sstruct_int.a
 ranlib ../hypre/lib/libHYPRE_sstruct_int.a
mkdir -p -- . ../hypre/include
 $install -m 644 HYPRE_sstruct_int.h ../hypre/include/HYPRE_sstruct_int.h

cd ../test

echo building ij_es$ext...
$compiler $compiler_options1 -DHYPRE_TIMING -I. -I../.. -I../hypre/include -I../../utilities -I../../krylov -I../../parcsr_ls -I../../parcsr_mv -I../../IJ_mv -I../../seq_mv $compiler_options2 -c ij_es.c 
/bin/sh ../../libtool --mode=link --tag=CC $builder $compiler_options2 -o ij_es ij_es.o -L../hypre/lib -lHYPRE_lobpcg -lHYPRE_parcsr_int -lHYPRE_multivector -lHYPRE_fortran_matrix -L../../hypre/lib -lHYPRE_parcsr_ls -lHYPRE_DistributedMatrixPilutSolver -lHYPRE_ParaSails -lHYPRE_Euclid -lHYPRE_IJ_mv -lHYPRE_MatrixMatrix -lHYPRE_DistributedMatrix -lHYPRE_parcsr_mv -lHYPRE_seq_mv -lHYPRE_krylov -lHYPRE_utilities $lapack_libraries

echo building struct_es$ext... 
$compiler $compiler_options1 -DHYPRE_TIMING -I. -I../.. -I../hypre/include -I../../utilities  -I../../krylov -I../../struct_ls -I../../struct_mv -I../../seq_mv $compiler_options2 -c -o struct-struct_es.o `test -f 'struct_es.c' || echo './'`struct_es.c
/bin/sh ../../libtool --mode=link --tag=CC $builder  $compiler_options2 -o struct_es struct-struct_es.o -L../hypre/lib -lHYPRE_lobpcg -lHYPRE_struct_int -lHYPRE_multivector -lHYPRE_fortran_matrix -L../../hypre/lib -lHYPRE_struct_ls -lHYPRE_struct_mv -lHYPRE_krylov -lHYPRE_utilities $lapack_libraries

echo building sstruct_es$ext...
$compiler $compiler_options1 -DHYPRE_TIMING -I. -I../.. -I../hypre/include -I../../utilities -I../../krylov -I../../sstruct_ls -I../../sstruct_mv -I../../struct_ls -I../../struct_mv -I../../parcsr_ls -I../../parcsr_mv -I../../IJ_mv -I../../seq_mv $compiler_options2 -c -o sstruct-sstruct_es.o `test -f 'sstruct.c' || echo './'`sstruct_es.c
/bin/sh ../../libtool --mode=link --tag=CC $builder  $compiler_options2  -o sstruct_es sstruct-sstruct_es.o -L../hypre/lib -lHYPRE_lobpcg -lHYPRE_sstruct_int -lHYPRE_multivector -lHYPRE_fortran_matrix -L../../hypre/lib -lHYPRE_sstruct_ls -lHYPRE_sstruct_mv -lHYPRE_struct_ls -lHYPRE_struct_mv -lHYPRE_parcsr_ls -lHYPRE_DistributedMatrixPilutSolver -lHYPRE_ParaSails -lHYPRE_Euclid -lHYPRE_MatrixMatrix -lHYPRE_DistributedMatrix -lHYPRE_IJ_mv -lHYPRE_parcsr_mv -lHYPRE_seq_mv -lHYPRE_krylov -lHYPRE_utilities $lapack_libraries

#if ( $? != 0 ) then
#exit 1
#endif

echo building test scripts...
cd ./TEST_ij_es
cp ../ij_es$ext ./
cat ../$mpirun.run solvers.jobs.in > solvers.jobs
cat ../$mpirun.run options.jobs.in > options.jobs
cat ../$mpirun.run in_out.jobs.in > in_out.jobs
chmod +x solvers.jobs
chmod +x options.jobs
chmod +x in_out.jobs

cd ../TEST_struct_es
cp ../struct_es$ext ./
cat ../$mpirun.run solvers.jobs.in > solvers.jobs
cat ../$mpirun.run options.jobs.in > options.jobs
chmod +x solvers.jobs
chmod +x options.jobs

cd ../TEST_sstruct_es
cp ../sstruct_es$ext ./
cp ../sstruct.in.default ./
cat ../$mpirun.run solvers.jobs.in > solvers.jobs
cat ../$mpirun.run options.jobs.in > options.jobs
chmod +x solvers.jobs
chmod +x options.jobs


