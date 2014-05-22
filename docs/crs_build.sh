#!  /bin/bash

echo starting at $(mydate)

set -ex

# get into the .../hypre.../src directory
src=$(pwd | sed -e "s|/docs$||; s|/src$||")/src
cd $src

# use gnu compilers (for now, default uses intel compilers [on oslic])
opts="--prefix=$INSTALL/hypre CC=mpicc CXX=mpicxx F77=mpif77"
opts+=" --with-print-errors"
# opts+=" --enable-maxdim=4"
# opts+=" --enable-complex"
./configure $opts
make -j 4

echo finished at $(mydate)

