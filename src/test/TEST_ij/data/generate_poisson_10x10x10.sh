#!/bin/bash
#
# This bash script is intended to regenerate the binary files under
# the folders poisson_10x10x10_np1 and poisson_10x10x10_np4 if the
# binary input/output functions from the IJ interface get changed.

function movefile() {
   mv -v ${1}.${SUFFIX} ${DEST_DIR}/${1}.${PREC}.${SUFFIX}
}

CWD=$(pwd)
HYPRE_SRC=$(dirname "$(readlink -f "${CWD}/../../")")
PREC_ARRAY=("i4f4" "i4f8" "i8f4" "i8f8")
CONFOPTS_ARRAY=("--disable-mixedint --enable-single"
                "--disable-mixedint --disable-single"
                "--enable-mixedint --enable-single"
                "--enable-mixedint --disable-single")
IJOPTS="-n 10 10 10 -printbin -tol 1 -solver 2"

for I in {0..3}; do
    PREC=${PREC_ARRAY[${I}]}
    CONFOPTS=${CONFOPTS_ARRAY[${I}]}

    cd ${HYPRE_SRC}
    ./configure --enable-debug ${CONFOPTS}
    make clean
    make -j
    cd test
    make clean ij

    # Sequential case
    DEST_DIR=${HYPRE_SRC}/test/TEST_ij/data/poisson_10x10x10_np1
    mpirun -np 1 ./ij ${IJOPTS}
    SUFFIX=00000.bin
    movefile IJ.out.A
    movefile IJ.out.b
    movefile IJ.out.x0

    # Parallel case
    DEST_DIR=${HYPRE_SRC}/test/TEST_ij/data/poisson_10x10x10_np4
    mpirun -np 4 ./ij ${IJOPTS} -P 2 2 1
    for J in {0..3}; do
       SUFFIX=0000${J}.bin
       movefile IJ.out.A
       movefile IJ.out.b
       movefile IJ.out.x0
    done

    rm -rf IJ.out.x.*
done
