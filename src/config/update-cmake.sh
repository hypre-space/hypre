#!/bin/bash

DIRNAME=$(dirname $0)
FOLDERS=(blas examples IJ_mv krylov lapack parcsr_block_mv parcsr_ls parcsr_mv seq_block_mv seq_mv sstruct_ls sstruct_mv test utilities)

for FOLDER in ${FOLDERS[@]}; do
    python3 ${DIRNAME}/update-cmake.py -f ${DIRNAME}/../${FOLDER}
done
