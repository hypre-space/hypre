#!/bin/sh
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

## Should be run from src directory
# Loops over directories with multiprecision files and checks for changes in 
# saved multiprecision function objects.
# NOTE: It must be run on symbols generated from the 
# non-multiprecision build of hypre.

# Assumes hypre has been built (non-multiprecision) to generate object files
###
#  * make distclean
#  * ./configure –enable-debug --enable-complex --with-timing
#  * make -s 
###

# output: prints out directories whose saved files need updating
# Suggested next steps:
# 1. Inspect .saved and .new files in directories that need updating. It helps 
#    if the files are sorted first. So do:
#    sort -c new.srt <directory_name>_functions.new
#    sort -c sav.srt <directory_name>_functions.saved
#    <your-favorite-diff-tool> sav.srt new.srt (eg. meld sav.srt new.srt)
# 2. If new changes are all acceptable, you can do:
#    cp <directory_name>_functions.new to <directory_name>_functions.saved
# 3. Otherwise make edits as needed to new functions added then repeat check or 
#    directly modify .saved file.

SRCDIR=$PWD

MUP_DIRS="blas
      lapack
      utilities
      multivector
      krylov
      seq_mv
      parcsr_mv
      parcsr_block_mv
      distributed_matrix
      IJ_mv
      parcsr_ls
      struct_mv
      struct_ls
      sstruct_mv
      sstruct_ls
      "
for i in $MUP_DIRS; do 
#    echo "checking $i ..."
    cd $i
    rm -rf *.err *.new
    $SRCDIR/config/check_mup.sh
    ls -lt *.err | awk -v x=$i '$5 != 0 {print x" needs updating"}'
    cd $SRCDIR
done

DIST_MUP_DIRS="
      distributed_ls/pilut
      distributed_ls/ParaSails
      distributed_ls/Euclid
      "
for i in $DIST_MUP_DIRS; do 
#    echo "checking $i ..."
    cd $i
    rm -rf *.err *.new
    $SRCDIR/config/check_mup.sh
    ls -lt *.err | awk -v x=$i '$5 != 0 {print x" needs updating"}'
    cd $SRCDIR
done



