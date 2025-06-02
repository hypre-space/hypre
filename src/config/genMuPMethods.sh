#!/bin/sh
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Generate multiprecision method implementation.
# Run from folder where multiprecision functions reside.
# Usage: ../config/genMuPMethods.sh <function prototypes header file> <output file> <function object Root name>
# Example (krylov): ../config/genMuPMethods.sh krylov.h mp_hypre_pcg.c PCG

HDRFILENAME=$1
OUTFILENAME=$2
ROOTNAME=$3

# Generate temporary intermediate file from header file. This will be used to generate C code for multiprecision methods
gsed -n -e "/.*\(^hypre_$ROOTNAME.*HYPRE_$ROOTNAME.*\)/ {s/^[[:blank:]]//; s/(/ /; s/,/ /g; s/\(.*\))/\1/; s/;/ /g; p} " \
            -e "/.*\(^HYPRE_Real.*HYPRE_$ROOTNAME.*\)/ {s/^[[:blank:]]//; s/(/ /; s/,/ /g; s/\(.*\))/\1/; s/;/ /g; p} " \
            -e "/.*\(^void.*HYPRE_$ROOTNAME.*\)/ {s/^[[:blank:]]//; s/(/ /; s/,/ /g; s/\(.*\))/\1/; s/;/ /g; p} " $HDRFILENAME\
            -e "/.*\(^HYPRE_Int.*HYPRE_$ROOTNAME.*\)/ {s/^[[:blank:]]//; s/(/ /; s/,/ /g; s/\(.*\))/\1/; s/;/ /g; p} " > $HDRFILENAME.int

# Generate C code from intermediate file containing function prototypes
#MPSRC=$(awk -v root=$ROOTNAME 'BEGIN{ print "mp_"tolower(root)".c"}')
MPSRC=$OUTFILENAME
INTFILE=$HDRFILENAME.int
../config/genCodeFromProtos.sh $INTFILE $MPSRC $ROOTNAME 2>/dev/null

# Delete intermediate file
rm -rf $INTFILE