#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Check that current multiprecision functions are up to date.  The script will
# create three files:
#
#   mup_check.old - contains the original function list
#   mup_check.new - contains the newly generated list
#   mup_check.err - the difference between the old/new files
#
# NOTE: Must be run on symbols generated from the non-multiprecision build.

scriptdir=`dirname $0`

# Check if terminal supports colors
if [ -t 1 ]; then
    # Use colors
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    NC='\033[0m' # No Color
else
    # Plain text
    RED=''
    GREEN=''
    NC=''
fi

export LC_COLLATE=C  # sort by listing capital letters first

cat mup.fixed mup.functions mup.methods | sort | uniq  > mup_check.old
$scriptdir/generate_function_list.sh    | sort | uniq  > mup_check.new

# Remove functions listed in mup.exclude (if it exists)
if [ -e mup.exclude ]; then
    egrep -v -f mup.exclude mup_check.new > mup_check.new.tmp
    mv  mup_check.new.tmp mup_check.new
fi

diff -wc mup_check.old mup_check.new                   > mup_check.err

SZ=`ls -l mup_check.err | awk '{print $5}'`
if [ "$SZ" != 0 ]; then
    echo -ne "${RED}UPDATE${NC} - see $(pwd)/mup_check.err\n"
    exit 1
else
    echo -ne "${GREEN}OK${NC}\n"
    exit 0
fi
