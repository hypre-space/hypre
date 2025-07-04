#!/bin/bash
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Extract function prototype information in a format needed by code generation scripts.
#
# The script takes a file containing a list of functions and parses the input
# header file to generate the prototype information.
#
# Usage:   <this-script> <function-list> <header>
# Example: <this-script> mup_pre HYPRE_krylov.h > mup_pre.proto
#
# Each output line corresponds to one function prototype and consists of fields
# separated by commas as follows:
#
#   field 1     = function return type
#   field 2     = function name
#   field 3 ... = function prototype arguments
#
# The script should work for any prototype that doesn't contain parentheses as
# data types (e.g., pointers to functions).

scriptdir=`dirname $0`

FFILE=$1
HFILE=$2

# Create a temp header file where each line ends in a semicolon to ensure
# that function prototypes appear on a single line.  First insert EOL after
# every semicolon, then remove EOL on all lines without a semicolon.
sed 's/\;/\;\n/g' $HFILE | awk '{if ($0 ~ /[;]/) {print} else {printf "%s ", $0}}' > $HFILE.tmp

# Match and print the prototype for each function, then strip away extra
# space, parentheses, and commas.
cat $FFILE | while read -r FNAME
do
   awk -v fname=$FNAME '
   BEGIN { pattern = ("[a-zA-Z0-9_]+[[:blank:]*]+" fname "[[:blank:]]*[(][^)]*[)][[:blank:]]*[;]$") }
   {
      # The first call to match speeds things up a bit
      if ( match($0, fname) ){
      if ( match($0, pattern) )
      {
         proto = substr($0, RSTART, RLENGTH)
         match(proto, /[a-zA-Z0-9_]+[[:blank:]\*]+/)
         print substr(proto, RSTART, RLENGTH) , "," , substr(proto, RSTART+RLENGTH)
      }}

   }' $HFILE.tmp |
   sed -e 's/;//g' -e 's/(/,/g' -e 's/)/ /g' -e 's/,/ , /g' -e 's/[[:blank:]][[:blank:]]*/ /g'
done

# Clean up temporary files
rm -f $HFILE.tmp
