#!/bin/sh
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Generate C prototypes for multiprecision _pre functions
#
# The script takes a file containing a list of functions and one or more header
# files, then generates a C file with the function prototypes.
#
# Usage:   <this-script> <function-list> <header-code> <header-1> <header-2> ...
# Example: <this-script> mup.pre HYPRE_krylov.h _hypre_krylov.h > mup_pre.h

scriptdir=`dirname $0`

FFILE=$1
shift
HFILES="$*"

# Create a function prototype information file.  To protect against duplicate
# prototypes, the output is sorted and duplicate lines are deleted.
for HFILE in $HFILES
do
   ./$scriptdir/gen_proto_info.sh $FFILE $HFILE

done | (export LC_COLLATE=C; sort) | uniq > $FFILE.proto

# loop over lines and generate code for each function
cat <<@
$(
awk -v filename="$FFILE.proto" 'BEGIN {
   FS=" , "
   # Read the prototype info file
   while (getline < filename)
   {
      fret  = $1
      fdef  = $2
      tab   = "   "
      p_str = ""
      for(i=3; i<=NF; i++)
      {
         match($i, /[a-zA-Z0-9_]+[[:blank:]]*$/)
         argtype = substr($i, 0, RSTART-1)
         argname = substr($i, RSTART, RLENGTH)
         sub(/^[[:blank:]]*/, "", argtype); sub(/[[:blank:]]*$/, "", argtype)
         sub(/^[[:blank:]]*/, "", argname); sub(/[[:blank:]]*$/, "", argname)
         p_str = sprintf("%s %s %s", p_str, argtype, argname)
         if(i<NF)
         {
            p_str = sprintf("%s,", p_str)
         }
      }
      p_str=sprintf("%s ",p_str)

      arg_flt      = sprintf("%s",p_str)
      arg_dbl      = sprintf("%s",p_str)
      arg_long_dbl = sprintf("%s",p_str)
      arg_pre      = sprintf(" HYPRE_Precision precision,%s",p_str)

      gsub(/(HYPRE_Real|HYPRE_Complex)/, "hypre_float", arg_flt)
      gsub(/(HYPRE_Real|HYPRE_Complex)/, "hypre_double", arg_dbl)
      gsub(/(HYPRE_Real|HYPRE_Complex)/, "hypre_long_double", arg_long_dbl)

      # First replace HYPRE_Real* and HYPRE_Complex* with void*
      gsub(/(HYPRE_Real|HYPRE_Complex)[[:blank:]]*[*]/, "void *", arg_pre)
      gsub(/(HYPRE_Real|HYPRE_Complex)/, "hypre_long_double", arg_pre)

      print fret"\n"fdef"_flt("arg_flt");"
      print fret"\n"fdef"_dbl("arg_dbl");"
      print fret"\n"fdef"_long_dbl("arg_long_dbl");"
      print fret"\n"fdef"_pre("arg_pre");\n"
   }
   close(filename)
}')
@
