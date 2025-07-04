#!/bin/sh
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Generate C implementations for multiprecision _pre functions
#
# The script takes a file containing a list of functions and one or more header
# files, then generates a C file with the function implementations.
#
# Usage:   <this-script> <function-list> <header-1> <header-2> ...
# Example: <this-script> mup.pre HYPRE_krylov.h _hypre_krylov.h > mup_pre.c

scriptdir=`dirname $0`

PFILE=$1
OUTC="$2.c"
OUTH="$2.h"

echo "" > $OUTC
echo "" > $OUTH

# loop over lines and generate code for each function
awk -v filename="$PFILE" -v outc="$OUTC" -v outh="$OUTH" 'BEGIN {
   FS=" , "
   # Read the prototype info file
   while (getline < filename)
   {
      fret  = $1
      fdef  = $2
      tab   = "   "
      p_str = ""
      s_str = ""
      for(i=3; i<=NF; i++)
      {
         match($i, /[a-zA-Z0-9_]+[[:blank:]]*$/)
         argtype = substr($i, 0, RSTART-1)
         argname = substr($i, RSTART, RLENGTH)
         sub(/^[[:blank:]]*/, "", argtype); sub(/[[:blank:]]*$/, "", argtype)
         sub(/^[[:blank:]]*/, "", argname); sub(/[[:blank:]]*$/, "", argname)
         p_str = sprintf("%s %s %s", p_str, argtype, argname)
         s_str = sprintf("%s %s", s_str, argname)
         if(i<NF)
         {
            p_str = sprintf("%s,", p_str)
            s_str = sprintf("%s,", s_str)
         }
      }
      p_str=sprintf("%s ",p_str)
      s_str=sprintf("%s ",s_str)

      arg_flt      = sprintf("%s",p_str)
      arg_dbl      = sprintf("%s",p_str)
      arg_long_dbl = sprintf("%s",p_str)

      gsub(/(HYPRE_Real|HYPRE_Complex)/, "hypre_float", arg_flt)
      gsub(/(HYPRE_Real|HYPRE_Complex)/, "hypre_double", arg_dbl)
      gsub(/(HYPRE_Real|HYPRE_Complex)/, "hypre_long_double", arg_long_dbl)

      print fret"\n"fdef"_flt("arg_flt");"           >> outh
      print fret"\n"fdef"_dbl("arg_dbl");"           >> outh
      print fret"\n"fdef"_long_dbl("arg_long_dbl");" >> outh

      # Put fixed implementation code here if needed
   }
   close(filename)
}'
