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

      arg_pre      = sprintf(" HYPRE_Precision precision,%s",p_str)

      # First replace HYPRE_Real* and HYPRE_Complex* with void*
      gsub(/(HYPRE_Real|HYPRE_Complex)[[:blank:]]*[*]/, "void *", arg_pre)
      gsub(/(HYPRE_Real|HYPRE_Complex)/, "hypre_long_double", arg_pre)

      print fret"\n"fdef"_pre("arg_pre");\n"         >> outh

      print "/*--------------------------------------------------------------------------*/\n" >> outc
      print fret"\n"fdef"_pre("arg_pre")"                                                      >> outc
      print "{"                                                                                >> outc
      print tab "switch (precision)"                                                           >> outc
      print tab "{"                                                                            >> outc
      print tab tab "case HYPRE_REAL_SINGLE:"                                                  >> outc
      print tab tab tab "return "fdef"_flt("s_str");"                                          >> outc
      print tab tab "case HYPRE_REAL_DOUBLE:"                                                  >> outc
      print tab tab tab "return "fdef"_dbl("s_str");"                                          >> outc
      print tab tab "case HYPRE_REAL_LONGDOUBLE:"                                              >> outc
      print tab tab tab "return "fdef"_long_dbl("s_str");"                                     >> outc
      print tab tab "default:" >> outc
      if(fret == "void")
      {
         print tab tab tab "hypre_printf(\"Unknown solver precision\");"                       >> outc
      }
      else
      {
         print tab tab tab "{ "fret" value = 0; hypre_printf(\"Unknown solver precision\"); return value; }" >> outc
      }
      print tab "}"                                                                            >> outc
      print "}\n"                                                                              >> outc
   }
   close(filename)
}'
