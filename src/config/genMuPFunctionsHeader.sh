#!/bin/sh
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Generate multiprecision type protos
# Run from folder where multiprecision functions reside.
# Usage: ../config/genMuPFunctionsHeader.sh <function prototypes header file> <output file> <multiprecision functions saved file>
# Example (struct_ls): ../config/genMuPFunctionsHeader.sh protos.h hypre_struct_ls_mup.h struct_ls_functions.saved

FNAME=$1
FOUT=$2
FLIST=$3

rm -f $FOUT
rm -f $FOUT.int


MUP_HEADER=$FOUT

HEADER_GUARD="${FOUT%.*}_HEADER"
HEADER_GUARD=${HEADER_GUARD^^}

# Generate copyright header
../config/writeHeader.sh $MUP_HEADER

cat >> $MUP_HEADER <<@

/******************************************************************************
 * Header file of multiprecision function prototypes.
 * This is needed for mixed-precision algorithm development.
 *****************************************************************************/

#ifndef $HEADER_GUARD
#define $HEADER_GUARD

#ifdef __cplusplus
extern "C" {
#endif

#if defined (HYPRE_MIXED_PRECISION)

@

while read -r func_name
do
#   sed -n -e "/^.*\b$func_name\b/ {p "$func_name"}" $FNAME >> $FOUT.int 
#   grep -wo $func_name $FNAME >> $FOUT.int

   sed -n -e "/^.*\b$func_name\b/{:continue {s/[[:blank:]]*$//}; {s/ \*[[:space:]]*/ \*/}; p; /).*;/q; n; b continue }" $FNAME >> $FOUT.int 

done < $FLIST

# loop over lines and generate code for each function
FIN=$FOUT.int
cat>> $MUP_HEADER <<@
$(
awk -v filename="$FIN" 'BEGIN{
   FS="[( ]";
   RS=";\n";
   # Read saved file data into array
   while (getline < filename)
   {
# remove leading and trailing white space
         gsub(/^[ \t]+/,"",$0);
         gsub(/[ \t]+$/,"",$0);

         f_str=$2;
         
         # float
         p_str=$0;
         sub($2, f_str"_flt ", p_str);
         p_str=sprintf("%s%s",p_str,";");
         gsub(/HYPRE_Real|HYPRE_Complex/,"hypre_float",p_str);
         print p_str;
         
         # double
         p_str=$0;
         sub($2, f_str"_dbl ", p_str);
         p_str=sprintf("%s%s",p_str,";");
         gsub(/HYPRE_Real|HYPRE_Complex/,"hypre_double",p_str);
         print p_str;
                  
         # long double
         p_str=$0;
         sub($2, f_str"_long_dbl ", p_str);
         p_str=sprintf("%s%s",p_str,";");
         gsub(/HYPRE_Real|HYPRE_Complex/,"hypre_long_double",p_str);
         print p_str;
   }
   close(filename);
}')

#endif

#ifdef __cplusplus
}
#endif

#endif
@

# remove intermediate file
rm -f $FOUT.int
