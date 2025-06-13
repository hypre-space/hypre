#!/bin/sh
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Generate C code for multiprecision _pre functions
#
# The script takes a file containing a list of functions, a file containing
# header source code (e.g., include lines), and one or more header files, then
# generates a C file with the implementations of each of the functions.
#
# Usage:   <this-script> <function-list> <header-code> <header-1> <header-2> ...
# Example: <this-script> mup_pre HYPRE_krylov.h _hypre_krylov.h > mup_pre.c

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
   FS=" ";
   key = 0;
   # Read saved file data into array
   while (getline < filename)
   {
      structobj = tolower(base)"_data";
      struct = substr($4, 1);
      d = split($2,def,"*");
      fdef = def[d];
      tab  = "   ";
      p_str="";      
      s_str="";
### special case for hypre_SetPrecond function
      if($2~/SetPrecond/)
      {
         fopts="(void*,void*,void*,void*)";
         f1="precond";
         f2="precond_setup";
         data="precond_data";
         n = split($4, si, "*");
         str1="void *"si[n];         
         str2="HYPRE_Int (*"f1")"fopts;
         str3="HYPRE_Int (*"f2")"fopts;
         str4="void *"data;
         p_str=sprintf("%s,\n \t\t%s,\n \t\t%s,\n \t\t%s",str1,str2,str3,str4);
         s_str=sprintf("%s, %s, %s, %s",si[n],f1,f2,data);
      }
      else
      { 
         for(i=3; i<=NF; i+=2)
         {
            j = i+1;
            if(j<NF)
            {
               p_str=sprintf("%s %s %s,",p_str,$i,$j);
               n = split($j, si, "*");
               n = s_str=sprintf("%s %s,",s_str,si[n]);
            }
            else 
            {
               p_str=sprintf("%s %s %s",p_str,$i,$j);
               n = split($j, si, "*");
               s_str=sprintf("%s %s",s_str,si[n]);
            }
         }
         p_str=sprintf(" HYPRE_Precision precision,%s ",p_str);
         s_str=sprintf("%s ",s_str);
      }
      print "/*--------------------------------------------------------------------------*/\n"
      print $1" \n"$2"_pre("p_str")";
      print "{"
      print tab "switch (precision)"
      print tab "{"
      print tab tab "case HYPRE_REAL_SINGLE:"
      print tab tab tab fdef"_flt ("s_str");"
      print tab tab tab "break;"
      print tab tab "case HYPRE_REAL_DOUBLE:"
      print tab tab tab fdef"_dbl ("s_str");"
      print tab tab tab "break;"
      print tab tab "case HYPRE_REAL_LONGDOUBLE:"
      print tab tab tab fdef"_ldbl ("s_str");"
      print tab tab tab "break;"      
      print tab tab "default:"
      print tab tab tab "hypre_printf(\"Unknown solver precision\");"
      print tab "}"
      if($1~/HYPRE_Int/)  # assume we only have HYPRE_Int or void function types for now
      {
         print tab "return hypre_error_flag;"
      }
      print "}\n" 
   }
   close(filename);
}')
@
