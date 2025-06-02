#!/bin/sh
# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# Generate C code from intermediate file containing function prototypes. See genMuPMethods.sh for intermediate file generation.
# Usage: ../config/genCodeFromProtos.sh <intermediate_input_file> <output file> <function object Root name>
# Example (krylov): ../config/genCodeFromProtos.sh krylov.h.int mp_hypre_pcg.c PCG

FIN=$1
FOUT=$2
ROOTNAME=$3
INTERNAL_HEADER=_hypre_utilities.h
FINC=$(echo $FIN | awk -F '.int' '{print $1}')

# Generate copyright header
../config/writeHeader.sh $FOUT
# include files
cat >> $FOUT <<@

#include "$INTERNAL_HEADER"
#include "$FINC"
@

# loop over lines and generate code for each function
cat>> $FOUT <<@
$(
awk -v filename="$FIN" -v base="$ROOTNAME" 'BEGIN{
   FS=" ";
   key = 0;
   # Read saved file data into array
   print "\n\n"
   while (getline < filename)
   {
      structptr = "hypre_"toupper(base)"Data *";
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
      }
      print "/*--------------------------------------------------------------------------"
      print "*", $2;
      print "*--------------------------------------------------------------------------*/"
      print $1" \n"$2"("p_str")";
      print "{"
      print tab structptr structobj " = ("structptr")"struct";"
      print tab "switch ("structobj "-> solver_precision)"
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
      print tab tab tab "hypre_printf(\"Unknown solver precision\" );"
      print tab "}"
      if($1~/HYPRE_Int/)
      {
         print tab "return hypre_error_flag;"
      }
      else if($1~/void/) # void * case (too specific, but works for now)
      {
         print tab "return ("$1 "\*)"structobj";"
      }
      print "}\n" 
   }
   close(filename);
}')
@
