#!/bin/sh
# Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# example call
# ./runcheck.sh fname.out fname.saved 1.0e-6 1.0e-6

FNAME=$1
SNAME=$2
RTOL=$3
ATOL=$4

if [ x$RTOL = "x" ];
then
    RTOL=0.0
fi

if [ x$ATOL = "x" ];
then
    ATOL=-1.0
fi

#echo "runcheck rtol = $RTOL, atol = $ATOL"

awk -v ofilename="$FNAME" -v sfilename="$SNAME" 'BEGIN{
   FS=" ";
   saved_key = 0;
   ln = 0;
   # Read saved file data into array
   while (getline < sfilename)
   {
      ln++;
      if(NF > 0 && substr($1,1,1) !~ /#/)
      {
         # loop over fields in current line
         for(id=1; id<=NF; id++)
         {
            # check if field is numeric
            if($id ~ /^[0-9]+/)
            {
               ln_id[saved_key]=ln;
               saved_line[saved_key]=$0;
               saved_array[++saved_key]=$id;
            }
         }
      }
   }
   close(sfilename);

   # Read out file data into array
   out_key=0;
   ln=0;
   while (getline < ofilename)
   {
      if(NF > 0 && substr($1,1,1) !~ /#/)
      {
         # loop over fields in current line
         for(id=1; id<=NF; id++)
         {
            # check if field is numeric
            if($id ~ /^[0-9]+/)
            {
               out_line[out_key]=$0;
               out_array[++out_key]=$id;
            }
         }
      }
   }
   close(ofilename);
   
   # compare data arrays
   if(saved_key != out_key)
   {
       printf "Number of numeric entries do not match!!\n"
       printf "Saved file (%d entries)  Output file (%d entries)\n\n", saved_key, out_key
   }
   
   # compare numeric entries
   rtol = "'"$RTOL"'" + 0. # adding zero is necessary to convert from string to number
   atol = "'"$ATOL"'" + 0. # adding zero is necessary to convert from string to number
   for(id=1; id<=saved_key; id++)
   {
      # get value from arrays
      saved_val = saved_array[id];
      out_val = out_array[id];      

      # floating point field comparison
      if(length(saved_val) != length(int(saved_val)) && length(out_val) != length(int(out_val))) 
      {
         err = saved_val - out_val;
         # get absolute value of err and saved_val
         err = err < 0 ? -err : err;
         saved_val = saved_val < 0 ? -saved_val : saved_val;
         # abs err <= atol or rel err <= rtol
         if(err <= atol || err <= rtol*saved_val)
         {
            #print "PASSED"
         }
         else
         {
            pass=0;
            printf "(%d) - %s\n", ln_id[id-1], saved_line[id-1]
            printf "(%d) + %s      (err %.2e)\n\n", ln_id[id-1], out_line[id-1], err
         }
      }
      else if(length(saved_val) == length(int(saved_val)) && length(out_val) == length(int(out_val)))# integer comparison
      {
         tau = saved_val - out_val;
         # get absolute value of tau
         tau = tau < 0 ? -tau : tau;
         # get ceiling of rtol*saved_val (= max allowed change)
         gamma = int(1.0 + rtol*saved_val);
         if(tau <= gamma)
         {
            #print "PASSED"
         }
         else
         {
            pass=0;
            printf "(%d) %s <-- %s, err %d\n", ln, $0, saved_val, tau
         }
      }      
      else # type mismatch
      {
         printf "Numeric type mismatch in floating point or integer comparison!!\n"
         printf "(%d) - %s \n", ln_id[id-1], saved_line[id-1]
         printf "(%d) + %s \n\n", ln_id[id-1], out_line[id-1]      
      }      
   }
}'

