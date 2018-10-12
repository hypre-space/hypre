#!/bin/sh
#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision$
#EHEADER**********************************************************************
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

awk -v filename="$SNAME" 'BEGIN{
   FS=" ";
   key = 0;
   # Read saved file data into array
   while (getline < filename)
   {
      if(NF > 0 && substr($1,1,1) !~ /#/)
      {
         saved_line[key]=$0;
         saved_array[++key]=$NF;
      }
   }
   close(filename);

   # read out file and compare
   filename="'"$FNAME"'"
   rtol = "'"$RTOL"'" + 0. # adding zero is necessary to convert from string to number
   atol = "'"$ATOL"'" + 0. # adding zero is necessary to convert from string to number
   key=0;
   ln=0;
   pass=1;

   while (getline < filename)
   {
      ln++;
      if(NF > 0 && substr($1,1,1) !~ /#/)
      {
         # get corresponding value in saved array
         val = saved_array[++key];

         # floating point field comparison
         if($NF != int($NF))
         {
            err = val - $NF;
            # get absolute value of err and val
            err = err < 0 ? -err : err;
            val = val < 0 ? -val : val;
            # abs err <= atol or rel err <= rtol
            if(err <= atol || err <= rtol*val)
            {
               #print "PASSED"
            }
            else
            {
               pass=0;
               printf "(%d) - %s\n", ln, saved_line[key-1]
               printf "(%d) + %s      (err %.2e)\n\n", ln, $0, err
               #printf "(%d) + %s <-- %s, err %.2e\n", ln, $0, val, err
            }
         }
         else # integer comparison
         {
            tau = val - $NF;
            # get absolute value of tau
            tau = tau < 0 ? -tau : tau;
            # get ceiling of rtol*val (= max allowed change)
            gamma = int(1.0 + rtol*val);
            if(tau <= gamma)
            {
               #print "PASSED"
            }
            else
            {
               pass=0;
               printf "(%d) %s <-- %s, err %d\n", ln, $0, val, tau
            }
         }
      }
   }
}'

#if [ "x$PASSFAIL" != "x" ];
#then
#    echo $PASSFAIL
#     diff -U3 -bI"time" $SNAME $FNAME >&2
#fi
