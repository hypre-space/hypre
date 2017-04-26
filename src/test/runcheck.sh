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
# ./runcheck.sh fname.out fname.saved 1.0e-6

FNAME=$1
SNAME=$2
CONVTOL=$3

if [ x$CONVTOL = "x" ]; 
then
    CONVTOL=0.0
fi
#echo "runcheck tol = $CONVTOL"

PASSFAIL=$(awk -v filename="$SNAME" 'BEGIN{{FS=" "}

   key = 0;
   
   # Read saved file data into array
   while (getline < filename)
   {
      if(NF > 0 && substr($1,1,1) !~ /#/)
      {            
         saved_array[++key]=$NF;   
      }
   }
   close(filename);
  
   # read out file and compare
   filename="'"$FNAME"'"
   tol = "'"$CONVTOL"'" + 0. # adding zero is necessary to convert from string to number
   key=0;
   pass=1;
   while (getline < filename)
   {
      if(NF > 0 && substr($1,1,1) !~ /#/)
      {
         # get corresponding value in saved array
         val = saved_array[++key];
         
         # floating point field comparison         
         if($NF != int($NF))
         {
            tau = (val - $NF)/val;
            # get absolute value of tau
            tau = tau < 0 ? -tau : tau;
            if(tau < tol)
               #print "PASSED"
               continue;
            else
               pass=0;
               #print "FAILED"
         }
         else # integer comparison
         {
            tau = val - $NF;
            # get absolute value of tau
            tau = tau < 0 ? -tau : tau;
            # get ceiling of tol*val (= max allowed change)
            gamma = int(1.0 + tol*val);
            if(tau <= gamma)
               #print "PASSED"
               continue;
            else
               pass=0;
               #print "FAILED"           
         }
      }
   }
   if(pass != 1)
      print "FAILED"
}')

if [ x$PASSFAIL != "x" ]; 
then
#    echo $PASSFAIL
     diff -U3 -bI"time" $SNAME $FNAME >&2
fi
