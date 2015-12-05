#BHEADER**********************************************************************
# Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# This file is part of HYPRE.  See file COPYRIGHT for details.
#
# HYPRE is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# $Revision: 2.4 $
#EHEADER**********************************************************************




#===========================================================================
# To use, do:
# 
# /usr/xpg4/bin/awk -f {this file} < {input file} > {output file}
#
#===========================================================================

/ P\(\(/ {
  ####################################################
  # parse prototype and define various variables
  ####################################################

  split($0, b, "[\ \t]*P\(\([\ \t]*");
  routine_string = b[1];

  n = split(routine_string, a, "[^A-Za-z_0-9]");
  routine = a[n];
  routine_push = routine"Push";

  print "#define "routine" "routine_push;
}


