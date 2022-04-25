# Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
# HYPRE Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

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


