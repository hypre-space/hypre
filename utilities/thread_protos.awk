#BHEADER***********************************************************************
# (c) 1998   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision$
#EHEADER***********************************************************************

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


