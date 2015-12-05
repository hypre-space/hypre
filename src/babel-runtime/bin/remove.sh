#! /bin/sh
## File:        remove.sh
## Package:     Babel binary
## Copyright:   (c) 2000-2001 The Regents of the University of California
## Release:     $Name: V1-9-0b $
## Revision:    $Revision: 1.4 $
## Modified:    $Date: 2003/04/07 21:44:10 $
## Description: utility to remove files for babel build
##

srcdir=$1
shift

if test "X$srcdir" != "X."; then
  echo rm -f $*
  rm -f $*
fi

exit 0
