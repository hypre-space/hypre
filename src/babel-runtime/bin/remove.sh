#! /bin/sh
## File:        remove.sh
## Package:     Babel binary
## Copyright:   (c) 2000-2001 The Regents of the University of California
## Revision:    $Revision: 1.10 $
## Modified:    $Date: 2006/08/29 22:29:10 $
## Description: utility to remove files for babel build
##

srcdir=$1
shift

if test "X$srcdir" != "X."; then
  echo rm -f $*
  rm -f $*
fi

exit 0
