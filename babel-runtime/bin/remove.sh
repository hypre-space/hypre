#! /bin/sh
## File:        remove.sh
## Package:     Babel binary
## Copyright:   (c) 2000-2001 The Regents of the University of California
## Revision:    $Revision$
## Modified:    $Date$
## Description: utility to remove files for babel build
##

srcdir=$1
shift

if test "X$srcdir" != "X."; then
  echo rm -f $*
  rm -f $*
fi

exit 0
