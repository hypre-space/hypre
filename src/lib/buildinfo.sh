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

#=============================================================================
# This script generates build info for hypre
#=============================================================================

echo '#define HYPRE_BUILD_INFO "\'

(

echo "Date:"
date

echo
echo "Machine:"
uname -a

echo
echo "C compile:"
make -f buildmakefile buildtestC

echo
echo "C++ compile:"
make -f buildmakefile buildtestCXX

) | grep -v "Entering directory" | grep -v "Leaving directory" | sed 's/$/\\n\\/'

echo '"'

