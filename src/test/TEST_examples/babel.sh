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
# $Revision: 1.6 $
#EHEADER**********************************************************************






#=============================================================================
# EXAMPLES: Compare ex*.base files with babel.out.* files from current runs
#           differences (except for timings) indicate errors
#=============================================================================


tail -21 babel.out.5b > babel.test.tmp
head babel.test.tmp > babel.test

tail -21 ex5b.base > babel.base.tmp
head babel.base.tmp > babel.base

diff babel.base babel.test >&2

diff ex5b77.base babel.out.5b77 >&2

rm -f babel.test.tmp babel.test babel.base.tmp babel.base
