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
# $Revision: 1.11 $
#EHEADER**********************************************************************






#=============================================================================
# EXAMPLES: Compare ex*.base files with default.out.* files from current runs
#           differences (except for timings) indicate errors
#=============================================================================

diff -U3 -bI"time" ex1.base default.out.1 >&2

diff -U3 -bI"time" ex2.base default.out.2 >&2

diff -U3 -bI"time" ex3.base default.out.3 >&2

diff -U3 -bI"time" ex4.base default.out.4 >&2

diff -U3 -bI"time" ex5.base default.out.5 >&2

diff -U3 -bI"time" ex5f.base default.out.5f >&2

diff -U3 -bI"time" ex6.base default.out.6 >&2

diff -U3 -bI"time" ex7.base default.out.7 >&2

diff -U3 -bI"time" ex8.base default.out.8 >&2

diff -U3 -bI"time" ex9.base default.out.9 >&2

diff -U3 -bI"time" ex10.base default.out.10 >&2

diff -U3 -bI"time" ex11.base default.out.11 >&2

diff -U3 -bI"time" ex12.base default.out.12 >&2

diff -U3 -bI"time" ex12f.base default.out.12f >&2

diff -U3 -bI"time" ex13.base default.out.13 >&2

diff -U3 -bI"time" ex14.base default.out.14 >&2

diff -U3 -bI"time" ex15.base default.out.15 >&2
