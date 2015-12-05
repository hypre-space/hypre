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
# $Revision: 1.5 $
#EHEADER**********************************************************************






#=============================================================================
# FORTRAN_EXAMPLES: compare fortran_examples.out.* with ex*.base files
#=============================================================================

tail -21 fortran_examples.out.1 > fortran_examples.test.tmp1
head fortran_examples.test.tmp1 > fortran_examples.test1

tail -21 ex1.base > fortran_examples.base.tmp1
head fortran_examples.base.tmp1 > fortran_examples.base1
diff fortran_examples.base1 fortran_examples.test1 >&2


tail -21 fortran_examples.out.3 > fortran_examples.test.tmp3
head fortran_examples.test.tmp3 > fortran_examples.test3

tail -21 ex3.base > fortran_examples.base.tmp3
head fortran_examples.base.tmp3 > fortran_examples.base3
diff fortran_examples.base3 fortran_examples.test3 >&2


tail -21 fortran_examples.out.5 > fortran_examples.test.tmp5
head fortran_examples.test.tmp5 > fortran_examples.test5

tail -21 ex5.base > fortran_examples.base.tmp5
head fortran_examples.base.tmp5 > fortran_examples.base5
diff fortran_examples.base5 fortran_examples.test5 >&2


tail -21 fortran_examples.out.6 > fortran_examples.test.tmp6
head fortran_examples.test.tmp6 > fortran_examples.test6

tail -21 ex6.base > fortran_examples.base.tmp6
head fortran_examples.base.tmp6 > fortran_examples.base6
diff fortran_examples.base6 fortran_examples.test6 >&2


tail -21 fortran_examples.out.7 > fortran_examples.test.tmp7
head fortran_examples.test.tmp7 > fortran_examples.test7

tail -21 ex7.base > fortran_examples.base.tmp7
head fortran_examples.base.tmp7 > fortran_examples.base7
diff fortran_examples.base7 fortran_examples.test7 >&2

rm -f fortran_examples.base* fortran_examples.test*
