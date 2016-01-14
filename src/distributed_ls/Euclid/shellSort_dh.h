/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#ifndef SUPPORT_DH
#define SUPPORT_DH

/* #include "euclid_common.h" */

extern void shellSort_int(const HYPRE_Int n, HYPRE_Int *x);
extern void shellSort_float(HYPRE_Int n, HYPRE_Real *v);

/*
extern void shellSort_int_int(const HYPRE_Int n, HYPRE_Int *x, HYPRE_Int *y);
extern void shellSort_int_float(HYPRE_Int n, HYPRE_Int *x, HYPRE_Real *v);
extern void shellSort_int_int_float(HYPRE_Int n, HYPRE_Int *x, HYPRE_Int *y, HYPRE_Real *v);
*/

#endif
