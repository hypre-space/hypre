/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




/* shell sort adopted from Edmond Chow */

#include "shellSort_dh.h"

#undef __FUNC__
#define __FUNC__ "shellSort_int"
void shellSort_int(const HYPRE_Int n, HYPRE_Int *x)
{
  START_FUNC_DH
  HYPRE_Int m, max, j, k, itemp;

  m = n/2;
  while (m > 0) {
    max = n - m;
    for (j=0; j<max; j++) {
      for (k=j; k>=0; k-=m) {
        if (x[k+m] >= x[k]) break;
        itemp = x[k+m];
        x[k+m] = x[k];
        x[k] = itemp;
      }
    }
    m = m/2;
  }
  END_FUNC_DH
}

#undef __FUNC__
#define __FUNC__ "shellSort_float"
void shellSort_float(const HYPRE_Int n, double *x)
{
  START_FUNC_DH
  HYPRE_Int m, max, j, k;
  double itemp;

  m = n/2;
  while (m > 0) {
    max = n - m;
    for (j=0; j<max; j++) {
      for (k=j; k>=0; k-=m) {
        if (x[k+m] >= x[k]) break;
        itemp = x[k+m];
        x[k+m] = x[k];
        x[k] = itemp;
      }
    }
    m = m/2;
  }
  END_FUNC_DH
}


#if 0
#undef __FUNC__
#define __FUNC__ "shellSort_int_float"
void shellSort_int_float(HYPRE_Int n, HYPRE_Int *x, VAL_DH *xVals)
{
  START_FUNC_DH
  HYPRE_Int m, max, j, k, itemp;
  VAL_DH atemp;

  m = n/2;
  while (m > 0) {
    max = n - m;
    for (j=0; j<max; j++) {
      for (k=j; k>=0; k-=m) {
        if (x[k+m] >= x[k]) break;
        itemp = x[k+m];
        atemp = xVals[k+m];
        x[k+m] = x[k];
        /* xVals[k+m] = xVals[k]; */
        x[k] = itemp;
        xVals[k] = atemp;
      }
    }
    m = m/2;
  }
  END_FUNC_DH
}
#endif
