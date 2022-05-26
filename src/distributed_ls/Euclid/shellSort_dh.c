/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_Euclid.h"

/* shell sort adopted from Edmond Chow */

/* #include "shellSort_dh.h" */

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
void shellSort_float(const HYPRE_Int n, HYPRE_Real *x)
{
  START_FUNC_DH
  HYPRE_Int m, max, j, k;
  HYPRE_Real itemp;

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
