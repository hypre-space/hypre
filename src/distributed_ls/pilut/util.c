/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
 * util.c
 *
 * This function contains various utility routines
 *
 * Started 9/28/95
 * George
 *
 * $Id$
 */

#include "ilu.h"
#include "DistributedMatrixPilutSolver.h"

/*************************************************************************
* This function finds the minimum value in the array removes it and
* returns it. It decreases the size of the array.
**************************************************************************/
HYPRE_Int hypre_ExtractMinLR( hypre_PilutSolverGlobals *globals )
{
  HYPRE_Int i, j=0 ;

  for (i=1; i<lastlr; i++) {
    if (hypre_lr[i] < hypre_lr[j])
      j = i;
  }
  i = hypre_lr[j];

  /* Remove it */
  lastlr-- ;
  if (j < lastlr)
    hypre_lr[j] = hypre_lr[lastlr];

  return i;
}


/*************************************************************************
* This function sums the entries in an array
**************************************************************************/
HYPRE_Int hypre_sasum(HYPRE_Int n, HYPRE_Int *x)
{
  HYPRE_Int sum = 0;
  HYPRE_Int i;

  for (i=0; i<n; i++)
    sum += x[i];

  return sum;
}


/*************************************************************************
* This function compares 2 ints for sorting in inc order
**************************************************************************/
static HYPRE_Int incshort(const void *v1, const void *v2)
{
  return (*((HYPRE_Int *)v1) - *((HYPRE_Int *)v2));
}

/*************************************************************************
* This function compares 2 ints for sorting in dec order
**************************************************************************/
static HYPRE_Int decshort(const void *v1, const void *v2)
{
  return (*((HYPRE_Int *)v2) - *((HYPRE_Int *)v1));
}

/*************************************************************************
* These functions sorts an array of XXX
**************************************************************************/
void hypre_sincsort(HYPRE_Int n, HYPRE_Int *a)
{
  hypre_tex_qsort((char *)a, (size_t)n, (size_t)sizeof(HYPRE_Int), (HYPRE_Int (*) (char*,char*))incshort);
}


void hypre_sdecsort(HYPRE_Int n, HYPRE_Int *a)
{
  hypre_tex_qsort((char *)a, (size_t)n, (size_t)sizeof(HYPRE_Int),(HYPRE_Int (*) (char*,char*)) decshort);
}
