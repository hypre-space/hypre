/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
 * pblas1.c
 *
 * This file contains functions that implement various distributed
 * level 1 BLAS routines
 *
 * Started 11/28/95
 * George
 *
 * $Id$
 *
 */

#include "_hypre_blas.h"
#include "DistributedMatrixPilutSolver.h"


/*************************************************************************
* This function computes the 2 norm of a vector. The result is returned
* at all the processors
**************************************************************************/
HYPRE_Real hypre_p_dnrm2(DataDistType *ddist, HYPRE_Real *x, hypre_PilutSolverGlobals *globals)
{
  HYPRE_Int incx=1;
  HYPRE_Real sum;

  sum = hypre_dnrm2(&(ddist->ddist_lnrows), x, &incx);
  return hypre_sqrt(hypre_GlobalSESumDouble(sum*sum, pilut_comm));
}


/*************************************************************************
* This function computes the dot product of 2 vectors. 
* The result is returned at all the processors
**************************************************************************/
HYPRE_Real hypre_p_ddot(DataDistType *ddist, HYPRE_Real *x, HYPRE_Real *y,
              hypre_PilutSolverGlobals *globals)
{
  HYPRE_Int incx=1;

  return hypre_GlobalSESumDouble(hypre_ddot(&(ddist->ddist_lnrows), x, &incx, y, &incx), 
         pilut_comm );
}


/*************************************************************************
* This function performs y = alpha*x, where alpha resides on pe 0
**************************************************************************/
void hypre_p_daxy(DataDistType *ddist, HYPRE_Real alpha, HYPRE_Real *x, HYPRE_Real *y)
{
  HYPRE_Int i, local_lnrows=ddist->ddist_lnrows;

  for (i=0; i<local_lnrows; i++)
    y[i] = alpha*x[i];
}


/*************************************************************************
* This function performs y = alpha*x+y, where alpha resides on pe 0
**************************************************************************/
void hypre_p_daxpy(DataDistType *ddist, HYPRE_Real alpha, HYPRE_Real *x, HYPRE_Real *y)
{
  HYPRE_Int i, local_lnrows=ddist->ddist_lnrows;

  for (i=0; i<local_lnrows; i++)
    y[i] += alpha*x[i];
}



/*************************************************************************
* This function performs z = alpha*x+beta*y, where alpha resides on pe 0
**************************************************************************/
void hypre_p_daxbyz(DataDistType *ddist, HYPRE_Real alpha, HYPRE_Real *x, HYPRE_Real beta, 
              HYPRE_Real *y, HYPRE_Real *z)
{
  HYPRE_Int i, local_lnrows=ddist->ddist_lnrows;

  for (i=0; i<local_lnrows; i++)
    z[i] = alpha*x[i] + beta*y[i];
}

/*************************************************************************
* This function prints a vector
**************************************************************************/
HYPRE_Int hypre_p_vprintf(DataDistType *ddist, HYPRE_Real *x,
                    hypre_PilutSolverGlobals *globals )
{
  HYPRE_Int pe, i;

  for (pe=0; pe<npes; pe++) {
    if (mype == pe) {
      for (i=0; i<ddist->ddist_lnrows; i++)
        hypre_printf("%d:%f, ", ddist->ddist_rowdist[mype]+i, x[i]);
      if (pe == npes-1)
        hypre_printf("\n");
    }
    hypre_MPI_Barrier( pilut_comm );
  }
  fflush(stdout);
  hypre_MPI_Barrier( pilut_comm );

  return 0;
}
