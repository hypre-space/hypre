/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/




/*
 * pblas1.c
 *
 * This file contains functions that implement various distributed
 * level 1 BLAS routines
 *
 * Started 11/28/95
 * George
 *
 * $Id: pblas1.c,v 2.7 2010/12/20 19:27:34 falgout Exp $
 *
 */

#include "ilu.h"
#include "DistributedMatrixPilutSolver.h"


/*************************************************************************
* This function computes the 2 norm of a vector. The result is returned
* at all the processors
**************************************************************************/
double hypre_p_dnrm2(DataDistType *ddist, double *x, hypre_PilutSolverGlobals *globals)
{
  HYPRE_Int incx=1;
  double sum;

  sum = SNRM2(&(ddist->ddist_lnrows), x, &incx);
  return sqrt(hypre_GlobalSESumDouble(sum*sum, pilut_comm));
}


/*************************************************************************
* This function computes the dot product of 2 vectors. 
* The result is returned at all the processors
**************************************************************************/
double hypre_p_ddot(DataDistType *ddist, double *x, double *y,
              hypre_PilutSolverGlobals *globals)
{
  HYPRE_Int incx=1;

  return hypre_GlobalSESumDouble(SDOT(&(ddist->ddist_lnrows), x, &incx, y, &incx), 
         pilut_comm );
}


/*************************************************************************
* This function performs y = alpha*x, where alpha resides on pe 0
**************************************************************************/
void hypre_p_daxy(DataDistType *ddist, double alpha, double *x, double *y)
{
  HYPRE_Int i, local_lnrows=ddist->ddist_lnrows;

  for (i=0; i<local_lnrows; i++)
    y[i] = alpha*x[i];
}


/*************************************************************************
* This function performs y = alpha*x+y, where alpha resides on pe 0
**************************************************************************/
void hypre_p_daxpy(DataDistType *ddist, double alpha, double *x, double *y)
{
  HYPRE_Int i, local_lnrows=ddist->ddist_lnrows;

  for (i=0; i<local_lnrows; i++)
    y[i] += alpha*x[i];
}



/*************************************************************************
* This function performs z = alpha*x+beta*y, where alpha resides on pe 0
**************************************************************************/
void hypre_p_daxbyz(DataDistType *ddist, double alpha, double *x, double beta, 
              double *y, double *z)
{
  HYPRE_Int i, local_lnrows=ddist->ddist_lnrows;

  for (i=0; i<local_lnrows; i++)
    z[i] = alpha*x[i] + beta*y[i];
}

/*************************************************************************
* This function prints a vector
**************************************************************************/
HYPRE_Int hypre_p_vprintf(DataDistType *ddist, double *x,
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
