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
 * debug.c
 *
 * This file implements some debugging utilities.
 * I use checksums to compare entire arrays easily. Note that the
 * perm and iperm arrays always have the same checksum, even
 * though they are in a different order.
 *
 * Started 7/8/97
 * Mark
 *
 */

#undef NDEBUG

#include "./DistributedMatrixPilutSolver.h"

/*************************************************************************
* This function prints a message and file/line number
**************************************************************************/
void hypre_PrintLine(char *str, hypre_PilutSolverGlobals *globals)
{
  hypre_printf("PE %d ---- %-27s (%s, %d)\n",
	 mype, str, __FILE__, __LINE__);
  fflush(stdout);
}


/*************************************************************************
* This function exits if i is not in [low, up)
**************************************************************************/
void hypre_CheckBounds(HYPRE_Int low, HYPRE_Int i ,HYPRE_Int up, hypre_PilutSolverGlobals *globals)
{
  if ((i < low)  ||  (i >= up))
    hypre_errexit("PE %d Bad bound: %d <= %d < %d (%s %d)\n", 
	    mype, low, i, up, __FILE__, __LINE__ );
}

/*************************************************************************
* This function prints a checksum for an HYPRE_Int (HYPRE_Int) array
**************************************************************************/
hypre_longint hypre_IDX_Checksum(const HYPRE_Int *v, HYPRE_Int len, const char *msg, HYPRE_Int tag,
          hypre_PilutSolverGlobals *globals)
{
  static HYPRE_Int numChk = 0;
  HYPRE_Int i;
  hypre_ulongint sum = 0;

  for (i=0; i<len; i++)
    sum += v[i] * i;

  hypre_printf("PE %d [i%3d] %15s/%3d chk: %16lx [len %4d]\n", 
	 mype, numChk, msg, tag, sum, len);
  fflush(stdout);

  numChk++;

  return sum;
}

/*************************************************************************
* This function prints a checksum for an HYPRE_Int (HYPRE_Int) array
**************************************************************************/
hypre_longint hypre_INT_Checksum(const HYPRE_Int *v, HYPRE_Int len, const char *msg, HYPRE_Int tag,
          hypre_PilutSolverGlobals *globals)
{
  static HYPRE_Int numChk = 0;
  HYPRE_Int i;
  hypre_ulongint sum = 0;

  for (i=0; i<len; i++)
    sum += v[i] * i;

  hypre_printf("PE %d [d%3d] %15s/%3d chk: %16lx [len %4d]\n",
	 mype, numChk, msg, tag, sum, len);
  fflush(stdout);

  numChk++;

  return sum;
}

/*************************************************************************
* This function prints a checksum for a float (double) array
**************************************************************************/
hypre_longint hypre_FP_Checksum(const double *v, HYPRE_Int len, const char *msg, HYPRE_Int tag,
          hypre_PilutSolverGlobals *globals)
{
  static HYPRE_Int numChk = 0;
  HYPRE_Int i;
  hypre_ulongint sum = 0;
  HYPRE_Int *vv = (HYPRE_Int*)v;

  for (i=0; i<len; i++)
    sum += vv[i] * i;

  hypre_printf("PE %d [f%3d] %15s/%3d chk: %16lx [len %4d]\n",
	 mype, numChk, msg, tag, sum, len);
  fflush(stdout);

  numChk++;

  return sum;
}

/*************************************************************************
* This function prints checksums for each array of the rmat struct
**************************************************************************/
hypre_longint hypre_RMat_Checksum(const ReduceMatType *rmat,
          hypre_PilutSolverGlobals *globals)
{
  HYPRE_Int i;
  static HYPRE_Int numChk = 0;

  /* for safety */
  if ( rmat          == NULL  ||
       rmat->rmat_rnz     == NULL  ||
       rmat->rmat_rrowlen == NULL  ||
       rmat->rmat_rcolind == NULL  ||
       rmat->rmat_rvalues == NULL ) {
    hypre_printf("PE %d [r%3d] rmat checksum -- not initializied\n",
	   mype, numChk);
    fflush(stdout);

    numChk++;
    return 0;
  }

  /* print ints */
  hypre_printf("PE %d [r%3d] rmat checksum -- ndone %d ntogo %d nlevel %d\n",
	 mype, numChk, rmat->rmat_ndone, rmat->rmat_ntogo, rmat->rmat_nlevel);
  fflush(stdout);

  /* print checksums for each array */
  hypre_IDX_Checksum(rmat->rmat_rnz,     rmat->rmat_ntogo, "rmat->rmat_rnz",     numChk,
      globals);
  hypre_IDX_Checksum(rmat->rmat_rrowlen, rmat->rmat_ntogo, "rmat->rmat_rrowlen", numChk,
      globals);

  for (i=0; i<rmat->rmat_ntogo; i++) {
    hypre_IDX_Checksum(rmat->rmat_rcolind[i], rmat->rmat_rrowlen[i], "rmat->rmat_rcolind", i,
      globals);
     hypre_FP_Checksum(rmat->rmat_rvalues[i], rmat->rmat_rrowlen[i], "rmat->rmat_rvalues", i,
      globals);
  }

  return 1;
}

/*************************************************************************
* This function prints checksums for some arrays of the LDU struct
**************************************************************************/
hypre_longint hypre_LDU_Checksum(const FactorMatType *ldu,
          hypre_PilutSolverGlobals *globals)
{
  HYPRE_Int i, j;
  hypre_ulongint lisum=0, ldsum=0, uisum=0, udsum=0, dsum=0;
  static HYPRE_Int numChk = 0;

  if (ldu->lsrowptr == NULL  ||
      ldu->lerowptr == NULL  ||
      ldu->lcolind  == NULL  ||
      ldu->lvalues  == NULL  ||
      ldu->usrowptr == NULL  ||
      ldu->uerowptr == NULL  ||
      ldu->ucolind  == NULL  ||
      ldu->uvalues  == NULL  ||
      ldu->dvalues  == NULL  ||
      ldu->nrm2s    == NULL) {
    hypre_printf("PE %d [S%3d] LDU check -- not initializied\n",
	   mype, numChk);
    fflush(stdout);
    return 0;
  }

  for (i=0; i<lnrows; i++) {
    for (j=ldu->lsrowptr[i]; j<ldu->lerowptr[i]; j++) {
      lisum += ldu->lcolind[j];
      ldsum += (hypre_longint)ldu->lvalues[j];
    }

    for (j=ldu->usrowptr[i]; j<ldu->uerowptr[i]; j++) {
      uisum += ldu->ucolind[j];
      udsum += (hypre_longint)ldu->uvalues[j];
    }

    if (ldu->usrowptr[i] < ldu->uerowptr[i])
      dsum += (hypre_longint)ldu->dvalues[i];
  }

  hypre_printf("PE %d [S%3d] LDU check [%16lx %16lx] [%16lx] [%16lx %16lx]\n",
	 mype, numChk, lisum, ldsum, dsum, uisum, udsum);
  fflush(stdout);

  hypre_FP_Checksum(ldu->nrm2s, lnrows, "2-norms", numChk,
      globals);

  return 1;
}


/*************************************************************************
* This function prints a vector on each processor 
**************************************************************************/
void hypre_PrintVector(HYPRE_Int *v, HYPRE_Int n, char *msg,
          hypre_PilutSolverGlobals *globals)
{
  HYPRE_Int i, penum;

  for (penum=0; penum<npes; penum++) {
    if (mype == penum) {
      hypre_printf("PE %d %s: ", mype, msg);

      for (i=0; i<n; i++)
        hypre_printf("%d ", v[i]);
      hypre_printf("\n");
    }
    hypre_MPI_Barrier( pilut_comm );
  }
}
