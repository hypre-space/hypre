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
  printf("PE %d ---- %-27s (%s, %d)\n",
	 mype, str, __FILE__, __LINE__);
  fflush(0);
}


/*************************************************************************
* This function exits if i is not in [low, up)
**************************************************************************/
void hypre_CheckBounds(int low, int i ,int up, hypre_PilutSolverGlobals *globals)
{
  if ((i < low)  ||  (i >= up))
    hypre_errexit("PE %d Bad bound: %d <= %d < %d (%s %d)\n", 
	    mype, low, i, up, __FILE__, __LINE__ );
}

/*************************************************************************
* This function prints a checksum for an int (int) array
**************************************************************************/
long hypre_IDX_Checksum(const int *v, int len, const char *msg, int tag,
          hypre_PilutSolverGlobals *globals)
{
  static int numChk = 0;
  int i;
  long sum = 0;

  for (i=0; i<len; i++)
    sum += v[i] * i;

  printf("PE %d [i%3d] %15s/%3d chk: %16x [len %4d]\n", 
	 mype, numChk, msg, tag, sum, len);
  fflush(0);

  numChk++;

  return sum;
}

/*************************************************************************
* This function prints a checksum for an int (int) array
**************************************************************************/
long hypre_INT_Checksum(const int *v, int len, const char *msg, int tag,
          hypre_PilutSolverGlobals *globals)
{
  static int numChk = 0;
  int i;
  long sum = 0;

  for (i=0; i<len; i++)
    sum += v[i] * i;

  printf("PE %d [d%3d] %15s/%3d chk: %16x [len %4d]\n",
	 mype, numChk, msg, tag, sum, len);
  fflush(0);

  numChk++;

  return sum;
}

/*************************************************************************
* This function prints a checksum for a float (double) array
**************************************************************************/
long hypre_FP_Checksum(const double *v, int len, const char *msg, int tag,
          hypre_PilutSolverGlobals *globals)
{
  static int numChk = 0;
  int i;
  long sum = 0;
  int *vv = (int*)v;

  for (i=0; i<len; i++)
    sum += vv[i] * i;

  printf("PE %d [f%3d] %15s/%3d chk: %16x [len %4d]\n",
	 mype, numChk, msg, tag, sum, len);
  fflush(0);

  numChk++;

  return sum;
}

/*************************************************************************
* This function prints checksums for each array of the rmat struct
**************************************************************************/
long hypre_RMat_Checksum(const ReduceMatType *rmat,
          hypre_PilutSolverGlobals *globals)
{
  int i;
  static int numChk = 0;

  /* for safety */
  if ( rmat          == NULL  ||
       rmat->rmat_rnz     == NULL  ||
       rmat->rmat_rrowlen == NULL  ||
       rmat->rmat_rcolind == NULL  ||
       rmat->rmat_rvalues == NULL ) {
    printf("PE %d [r%3d] rmat checksum -- not initializied\n",
	   mype, numChk);
    fflush(0);

    numChk++;
    return 0;
  }

  /* print ints */
  printf("PE %d [r%3d] rmat checksum -- ndone %d ntogo %d nlevel %d\n",
	 mype, numChk, rmat->rmat_ndone, rmat->rmat_ntogo, rmat->rmat_nlevel);
  fflush(0);

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
long hypre_LDU_Checksum(const FactorMatType *ldu,
          hypre_PilutSolverGlobals *globals)
{
  int i, j;
  long lisum=0, ldsum=0, uisum=0, udsum=0, dsum=0, nsum=0;
  static int numChk = 0;

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
    printf("PE %d [S%3d] LDU check -- not initializied\n",
	   mype, numChk);
    fflush(0);
    return 0;
  }

  for (i=0; i<lnrows; i++) {
    for (j=ldu->lsrowptr[i]; j<ldu->lerowptr[i]; j++) {
      lisum += ldu->lcolind[j];
      ldsum += (long)ldu->lvalues[j];
    }

    for (j=ldu->usrowptr[i]; j<ldu->uerowptr[i]; j++) {
      uisum += ldu->ucolind[j];
      udsum += (long)ldu->uvalues[j];
    }

    if (ldu->usrowptr[i] < ldu->uerowptr[i])
      dsum += (long)ldu->dvalues[i];
  }

  printf("PE %d [S%3d] LDU check [%16x %16x] [%16x] [%16x %16x]\n",
	 mype, numChk, lisum, ldsum, dsum, uisum, udsum);
  fflush(0);

  hypre_FP_Checksum(ldu->nrm2s, lnrows, "2-norms", numChk,
      globals);

  return 1;
}


/*************************************************************************
* This function prints a vector on each processor 
**************************************************************************/
void hypre_PrintVector(int *v, int n, char *msg,
          hypre_PilutSolverGlobals *globals)
{
  int i, penum;

  for (penum=0; penum<npes; penum++) {
    if (mype == penum) {
      printf("PE %d %s: ", mype, msg);

      for (i=0; i<n; i++)
        printf("%d ", v[i]);
      printf("\n");
    }
    MPI_Barrier( pilut_comm );
  }
}
