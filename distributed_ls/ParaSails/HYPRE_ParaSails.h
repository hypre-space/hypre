/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_ParaSails.h header file.
 *
 *****************************************************************************/

#include "HYPRE_distributed_matrix_protos.h"

typedef void *HYPRE_ParaSails;

int HYPRE_ParaSailsCreate(MPI_Comm comm, HYPRE_DistributedMatrix *distmat, 
  HYPRE_ParaSails *obj);
int HYPRE_ParaSailsDestroy(HYPRE_ParaSails ps);
int HYPRE_ParaSailsSelectThresh(HYPRE_ParaSails ps, double *threshp);
int HYPRE_ParaSailsSetup(HYPRE_ParaSails ps, int sym, double thresh, 
  int nlevels, double filter);
int HYPRE_ParaSailsApply(HYPRE_ParaSails ps, double *u, double *v);
