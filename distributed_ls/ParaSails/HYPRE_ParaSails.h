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

int HYPRE_ParaSailsCreate(MPI_Comm comm, HYPRE_ParaSails *obj);
int HYPRE_ParaSailsDestroy(HYPRE_ParaSails ps);
int HYPRE_ParaSailsSetup(HYPRE_ParaSails obj,
  HYPRE_DistributedMatrix *distmat, int sym, double thresh, int nlevels,
  double filter, double loadbal, int logging);
int HYPRE_ParaSailsSetupPattern(HYPRE_ParaSails obj,
  HYPRE_DistributedMatrix *distmat, int sym, double thresh, int nlevels, 
  int logging);
int HYPRE_ParaSailsSetupValues(HYPRE_ParaSails obj,
  HYPRE_DistributedMatrix *distmat, double filter, double loadbal, 
  int logging);
int HYPRE_ParaSailsApply(HYPRE_ParaSails ps, double *u, double *v);
