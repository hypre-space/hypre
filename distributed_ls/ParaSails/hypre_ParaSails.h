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
 * hypre_ParaSails.h header file.
 *
 *****************************************************************************/

#include "HYPRE_distributed_matrix_protos.h"

typedef void *hypre_ParaSails;

int hypre_ParaSailsCreate(MPI_Comm comm, hypre_ParaSails *obj);
int hypre_ParaSailsDestroy(hypre_ParaSails ps);
int hypre_ParaSailsSetup(hypre_ParaSails obj,
  HYPRE_DistributedMatrix *distmat, int sym, double thresh, int nlevels,
  double filter, double loadbal, int logging);
int hypre_ParaSailsSetupPattern(hypre_ParaSails obj,
  HYPRE_DistributedMatrix *distmat, int sym, double thresh, int nlevels, 
  int logging);
int hypre_ParaSailsSetupValues(hypre_ParaSails obj,
  HYPRE_DistributedMatrix *distmat, double filter, double loadbal, 
  int logging);
int hypre_ParaSailsApply(hypre_ParaSails ps, double *u, double *v);
int hypre_ParaSailsApplyTrans(hypre_ParaSails ps, double *u, double *v);
