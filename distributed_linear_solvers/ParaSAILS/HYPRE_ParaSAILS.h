/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "mpi.h"
#include "HYPRE_distributed_matrix_types.h"
#include "HYPRE_parcsr_mv.h"

#ifndef _HYPRE_PARASAILS_H
#define _HYPRE_PARASAILS_H

#ifdef __cplusplus
extern "C" {
#endif


typedef void *HYPRE_ParaSAILS;

/* pruning algorithms */
enum
{
    PRUNE_NO,
    PRUNE_THRESH,
    PRUNE_LFIL
};


#define        P(s) s
 
HYPRE_ParaSAILS HYPRE_ParaSAILS_New P((MPI_Comm comm, 
  HYPRE_DistributedMatrix matrix));

int HYPRE_ParaSAILS_Free P((HYPRE_ParaSAILS in_ptr));

int HYPRE_ParaSAILS_Init P((HYPRE_ParaSAILS solver));

int HYPRE_ParaSAILS_SetMat P((HYPRE_ParaSAILS in_ptr, 
  HYPRE_DistributedMatrix matrix));

HYPRE_DistributedMatrix HYPRE_ParaSAILS_GetMat P((HYPRE_ParaSAILS in_ptr));

int HYPRE_ParaSAILS_Setup P((HYPRE_ParaSAILS in_ptr));

int HYPRE_ParaSAILS_Solve P((HYPRE_ParaSAILS in_ptr, 
  HYPRE_ParVector x, HYPRE_ParVector b));

#undef P

#ifdef __cplusplus
}
#endif

#endif /* _HYPRE_PARASAILS_H */
