/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Routine for building a DistributedMatrix from a ParCSRMatrix
 *
 *****************************************************************************/

#ifdef HYPRE_DEBUG
#include <gmalloc.h>
#endif

#include "general.h"

#include "../HYPRE.h"
#include "../utilities/HYPRE_utilities.h"

/* Prototypes for DistributedMatrix */
#include "../distributed_matrix/HYPRE_distributed_matrix_types.h"
#include "../distributed_matrix/HYPRE_distributed_matrix_protos.h"

/* Matrix prototypes for ParCSR */
#include "HYPRE_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * HYPRE_ConvertParCSRMatrixToDistributedMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_ConvertParCSRMatrixToDistributedMatrix( 
                 HYPRE_ParCSRMatrix parcsr_matrix,
                 HYPRE_DistributedMatrix *DistributedMatrix )
{
   int ierr;
   MPI_Comm comm;
   int M, N;



   if (!parcsr_matrix) return(-1);

   ierr = HYPRE_GetCommParCSR( parcsr_matrix, &comm);

   *DistributedMatrix = HYPRE_NewDistributedMatrix( comm );

   ierr = HYPRE_SetDistributedMatrixLocalStorageType( *DistributedMatrix,
                                                     HYPRE_PARCSR );
   if(ierr) return(ierr);

   ierr = HYPRE_InitializeDistributedMatrix( *DistributedMatrix );
   if(ierr) return(ierr);

   ierr = HYPRE_SetDistributedMatrixLocalStorage( *DistributedMatrix, parcsr_matrix );
   if(ierr) return(ierr);
   

   ierr = HYPRE_GetDimsParCSR( parcsr_matrix, &M, &N); if(ierr) return(ierr);
   ierr = HYPRE_SetDistributedMatrixDims( *DistributedMatrix, M, N);

   ierr = HYPRE_AssembleDistributedMatrix( *DistributedMatrix );
   if(ierr) return(ierr);

   return(0);
}

