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

#include <HYPRE_config.h>

#include "general.h"

#include "HYPRE.h"
#include "HYPRE_utilities.h"

/* Prototypes for DistributedMatrix */
#include "HYPRE_distributed_matrix_types.h"
#include "HYPRE_distributed_matrix_protos.h"

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

#ifdef HYPRE_TIMING
   int           timer;
   timer = hypre_InitializeTiming( "ConvertParCSRMatrisToDistributedMatrix");
   hypre_BeginTiming( timer );
#endif


   if (!parcsr_matrix) return(-1);

   ierr = HYPRE_ParCSRMatrixGetComm( parcsr_matrix, &comm);

   ierr = HYPRE_DistributedMatrixCreate( comm, DistributedMatrix );

   ierr = HYPRE_DistributedMatrixSetLocalStorageType( *DistributedMatrix,
                                                     HYPRE_PARCSR );
   if(ierr) return(ierr);

   ierr = HYPRE_DistributedMatrixInitialize( *DistributedMatrix );
   if(ierr) return(ierr);

   ierr = HYPRE_DistributedMatrixSetLocalStorage( *DistributedMatrix, parcsr_matrix );
   if(ierr) return(ierr);
   

   ierr = HYPRE_ParCSRMatrixGetDims( parcsr_matrix, &M, &N); if(ierr) return(ierr);
   ierr = HYPRE_DistributedMatrixSetDims( *DistributedMatrix, M, N);

   ierr = HYPRE_DistributedMatrixAssemble( *DistributedMatrix );
   if(ierr) return(ierr);

#ifdef HYPRE_TIMING
   hypre_EndTiming( timer );
   /* hypre_FinalizeTiming( timer ); */
#endif

   return(0);
}

