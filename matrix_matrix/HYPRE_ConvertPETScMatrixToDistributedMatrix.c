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
 * Routine for building a DistributedMatrix from a MPIAIJ Mat, i.e. PETSc matrix
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

#ifdef PETSC_AVAILABLE

/* Matrix structure from PETSc */
#include "sles.h"
/*--------------------------------------------------------------------------
 * HYPRE_ConvertPETScMatrixToDistributedMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_ConvertPETScMatrixToDistributedMatrix( 
                 Mat PETSc_matrix,
                 HYPRE_DistributedMatrix *DistributedMatrix )
{
   int ierr;
   MPI_Comm MPI_Comm;
   int M, N;



   if (!PETSc_matrix) return(-1);

   ierr = PetscObjectGetComm( (PetscObject) PETSc_matrix, &MPI_Comm); CHKERRA(ierr);

   *DistributedMatrix = HYPRE_NewDistributedMatrix( MPI_Comm );

   ierr = HYPRE_SetDistributedMatrixLocalStorageType( *DistributedMatrix,
                                                     HYPRE_PETSC );
   if(ierr) return(ierr);

   ierr = HYPRE_InitializeDistributedMatrix( *DistributedMatrix );
   if(ierr) return(ierr);

   ierr = HYPRE_SetDistributedMatrixLocalStorage( *DistributedMatrix, PETSc_matrix );
   if(ierr) return(ierr);
   /* Note that this is kind of cheating, since the Mat structure contains more
      than local information... the alternative is to extract the global info
      from the Mat and put it into DistributedMatrixAuxiliaryStorage. However,
      the latter is really a "just in case" option, and so if we don't *have*
      to use it, we won't.*/

   ierr = MatGetSize( PETSc_matrix, &M, &N); if(ierr) return(ierr);
   ierr = HYPRE_SetDistributedMatrixDims( *DistributedMatrix, M, N);

   ierr = HYPRE_AssembleDistributedMatrix( *DistributedMatrix );
   if(ierr) return(ierr);

   return(0);
}

#endif
