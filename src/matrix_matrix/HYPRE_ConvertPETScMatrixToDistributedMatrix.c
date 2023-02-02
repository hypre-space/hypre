/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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

HYPRE_Int
HYPRE_ConvertPETScMatrixToDistributedMatrix(
   Mat PETSc_matrix,
   HYPRE_DistributedMatrix *DistributedMatrix )
{
   HYPRE_Int ierr;
   MPI_Comm hypre_MPI_Comm;
   HYPRE_BigInt M, N;
#ifdef HYPRE_TIMING
   HYPRE_Int           timer;
#endif



   if (!PETSc_matrix) { return (-1); }

#ifdef HYPRE_TIMING
   timer = hypre_InitializeTiming( "ConvertPETScMatrixToDistributedMatrix");
   hypre_BeginTiming( timer );
#endif


   ierr = PetscObjectGetComm( (PetscObject) PETSc_matrix, &MPI_Comm); CHKERRA(ierr);

   ierr = HYPRE_DistributedMatrixCreate( MPI_Comm, DistributedMatrix );
   /* if(ierr) return(ierr); */

   ierr = HYPRE_DistributedMatrixSetLocalStorageType( *DistributedMatrix,
                                                      HYPRE_PETSC );
   /* if(ierr) return(ierr);*/

   ierr = HYPRE_DistributedMatrixInitialize( *DistributedMatrix );
   /* if(ierr) return(ierr);*/

   ierr = HYPRE_DistributedMatrixSetLocalStorage( *DistributedMatrix, PETSc_matrix );
   /* if(ierr) return(ierr); */
   /* Note that this is kind of cheating, since the Mat structure contains more
      than local information... the alternative is to extract the global info
      from the Mat and put it into DistributedMatrixAuxiliaryStorage. However,
      the latter is really a "just in case" option, and so if we don't *have*
      to use it, we won't.*/

   ierr = MatGetSize( PETSc_matrix, &M, &N);
   if (ierr) { return (ierr); }
   ierr = HYPRE_DistributedMatrixSetDims( *DistributedMatrix, M, N);

   ierr = HYPRE_DistributedMatrixAssemble( *DistributedMatrix );
   /* if(ierr) return(ierr);*/

#ifdef HYPRE_TIMING
   hypre_EndTiming( timer );
   /* hypre_FinalizeTiming( timer ); */
#endif

   return (0);
}

#endif
