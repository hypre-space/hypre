/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Routine for building a DistributedMatrix from a ParCSRMatrix
 *
 *****************************************************************************/

#ifdef HYPRE_DEBUG
#include <gmalloc.h>
#endif

#include <HYPRE_config.h>

#include "_hypre_utilities.h"
#include "HYPRE.h"

/* Prototypes for DistributedMatrix */
#include "HYPRE_distributed_matrix_types.h"
#include "HYPRE_distributed_matrix_protos.h"

/* Matrix prototypes for ParCSR */
#include "HYPRE_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * HYPRE_ConvertParCSRMatrixToDistributedMatrix
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ConvertParCSRMatrixToDistributedMatrix(
   HYPRE_ParCSRMatrix parcsr_matrix,
   HYPRE_DistributedMatrix *DistributedMatrix )
{
   MPI_Comm comm;
   HYPRE_BigInt M, N;

#ifdef HYPRE_TIMING
   HYPRE_Int           timer;
   timer = hypre_InitializeTiming( "ConvertParCSRMatrisToDistributedMatrix");
   hypre_BeginTiming( timer );
#endif


   if (!parcsr_matrix)
   {
      hypre_error(HYPRE_ERROR_ARG);
      return hypre_error_flag;
   }

   HYPRE_ParCSRMatrixGetComm( parcsr_matrix, &comm);

   HYPRE_DistributedMatrixCreate( comm, DistributedMatrix );

   HYPRE_DistributedMatrixSetLocalStorageType( *DistributedMatrix, HYPRE_PARCSR );

   HYPRE_DistributedMatrixInitialize( *DistributedMatrix );

   HYPRE_DistributedMatrixSetLocalStorage( *DistributedMatrix, parcsr_matrix );


   HYPRE_ParCSRMatrixGetDims( parcsr_matrix, &M, &N);
   HYPRE_DistributedMatrixSetDims( *DistributedMatrix, M, N);

   HYPRE_DistributedMatrixAssemble( *DistributedMatrix );

#ifdef HYPRE_TIMING
   hypre_EndTiming( timer );
   /* hypre_FinalizeTiming( timer ); */
#endif

   return hypre_error_flag;
}

