/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/




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

HYPRE_Int 
HYPRE_ConvertParCSRMatrixToDistributedMatrix( 
                 HYPRE_ParCSRMatrix parcsr_matrix,
                 HYPRE_DistributedMatrix *DistributedMatrix )
{
   HYPRE_Int ierr;
   MPI_Comm comm;
   HYPRE_Int M, N;

#ifdef HYPRE_TIMING
   HYPRE_Int           timer;
   timer = hypre_InitializeTiming( "ConvertParCSRMatrisToDistributedMatrix");
   hypre_BeginTiming( timer );
#endif


   if (!parcsr_matrix) return(-1);

   ierr = HYPRE_ParCSRMatrixGetComm( parcsr_matrix, &comm);

   ierr = HYPRE_DistributedMatrixCreate( comm, DistributedMatrix );

   ierr = HYPRE_DistributedMatrixSetLocalStorageType( *DistributedMatrix,
                                                     HYPRE_PARCSR );
   /* if(ierr) return(ierr); */

   ierr = HYPRE_DistributedMatrixInitialize( *DistributedMatrix );
   /* if(ierr) return(ierr); */

   ierr = HYPRE_DistributedMatrixSetLocalStorage( *DistributedMatrix, parcsr_matrix );
   /* if(ierr) return(ierr);*/
   

   ierr = HYPRE_ParCSRMatrixGetDims( parcsr_matrix, &M, &N); /* if(ierr) return(ierr); */
   ierr = HYPRE_DistributedMatrixSetDims( *DistributedMatrix, M, N);

   ierr = HYPRE_DistributedMatrixAssemble( *DistributedMatrix );
   /* if(ierr) return(ierr); */

#ifdef HYPRE_TIMING
   hypre_EndTiming( timer );
   /* hypre_FinalizeTiming( timer ); */
#endif

   return(0);
}

