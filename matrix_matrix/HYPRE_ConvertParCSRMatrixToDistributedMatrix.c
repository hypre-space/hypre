/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
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

