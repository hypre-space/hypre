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
#ifdef HYPRE_TIMING
   int           timer;
#endif



   if (!PETSc_matrix) return(-1);

#ifdef HYPRE_TIMING
   timer = hypre_InitializeTiming( "ConvertPETScMatrixToDistributedMatrix");
   hypre_BeginTiming( timer );
#endif


   ierr = PetscObjectGetComm( (PetscObject) PETSc_matrix, &MPI_Comm); CHKERRA(ierr);

   ierr = HYPRE_DistributedMatrixCreate( MPI_Comm, DistributedMatrix );
   if(ierr) return(ierr);

   ierr = HYPRE_DistributedMatrixSetLocalStorageType( *DistributedMatrix,
                                                     HYPRE_PETSC );
   if(ierr) return(ierr);

   ierr = HYPRE_DistributedMatrixInitialize( *DistributedMatrix );
   if(ierr) return(ierr);

   ierr = HYPRE_DistributedMatrixSetLocalStorage( *DistributedMatrix, PETSc_matrix );
   if(ierr) return(ierr);
   /* Note that this is kind of cheating, since the Mat structure contains more
      than local information... the alternative is to extract the global info
      from the Mat and put it into DistributedMatrixAuxiliaryStorage. However,
      the latter is really a "just in case" option, and so if we don't *have*
      to use it, we won't.*/

   ierr = MatGetSize( PETSc_matrix, &M, &N); if(ierr) return(ierr);
   ierr = HYPRE_DistributedMatrixSetDims( *DistributedMatrix, M, N);

   ierr = HYPRE_DistributedMatrixAssemble( *DistributedMatrix );
   if(ierr) return(ierr);

#ifdef HYPRE_TIMING
   hypre_EndTiming( timer );
   /* hypre_FinalizeTiming( timer ); */
#endif

   return(0);
}

#endif
