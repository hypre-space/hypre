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

#include "general.h"

#include "HYPRE.h"
#include "HYPRE_utilities.h"

/* Prototypes for DistributedMatrix */
#include "HYPRE_distributed_matrix_types.h"
#include "HYPRE_distributed_matrix_protos.h"

/* Matrix prototypes for IJMatrix */
#include "IJ_mv/HYPRE_IJ_mv.h"

/* Local routine prototypes */
HYPRE_Int HYPRE_IJMatrixSetLocalStorageType(HYPRE_IJMatrix ij_matrix,
                                            HYPRE_Int local_storage_type );

HYPRE_Int HYPRE_IJMatrixSetLocalSize(HYPRE_IJMatrix ij_matrix,
                                     HYPRE_Int row, HYPRE_Int col );

HYPRE_Int HYPRE_IJMatrixInsertRow( HYPRE_IJMatrix ij_matrix,
                                   HYPRE_Int size, HYPRE_BigInt i, HYPRE_BigInt *col_ind,
                                   HYPRE_Real *values );

/*--------------------------------------------------------------------------
 * HYPRE_BuildIJMatrixFromDistributedMatrix
 *--------------------------------------------------------------------------*/
/**
Builds an IJMatrix from a distributed matrix by pulling rows out of the
distributed_matrix and putting them into the IJMatrix. This routine does not
effect the distributed matrix. In essence, it makes a copy of the input matrix
in another format. NOTE: because this routine makes a copy and is not just
a simple conversion, it is memory-expensive and should only be used in
low-memory requirement situations (such as unit-testing code).
*/
HYPRE_Int
HYPRE_BuildIJMatrixFromDistributedMatrix(
   HYPRE_DistributedMatrix DistributedMatrix,
   HYPRE_IJMatrix *ij_matrix,
   HYPRE_Int local_storage_type )
{
   HYPRE_Int ierr;
   MPI_Comm comm;
   HYPRE_BigInt M, N;
   HYPRE_BigInt first_local_row, last_local_row;
   HYPRE_BigInt first_local_col, last_local_col;
   HYPRE_BigInt i;
   HYPRE_Int size;
   HYPRE_BigInt *col_ind;
   HYPRE_Real *values;



   if (!DistributedMatrix) { return (-1); }

   comm = HYPRE_DistributedMatrixGetContext( DistributedMatrix );
   ierr = HYPRE_DistributedMatrixGetDims( DistributedMatrix, &M, &N );

   ierr = HYPRE_DistributedMatrixGetLocalRange( DistributedMatrix,
                                                &first_local_row, &last_local_row,
                                                &first_local_col, &last_local_col );

   ierr = HYPRE_IJMatrixCreate( comm, first_local_row, last_local_row,
                                first_local_col, last_local_col,
                                ij_matrix );

   ierr = HYPRE_IJMatrixSetLocalStorageType(
             *ij_matrix, local_storage_type );
   /* if(ierr) return(ierr); */

   ierr = HYPRE_IJMatrixSetLocalSize( *ij_matrix,
                                      last_local_row - first_local_row + 1,
                                      last_local_col - first_local_col + 1 );

   ierr = HYPRE_IJMatrixInitialize( *ij_matrix );
   /* if(ierr) return(ierr);*/

   /* Loop through all locally stored rows and insert them into ij_matrix */
   for (i = first_local_row; i <= last_local_row; i++)
   {
      ierr = HYPRE_DistributedMatrixGetRow( DistributedMatrix, i, &size, &col_ind, &values );
      /* if( ierr ) return(ierr);*/

      ierr = HYPRE_IJMatrixInsertRow( *ij_matrix, size, i, col_ind, values );
      /* if( ierr ) return(ierr);*/

      ierr = HYPRE_DistributedMatrixRestoreRow( DistributedMatrix, i, &size, &col_ind, &values );
      /* if( ierr ) return(ierr); */

   }

   ierr = HYPRE_IJMatrixAssemble( *ij_matrix );
   /* if(ierr) return(ierr); */

   return (ierr);
}

