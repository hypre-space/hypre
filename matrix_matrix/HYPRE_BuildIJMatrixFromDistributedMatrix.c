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

/* Prototypes for DistributedMatrix */
#include "../distributed_matrix/HYPRE_distributed_matrix_types.h"
#include "../distributed_matrix/HYPRE_distributed_matrix_protos.h"

/* Matrix prototypes for IJMatirx */
#include "../IJ_matrix_vector/HYPRE_IJ_mv.h"

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
int 
HYPRE_BuildIJMatrixFromDistributedMatrix(
                 HYPRE_DistributedMatrix DistributedMatrix,
                 HYPRE_IJMatrix *ij_matrix,
                 int local_storage_type )
{
   int ierr;
   MPI_Comm comm;
   int M, N;
   int first_local_row, last_local_row;
   int i;
   int size, *col_ind;
   double *values;



   if (!DistributedMatrix) return(-1);

   comm = HYPRE_GetDistributedMatrixContext( DistributedMatrix );
   ierr = HYPRE_GetDistributedMatrixDims( DistributedMatrix, &M, &N );

   ierr = HYPRE_NewIJMatrix( comm, ij_matrix, M, N );

   ierr = HYPRE_SetIJMatrixLocalStorageType( 
                 *ij_matrix, local_storage_type );
   if(ierr) return(ierr);

   ierr = HYPRE_GetDistributedMatrixLocalRange( DistributedMatrix, 
             &first_local_row, &last_local_row );

   ierr = HYPRE_SetIJMatrixLocalSize( *ij_matrix, 
                last_local_row-first_local_row+1, N );

   ierr = HYPRE_InitializeIJMatrix( *ij_matrix );
   if(ierr) return(ierr);

   /* Loop through all locally stored rows and insert them into ij_matrix */
   for (i=first_local_row; i<= last_local_row; i++)
   {
      ierr = HYPRE_GetDistributedMatrixRow( DistributedMatrix, i, &size, &col_ind, &values );
      if( ierr ) return(ierr);

      ierr = HYPRE_InsertIJMatrixRow( *ij_matrix, size, i, col_ind, values );
      if( ierr ) return(ierr);

      ierr = HYPRE_RestoreDistributedMatrixRow( DistributedMatrix, i, &size, &col_ind, &values );
      if( ierr ) return(ierr);

   }

   ierr = HYPRE_AssembleIJMatrix( *ij_matrix );
   if(ierr) return(ierr);

   return(ierr);
}

