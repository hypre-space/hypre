/***********************************************************************
 *
 **********************************************************************/

#include "parcsr_mv.h"
#include "mli_matrix.h"
 
/***********************************************************************
 * Function  : MLI_AMG_SA_GenPLocal
 * Purpose   : create the prolongation operator (local scheme) 
 * Inputs    : graph 
 **********************************************************************/

void MLI_AMG_SA_GenPLocal( MLI_Aggregation *mli_aggr, 
                           hypre_ParCSRMatrix *Pmat)
{
   HYPRE_IJMatrix     IJPmat;
   hypre_ParCSRMatrix *Amat;
   MPI_Comm           comm;
   int                *partition, start_row, end_row, global_nrows;
   int                local_nrows, start_col, global_ncols, local_ncols;
   int                *row_lengths, ierr, naggr, node2aggr, num_nulls;
   int                *p_cols;
   double             *null_vecs, *p_vals;

   /*-----------------------------------------------------------------
    * fetch machine and matrix information
    *-----------------------------------------------------------------*/

   Amat = (hypre_ParCSRMatrix *) mli_aggr->matrix;
   comm = hypre_ParCSRMatrixComm(Amat);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&num_procs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) Amat,&partition);
   start_row    = partition[mypid];
   end_row      = partition[mypid+1] - 1;
   global_nrows = partition[num_procs];
   free( partition );
   local_nrows = end_row - start_row + 1;

   /*-----------------------------------------------------------------
    * fetch the coarse grid information 
    *-----------------------------------------------------------------*/

   naggr        = mli_aggr->num_aggregates;
   node2aggr    = mli_aggr->node_to_aggr;
   num_nulls    = mli_aggr->num_nullvectors;
   clocal_nrows = naggr * num_nulls;
   MLI_Comm_GenPartition(&partition, clocal_nrows, comm);
   start_col    = partition[mypid];
   global_ncols = partition[num_procs];
   local_ncols  = partition[mypid+1] - partition[mypid];
   free( partition );

   /*-----------------------------------------------------------------
    * create and initialize Pmat 
    *-----------------------------------------------------------------*/

   ierr = HYPRE_IJMatrixCreate(comm,&IJPmat,global_nrows,global_ncols);
   ierr = HYPRE_IJMatrixSetLocalStorageType(IJPmat, HYPRE_PARCSR);
   ierr = HYPRE_IJMatrixSetLocalSize(IJPmat, local_nrows, local_ncols);
   assert(!ierr);
   row_lengths = (int *) calloc( int, local_nrows );
   for ( i = 0; i < local_nrows; i++ ) row_lengths[i] = num_nulls;
   ierr = HYPRE_IJMatrixSetRowSizes(IJPmat, row_lengths);
   ierr = HYPRE_IJMatrixInitialize(IJPmat);
   assert(!ierr);
   free( row_lengths );

   /*-----------------------------------------------------------------
    * load and assemble Pmat 
    *-----------------------------------------------------------------*/

   null_vecs = mli_aggr->nullvectors;
   p_cols = (int *)    calloc( int, num_nulls);
   p_vals = (double *) calloc( double, num_nulls);
   for ( irow = 0; irow < local_nrows; irow++ )
   {
      if ( null_vecs != NULL )
      {
         for ( j = 0; j < num_nulls; j++ )
         {
            p_cols[j] = start_col + node2aggr[irow] * num_nulls + j;
            p_vals[j] = null_vecs[j][irow];
         }
      }
      else
      {
         for ( j = 0; j < num_nulls; j++ )
         {
            p_cols[j] = cstart_row + node2aggr[irow] * num_nulls + j;
            if ( irow % num_nulls == j ) p_vals[j] = 1.0;
            else                         p_vals[j] = 0.0;
         }
      }
      HYPRE_IJMatrixInsertRow(IJPmat,num_nulls,cstart_row+irow,p_cols,p_vals);
   }
   ierr = HYPRE_IJMatrixAssemble(IJPmat);
   assert( !ierr );
   Pmat = HYPRE_IJMatrixGetLocalStorageType(IJPmat, HYPRE_PARCSR);
   return 0;
}

