/***********************************************************************
 *
 **********************************************************************/

#include "parcsr_mv.h"
#include "mli_matrix.h"
 
#define MLI_AGGR_READY     -1
#define MLI_AGGR_SELECTED  -2
#define MLI_AGGR_PENDING   -3

/***********************************************************************
 * Function  : MLI_AMG_SA_FormGraph
 * Purpose   : Form graph from the matrix
 **********************************************************************/

void MLI_AMG_SA_FormGraph(MLI_Aggregation *mli_aggr)
{
   HYPRE_IJMatrix         IJGraph;
   hypre_ParCSRMatrix     *Amat;
   hypre_CSRMatrix        *Adiag_block;
   MPI_Comm               comm;
   int                    i, j, jj, index, mypid, num_procs, *partition;
   int                    start_row, end_row, local_nrow, *row_lengths;
   int                    *Adiag_rptr, *Adiag_cols, Adiag_nrows;
   int                    Aoffd_nrows, global_nrows, global_ncols;
   int                    irow, max_row_nnz, ierr, *col_ind;
   double                 threshold, *diag_data=NULL, *col_val;
   double                 *Adiag_vals, dcomp1, dcomp2;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   Amat = (hypre_ParCSRMatrix *) mli_aggr->Amat;
   comm = hypre_ParCSRMatrixComm(Amat);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&num_procs);

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) Amat,&partition);
   start_row    = partition[mypid];
   end_row      = partition[mypid+1] - 1;
   global_nrows = hypre_ParCSRMatrixGlobalNumRows(Amat);
   global_ncols = hypre_ParCSRMatrixGlobalNumRows(Amat);
   Adiag_block  = hypre_ParCSRMatrixDiag(Amat);
   Adiag_nrows  = hypre_CSRMatrixNumRows(Adiag_block);
   Adiag_rptr   = hypre_CSRMatrixI(Adiag_block);
   Adiag_cols   = hypre_CSRMatrixJ(Adiag_block);
   Adiag_vals   = hypre_CSRMatrixData(Adiag_block);
   
   /*-----------------------------------------------------------------
    * construct the diagonal array (diag_data) 
    *-----------------------------------------------------------------*/

   threshold = mli_aggr->threshold;
   if ( threshold > 0.0 )
   {
      diag_data = (double *) calloc(double, Adiag_nrows);

#define HYPRE_SMP_PRIVATE irow,j
#include "../utilities/hypre_smp_forloop.h"
      for (irow = 0; irow < Adiag_nrows; irow++)
      {
         for (j = Adiag_rptr[i]; j < Adiag_rptr[i+1]; j++)
         {
            if ( Adiag_cols[j] == (start_row + irow) )
            {
               diag_data[irow] = Adiag_vals[j];
               break;
            }
         }
      }
   }

   /*-----------------------------------------------------------------
    * initialize the graph
    *-----------------------------------------------------------------*/

   ierr = HYPRE_IJMatrixCreate(comm,&IJGraph,global_nrows,global_ncols);
   ierr = HYPRE_IJMatrixSetLocalStorageType(IJGraph, HYPRE_PARCSR);
   ierr = HYPRE_IJMatrixSetLocalSize(IJGraph, Adiag_nrows, Adiag_nrows);
   assert(!ierr);

   /*-----------------------------------------------------------------
    * find and initialize the length of each row in the graph
    *-----------------------------------------------------------------*/

   row_lengths = (int *) calloc( int, Adiag_nrows );

#define HYPRE_SMP_PRIVATE irow,j,jj,index,dcomp1,dcomp2
#include "../utilities/hypre_smp_forloop.h"
   for ( irow = 0; irow < Adiag_nrows; irow++ )
   {
      row_lengths[irow] = 0;
      index = start_row + irow;
      if ( threshold > 0.0 )
      {
         for (j = Adiag_rptr[irow]; j < Adiag_rptr[irow+1]; j++)
         {
            jj = Adiag_cols[j];
            if ( jj >= start_row && jj <= end_row && jj != (start_row+i) )
            {
               dcomp1 = Adiag_vals[j] * Adiag_vals[j];
               if ( dcomp1 > 0.0 )
               {
                  dcomp2 = dabs(diag_data[irow] * diag_data[jj-start_row]);
                  if ( dcomp1 >= threshold * dcomp2 ) row_lengths[irow]++;
               }
            }
         }
      }
      else 
      {
         for (j = Adiag_rptr[irow]; j < Adiag_rptr[irow+1]; j++)
         {
            jj = Adiag_cols[j];
            if ( jj >= start_row && jj <= end_row && jj != (start_row+i) )
               if ( Adiag_vals[j] != 0.0 ) row_lengths[irow]++;
         }
      }
   }
   max_row_nnz = 0;
   for ( irow = 0; irow < Adiag_nrows; irow++ )
   {
      if ( row_lengths[irow] > max_row_nnz ) max_row_nnz = row_lengths[irow];
   }
   ierr = HYPRE_IJMatrixSetRowSizes(IJGraph, row_lengths);
   ierr = HYPRE_IJMatrixInitialize(IJGraph);
   assert(!ierr);
   free( row_lengths );

   /*-----------------------------------------------------------------
    * load and assemble the graph
    *-----------------------------------------------------------------*/

   col_ind = (int *)    calloc( int,    max_row_nnz );
   col_val = (double *) calloc( double, max_row_nnz );
   for ( i = 0; i < max_row_nnz; i++ ) col_val[i] = 1.0;
   for ( irow = 0; irow < Adiag_nrows; irow++ )
   {
      length = 0;
      index  = start_row + i;
      if ( threshold > 0.0 )
      {
         for (j = Adiag_rptr[irow]; j < Adiag_rptr[irow+1]; j++)
         {
            jj = Adiag_cols[j];
            if ( jj >= start_row && jj <= end_row && jj != index )
            {
               dcomp1 = Adiag_vals[j] * Adiag_vals[j];
               if ( dcomp1 > 0.0 )
               {
                  dcomp2 = dabs(diag_data[irow] * diag_data[jj-start_row]);
                  if ( dcomp1 >= threshold * dcomp2 ) col_ind[length++] = jj;
               }
            }
         }
      }
      else 
      {
         for (j = Adiag_rptr[irow]; j < Adiag_rptr[irow+1]; j++)
         {
            jj = Adiag_cols[j];
            if ( jj >= start_row && jj <= end_row && jj != index )
               if ( Adiag_vals[j] != 0.0 ) col_ind[length++] = jj;
         }
      }
      HYPRE_IJMatrixInsertRow(IJGraph,length,index,col_ind,col_val);
   }
   ierr = HYPRE_IJMatrixAssemble(IJGraph);
   assert(!ierr);

   /*-----------------------------------------------------------------
    * store the graph and clean up
    *-----------------------------------------------------------------*/

   mli_aggr->graph = IJGraph;
   free( col_ind );
   free( col_val );
   if ( threshold > 0.0 ) free( diag_data );
   return 0;
}

