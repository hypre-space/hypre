/***********************************************************************
 *
 **********************************************************************/

#include "parcsr_mv.h"
#include "mli_matrix.h"
 
#define MLI_AGGR_READY     -1
#define MLI_AGGR_SELECTED  -2
#define MLI_AGGR_PENDING   -3

/***********************************************************************
 * Function  : MLI_Coarsen_AggregationLocal
 * Purpose   : Form aggregates
 * Inputs    : graph 
 **********************************************************************/

void MLI_Coarsen_AggregationLocal(MLI_Aggregation *mli_aggr, 
                                  int *mli_aggr_leng, int **mli_aggr_array)
{
   hypre_ParCSRMatrix *hypre_graph;
   MPI_Comm           comm;
   int                mypid, num_procs, *partition, start_row, end_row;
   int                local_nrow, naggr, *node2aggr, *aggr_size;
   int                irow, icol, col_num, row_num, row_leng, *cols;
   int                *node_stat;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   hypre_graph = (hypre_ParCSRMatrix *) mli_aggr->graph;
   comm        = hypre_ParCSRMatrixComm(hypre_graph);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&num_procs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypre_graph, 
                                        &partition);
   start_row   = partition[mypid];
   end_row     = partition[mypid+1] - 1;
   local_nrows = end_row - start_row + 1;

   /*-----------------------------------------------------------------
    * this array is used to determine which row has been aggregated
    *-----------------------------------------------------------------*/

   if ( local_nrows > 0 )
   {
      node2aggr = (int *) malloc( local_nrows * sizeof(int) );
      aggr_size = (int *) malloc( local_nrows * sizeof(int) );
      node_stat = (int *) malloc( local_nrows * sizeof(int) );
      for ( irow = 0; irow < local_nrows; irow++ ) 
      {
         aggr_size[i] = 0;
         node2aggr[i] = -1;
         node_stat[i] = MLI_AGGR_READY;
      }
   }
   else node2aggr = aggr_size = node_stat = NULL;

   /*-----------------------------------------------------------------
    * Phase 1 : form aggregates
    *-----------------------------------------------------------------*/

   for ( irow = 0; irow < local_nrows; irow++ )
   {
      if ( node_stat[irow] == MLI_AGGR_READY )
      {
         row_num = start_row + irow;
         hypre_ParCSRMatrixGetRow(hypre_graph,row_num,&row_leng,&cols,NULL);
         select_flag = 1;
         for ( icol = 0; icol < row_leng; icol++ )
         {
            col_num = cols[icol] - start_row;
            if ( col_num >= 0 && col_num < local_nrows )
            {
               if ( node_stat[col_num] != MLI_AGGR_READY )
               {
                  select_flag = 0;
                  break;
               }
            }
         }
         if ( select_flag == 1 )
         {
            node2aggr[irow]  = naggr;
            aggr_size[naggr] = 1;
            node_stat[irow]  = MLI_AGGR_SELECTED;
            for ( icol = 0; icol < row_leng; icol++ )
            {
               col_num = cols[icol] - start_row;
               if ( col_num >= 0 && col_num < local_nrows )
               {
                  node2aggr[col_num] = naggr;
                  node_stat[col_num] = MLI_AGGR_SELECTED;
                  aggr_size[naggr]++;
               }
            }
            naggr++;
         }
         hypre_ParCSRMatrixRestoreRow(hypre_graph,row_num,&row_leng,&cols,NULL);
      }
   }

   /*-----------------------------------------------------------------
    * Phase 2 : put the rest into one of the existing aggregates
    *-----------------------------------------------------------------*/

   for ( irow = 0; irow < local_nrows; irow++ )
   {
      if ( node_stat[irow] == MLI_AGGR_READY )
      {
         row_num = start_row + irow;
         hypre_ParCSRMatrixGetRow(hypre_graph,row_num,&row_leng,&cols,NULL);
         for ( icol = 0; icol < row_leng; icol++ )
         {
            col_num = cols[icol] - start_row;
            if ( col_num >= 0 && col_num < local_nrows )
            {
               if ( node_stat[col_num] == MLI_AGGR_SELECTED )
               {
                  node2aggr[irow] = node2aggr[col_num];
                  node_stat[irow] = MLI_AGGR_PENDING;
                  aggr_size[node2aggr[col_num]]++;
                  break;
               }
            }
         }
         hypre_ParCSRMatrixRestoreRow(hypre_graph,row_num,&row_leng,&cols,NULL);
      }
   }
   for ( irow = 0; irow < local_nrows; irow++ )
   {
      if ( node_stat[irow] == MLI_AGGR_PENDING )
         node_stat[irow] == MLI_AGGR_SELECTED;
   } 

   /*-----------------------------------------------------------------
    * Phase 3 : form aggregates for all other rows
    *-----------------------------------------------------------------*/

   for ( irow = 0; irow < local_nrows; irow++ )
   {
      if ( node_stat[irow] == MLI_AGGR_READY )
      {
         row_num = start_row + irow;
         hypre_ParCSRMatrixGetRow(hypre_graph,row_num,&row_leng,&cols,NULL);
         node2aggr[irow]  = naggr;
         aggr_size[naggr] = 1;
         for ( icol = 0; icol < row_leng; icol++ )
         {
            col_num = cols[icol] - start_row;
            if ( col_num >= 0 && col_num < local_nrows )
            {
               if ( node_stat[col_num] == MLI_AGGR_READY )
               {
                  node_stat[col_num] = MLI_AGGR_SELECTED;
                  node2aggr[col_num] = naggr;
                  aggr_size[naggr]++;
               }
            }
            naggr++;
         }
         hypre_ParCSRMatrixRestoreRow(hypre_graph,row_num,&row_leng,&cols,NULL);
      }
   }

   /*-----------------------------------------------------------------
    * clean up and initialize the output arrays 
    *-----------------------------------------------------------------*/

   if ( local_nrows > 0 ) free( aggr_sizes ); 
   if ( local_nrows > 0 ) free( node_stat ); 
   (*mli_aggr_array) = node2aggr;
   (*mli_aggr_leng)  = naggr;
}

