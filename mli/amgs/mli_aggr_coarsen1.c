/***********************************************************************
 *
 **********************************************************************/

#include "parcsr_mv.h"
#include "mli_matrix.h"
 
#define MLI_AGGR_READY     -1
#define MLI_AGGR_SELECTED  -2
#define MLI_AGGR_PENDING   -3

/***********************************************************************
 * Function  : MLI_AMG_SA_CoarsenLocal
 * Purpose   : Form aggregates
 * Inputs    : graph 
 **********************************************************************/

void MLI_AMG_SA_CoarsenLocal(MLI_Aggregation *mli_aggr, 
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

/***********************************************************************
 * Function  : MLI_AMG_SA_GenPLocal
 * Purpose   : Given Amat and aggregation information, create the 
 *             corresponding Pmat using the local aggregation scheme 
 * Inputs    : aggregate information, Amat 
 **********************************************************************/

void MLI_AMG_SA_GenPLocal( MLI_Aggregation *mli_aggr, 
                           hypre_ParCSRMatrix *Amat,
                           hypre_ParCSRMatrix *Pmat)
{
   HYPRE_IJMatrix     IJPmat;
   hypre_ParCSRMatrix *Amat, *A2mat, *Gmat, *Jmat;
   MPI_Comm           comm;
   int                *partition, start_row, end_row, global_nrows;
   int                local_nrows, start_col, global_ncols, local_ncols;
   int                *row_lengths, ierr, naggr, node2aggr, num_nulls;
   int                ierr, *p_cols;
   double             *null_vecs, *p_vals;

   /*-----------------------------------------------------------------
    * reduce Amat based on the block size information (if > 1)
    *-----------------------------------------------------------------*/

   blk_size = mli_aggr->num_PDEs;
   if ( blk_size > 1 ) MLI_Matrix_DoBlockingLocal(Amat, A2mat, blksize);
   else                A2mat = Amat;
   MLI_AMG_SA_FormGraph(mli_aggr, A2mat, Gmat);
   MLI_AMG_SA_CoarsenLocal(mli_aggr, Gmat, &node2aggr);
   if ( blk_size > 1 )
   {
      ierr = hypre_ParCSRMatrixDestroy(A2mat);
      assert( !ierr );
      free( A2mat );
   }
   ierr = hypre_ParCSRMatrixDestroy(Gmat);
   assert( !ierr );
   free( Gmat );

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
    * fetch aggregation information
    *-----------------------------------------------------------------*/

   naggr     = mli_aggr->num_aggregates;
   node2aggr = mli_aggr->node_to_aggr;
   num_nulls = mli_aggr->num_nullvectors;
   null_vecs = mli_aggr->null_vectors;

   /*-----------------------------------------------------------------
    * expand the aggregation information if block size > 1
    *-----------------------------------------------------------------*/

   if ( blk_size > 1 )
   {
      eqn2aggr = (int *) calloc( int, local_nrows );
      for ( i = 0; i < local_nrows; i++ )
         eqn2aggr[i] = node2aggr[i/blk_size];
      free( node2aggr );
   }
   else eqn2aggr = node2aggr;
 
   /*-----------------------------------------------------------------
    * fetch the coarse grid information 
    *-----------------------------------------------------------------*/

   local_ncols  = naggr * num_nulls;
   MLI_Comm_GenPartition(&partition, local_ncols, comm);
   start_col    = partition[mypid];
   global_ncols = partition[num_procs];
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

   p_cols = (int *)    calloc( int, num_nulls);
   p_vals = (double *) calloc( double, num_nulls);
   for ( irow = 0; irow < local_nrows; irow++ )
   {
      if ( null_vecs != NULL )
      {
         for ( j = 0; j < num_nulls; j++ )
         {
            p_cols[j] = start_col + eqn2aggr[irow] * num_nulls + j;
            p_vals[j] = null_vecs[j][irow];
         }
      }
      else
      {
         for ( j = 0; j < num_nulls; j++ )
         {
            p_cols[j] = cstart_row + eqn2aggr[irow] * num_nulls + j;
            if ( irow % blk_size == j ) p_vals[j] = 1.0;
            else                        p_vals[j] = 0.0;
         }
      }
      HYPRE_IJMatrixInsertRow(IJPmat,num_nulls,cstart_row+irow,p_cols,p_vals);
   }
   ierr = HYPRE_IJMatrixAssemble(IJPmat);
   assert( !ierr );
   Pmat2 = HYPRE_IJMatrixGetLocalStorageType(IJPmat, HYPRE_PARCSR);

   /*-----------------------------------------------------------------
    * compute the smoothed prolongator
    *-----------------------------------------------------------------*/

   smoothP_factor = mli_aggr->smoothP_factor;
   if ( smoothP_factor > 0.0 )
   {
      max_eigen = MLI_Eigen_ComputeMax(Amat);
      assert ( max_eigen <= 0.0 );
      MLI_Matrix_FormJacobi(Amat, Jmat, smoothP_factor/max_eigen);
      Pmat = hypre_ParMatmul( Jmat, Pmat2);
      hypre_ParCSRMatrixDestroy(Pmat2);
   }
   else Pmat = Pmat2;

   /*-----------------------------------------------------------------
    * set the block size of the next coarsening 
    *-----------------------------------------------------------------*/

   mli_aggr->num_PDEs = num_nulls;
   return;
}

