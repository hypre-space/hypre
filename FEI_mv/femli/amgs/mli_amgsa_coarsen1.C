/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

// *********************************************************************
// This file is customized to use HYPRE matrix format
// *********************************************************************

// *********************************************************************
// local includes
// ---------------------------------------------------------------------

#include <string.h>
#include <assert.h>
#include "HYPRE.h"
#include "utilities/utilities.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "seq_mv/seq_mv.h"
#include "parcsr_mv/parcsr_mv.h"

#include "vector/mli_vector.h"
#include "amgs/mli_method_amgsa.h"
#include "util/mli_utils.h"
#include "solver/mli_solver.h"
 
// *********************************************************************
// local defines
// ---------------------------------------------------------------------

#define MLI_METHOD_AMGSA_READY     -1
#define MLI_METHOD_AMGSA_SELECTED  -2
#define MLI_METHOD_AMGSA_PENDING   -3

#define habs(x) ((x > 0 ) ? x : -(x))

// *********************************************************************
// external subroutines
// ---------------------------------------------------------------------

extern "C"
{
   void qsort1(int *, double *, int, int);
}

// ********************************************************************* 
// Purpose   : Given Amat and aggregation information, create the 
//             corresponding Pmat using the local aggregation scheme 
// ---------------------------------------------------------------------

double MLI_Method_AMGSA::genPLocal(MLI_Matrix *mli_Amat,
                                   MLI_Matrix **Pmat_out,
                                   int init_count, int *init_aggr)
{
   HYPRE_IJMatrix         IJPmat;
   hypre_ParCSRMatrix     *Amat, *A2mat, *Pmat, *Gmat, *Jmat, *Pmat2;
   hypre_ParCSRCommPkg    *comm_pkg;
   MLI_Matrix             *mli_Pmat, *mli_Jmat, *mli_A2mat;
   MLI_Function           *func_ptr;
   MPI_Comm  comm;
   int       i, j, mypid, num_procs, A_start_row, A_end_row;
   int       A_local_nrows, *partition, naggr, *node2aggr, *eqn2aggr, ierr;
   int       P_local_ncols, P_start_col, P_global_ncols;
   int       P_local_nrows, P_start_row, *row_lengths, row_num;
   int       k, irow, *col_ind, *P_cols, index;
   int       blk_size, max_agg_size, *agg_cnt_array, **agg_ind_array;
   int       agg_size, info, nzcnt, *local_labels, A_global_nrows;
   double    *col_val, **P_vecs, max_eigen=0, alpha;
   double    *q_array, *new_null, *r_array, ritzValues[2];
   char      param_string[200];

   /*-----------------------------------------------------------------
    * fetch matrix and machine information
    *-----------------------------------------------------------------*/

   Amat = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   comm = hypre_ParCSRMatrixComm(Amat);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&num_procs);

   /*-----------------------------------------------------------------
    * fetch other matrix information
    *-----------------------------------------------------------------*/

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) Amat,&partition);
   A_start_row    = partition[mypid];
   A_end_row      = partition[mypid+1] - 1;
   A_global_nrows = partition[num_procs];
   A_local_nrows  = A_end_row - A_start_row + 1;
   free( partition );
   if ( A_global_nrows/curr_node_dofs <= min_coarse_size || 
        A_global_nrows/curr_node_dofs <= num_procs ) 
   {
      (*Pmat_out) = NULL;
      return 0.0;
   }

   /*-----------------------------------------------------------------
    * reduce Amat based on the block size information (if node_dofs > 1)
    *-----------------------------------------------------------------*/

   if ( init_aggr == NULL )
   {
      blk_size = curr_node_dofs;
      if (blk_size > 1) 
      {
         MLI_Matrix_Compress(mli_Amat, blk_size, &mli_A2mat);
         if ( sa_labels != NULL && sa_labels[curr_level] != NULL )
         {
            local_labels = new int[A_local_nrows/blk_size];
            for ( i = 0; i < A_local_nrows; i+=blk_size )
               local_labels[i/blk_size] = sa_labels[curr_level][i];
         }
         else local_labels = NULL;
      }
      else 
      {
         mli_A2mat = mli_Amat;
         if ( sa_labels != NULL && sa_labels[curr_level] != NULL )
            local_labels = sa_labels[curr_level];
         else
            local_labels = NULL;
      }
      A2mat = (hypre_ParCSRMatrix *) mli_A2mat->getMatrix();
   }

   /*-----------------------------------------------------------------
    * form aggregation graph by taking out weak edges
    *-----------------------------------------------------------------*/

   if ( init_aggr == NULL ) formLocalGraph(A2mat, &Gmat, local_labels);

   /*-----------------------------------------------------------------
    * perform coarsening
    *-----------------------------------------------------------------*/
  
   if ( init_aggr == NULL ) coarsenLocal(Gmat, &naggr, &node2aggr);
   else 
   {
      blk_size = curr_node_dofs;
      naggr = init_count;
      node2aggr = new int[A_local_nrows];
      for ( i = 0; i < A_local_nrows; i++ ) node2aggr[i] = init_aggr[i];
   }

   /*-----------------------------------------------------------------
    * clean up graph and clean up duplicate matrix if block size > 1
    *-----------------------------------------------------------------*/

   if ( init_aggr == NULL )
   {
      if ( blk_size > 1 ) 
      {
         delete mli_A2mat;
         if ( sa_labels != NULL && sa_labels[curr_level] != NULL )
            delete [] local_labels;
      }
      ierr = hypre_ParCSRMatrixDestroy(Gmat);
      assert( !ierr );
   }

   /*-----------------------------------------------------------------
    * fetch the coarse grid information and instantiate P
    * If coarse grid size is below a given threshold, stop
    *-----------------------------------------------------------------*/

   P_local_ncols  = naggr * nullspace_dim;
   MLI_Utils_GenPartition(comm, P_local_ncols, &partition);
   P_start_col    = partition[mypid];
   P_global_ncols = partition[num_procs];
   free( partition );
   if ( P_global_ncols/nullspace_dim <= min_coarse_size || 
        P_global_ncols/nullspace_dim <= num_procs ) 
   {
      (*Pmat_out) = NULL;
      delete [] node2aggr;
      return 0.0;
   }
   P_local_nrows  = A_local_nrows;
   P_start_row    = A_start_row;
   ierr = HYPRE_IJMatrixCreate(comm,P_start_row,P_start_row+P_local_nrows-1,
                          P_start_col,P_start_col+P_local_ncols-1,&IJPmat);
   ierr = HYPRE_IJMatrixSetObjectType(IJPmat, HYPRE_PARCSR);
   assert(!ierr);

   /*-----------------------------------------------------------------
    * expand the aggregation information if block size > 1 ==> eqn2aggr
    *-----------------------------------------------------------------*/

   if ( blk_size > 1 && init_aggr == NULL )
   {
      eqn2aggr = new int[A_local_nrows];
      for ( i = 0; i < A_local_nrows; i++ )
         eqn2aggr[i] = node2aggr[i/blk_size];
      delete [] node2aggr;
   }
   else eqn2aggr = node2aggr;
 
   /*-----------------------------------------------------------------
    * construct the next set of labels for the next level
    *-----------------------------------------------------------------*/

   if ( sa_labels != NULL && sa_labels[curr_level] != NULL )
   {
      if ( (curr_level+1) < max_levels )
      {
         if ( sa_labels[curr_level+1] != NULL ) 
            delete [] sa_labels[curr_level+1];
         sa_labels[curr_level+1] = new int[P_local_ncols];
         for ( i = 0; i < naggr; i++ )
         {
            for ( j = 0; j < A_local_nrows; j++ )
               if ( eqn2aggr[j] == i ) break;
            for ( k = 0; k < nullspace_dim; k++ )
               sa_labels[curr_level+1][i*nullspace_dim+k] = 
                                              sa_labels[curr_level][j];
         }
      }
   }

   /*-----------------------------------------------------------------
    * reset row corresponding to boundary conditions
    *-----------------------------------------------------------------*/

#if 0
   if ( nullspace_vec != NULL )
   {
      for ( irow = 0; irow < A_local_nrows; irow++ )
      {
         row_num = A_start_row + irow;
         hypre_ParCSRMatrixGetRow(Amat,row_num,&row_leng,&cols,NULL);
         if ( row_leng == 1 && cols[0] == row_num )
         {
            for ( i = 0; i < nullspace_dim; i++ )
               nullspace_vec[irow+i*A_local_nrows] = 0.0;
            eqn2aggr[irow] = -1;      
         }
         hypre_ParCSRMatrixRestoreRow(Amat,row_num,&row_leng,&cols,NULL);
      }
   }
#endif

   /*-----------------------------------------------------------------
    * compute smoothing factor for the prolongation smoother
    *-----------------------------------------------------------------*/

// if ( (curr_level > 0 && P_weight != 0.0) 
   if ( (curr_level >= 0 && P_weight != 0.0) || 
        !strcmp(pre_smoother, "mls") ||
        !strcmp(postsmoother, "mls") || init_aggr != NULL )
   {
      MLI_Utils_ComputeExtremeRitzValues(Amat, ritzValues, 1);
      max_eigen = ritzValues[0];
      if ( mypid == 0 && output_level > 1 )
         printf("\tEstimated spectral radius of A = %e\n", max_eigen);
      assert ( max_eigen > 0.0 );
      alpha = P_weight / max_eigen;
   }

   /*-----------------------------------------------------------------
    * create a compact form for the null space vectors 
    * (get ready to perform QR on them)
    *-----------------------------------------------------------------*/

   P_vecs = new double*[nullspace_dim];
   P_cols = new int[P_local_nrows];
   for (i = 0; i < nullspace_dim; i++) P_vecs[i] = new double[P_local_nrows];
   for ( irow = 0; irow < P_local_nrows; irow++ )
   {
      if ( eqn2aggr[irow] >= 0 )
      {
         P_cols[irow] = P_start_col + eqn2aggr[irow] * nullspace_dim;
         if ( nullspace_vec != NULL )
         {
            for ( j = 0; j < nullspace_dim; j++ )
               P_vecs[j][irow] = nullspace_vec[j*P_local_nrows+irow];
         }
         else
         {
            for ( j = 0; j < nullspace_dim; j++ )
            {
               if ( irow % blk_size == j ) P_vecs[j][irow] = 1.0;
               else                        P_vecs[j][irow] = 0.0;
            }
         }
      }
      else
      {
         P_cols[irow] = -1;
         for ( j = 0; j < nullspace_dim; j++ ) P_vecs[j][irow] = 0.0;
      }
   }

   /*-----------------------------------------------------------------
    * perform QR for null space
    *-----------------------------------------------------------------*/

   new_null = NULL;
   if ( P_local_nrows > 0 )
   {
      /* ------ count the size of each aggregate ------ */

      agg_cnt_array = new int[naggr];
      for ( i = 0; i < naggr; i++ ) agg_cnt_array[i] = 0;
      for ( irow = 0; irow < P_local_nrows; irow++ )
         if ( eqn2aggr[irow] >= 0 ) agg_cnt_array[eqn2aggr[irow]]++;
      max_agg_size = 0;
      for ( i = 0; i < naggr; i++ ) 
         if (agg_cnt_array[i] > max_agg_size) max_agg_size = agg_cnt_array[i];

      /* ------ register which equation is in which aggregate ------ */

      agg_ind_array = new int*[naggr];
      for ( i = 0; i < naggr; i++ ) 
      {
         agg_ind_array[i] = new int[agg_cnt_array[i]];
         agg_cnt_array[i] = 0;
      }
      for ( irow = 0; irow < P_local_nrows; irow++ )
      {
         index = eqn2aggr[irow];
         if ( index >= 0 )
            agg_ind_array[index][agg_cnt_array[index]++] = irow;
      }

      /* ------ allocate storage for QR factorization ------ */

      q_array  = new double[max_agg_size * nullspace_dim];
      r_array  = new double[nullspace_dim * nullspace_dim];
      new_null = new double[naggr*nullspace_dim*nullspace_dim]; 

      /* ------ perform QR on each aggregate ------ */

      for ( i = 0; i < naggr; i++ ) 
      {
         agg_size = agg_cnt_array[i];

         if ( agg_size < nullspace_dim )
         {
            printf("Aggregation ERROR : underdetermined system in QR.\n");
            printf("            error on Proc %d\n", mypid);
            printf("            error on aggr %d (%d)\n", i, naggr);
            printf("            aggr size is %d\n", agg_size);
            exit(1);
         }
          
         /* ------ put data into the temporary array ------ */

         for ( j = 0; j < agg_size; j++ ) 
         {
            for ( k = 0; k < nullspace_dim; k++ ) 
               q_array[agg_size*k+j] = P_vecs[k][agg_ind_array[i][j]]; 
         }

         /* ------ call QR function ------ */

#if 0
         if ( mypid == 0 )
         {
            for ( j = 0; j < agg_size; j++ ) 
            {
               printf("%5d : (size=%d)\n", agg_ind_array[i][j], agg_size);
               for ( k = 0; k < nullspace_dim; k++ ) 
                  printf("%10.3e ", q_array[agg_size*k+j]);
               printf("\n");
            }
         }
#endif
         info = MLI_Utils_QR(q_array, r_array, agg_size, nullspace_dim); 
         if (info != 0)
         {
            printf("%4d : Aggregation WARNING : QR returned a non-zero for\n",
                   mypid);
            printf("  aggregate %d, size = %d, info = %d\n",i,agg_size,info);
#if 0
            for ( j = 0; j < agg_size; j++ ) 
            {
               printf("%5d : ", agg_ind_array[i][j]);
               for ( k = 0; k < nullspace_dim; k++ ) 
                  printf("%10.3e ", q_array[agg_size*k+j]);
               printf("\n");
            }
#endif
         }

         /* ------ after QR, put the R into the next null space ------ */

         for ( j = 0; j < nullspace_dim; j++ )
            for ( k = 0; k < nullspace_dim; k++ )
               new_null[i*nullspace_dim+j+k*naggr*nullspace_dim] = 
                         r_array[j+nullspace_dim*k];

         /* ------ put the P to P_vecs ------ */

         for ( j = 0; j < agg_size; j++ )
         {
            for ( k = 0; k < nullspace_dim; k++ )
            {
               index = agg_ind_array[i][j];
               P_vecs[k][index] = q_array[ k*agg_size + j ];
            }
         } 
      }
      for ( i = 0; i < naggr; i++ ) delete [] agg_ind_array[i];
      delete [] agg_ind_array;
      delete [] agg_cnt_array;
      delete [] q_array;
      delete [] r_array;
   }
   if ( nullspace_vec != NULL ) delete [] nullspace_vec;
   nullspace_vec = new_null;
   curr_node_dofs = nullspace_dim;

#if 0
   FILE *fp;
   sprintf(param_string, "null%d.%d", curr_level, mypid);
   fp = fopen( param_string, "w" );
   for ( i = 0; i < naggr*nullspace_dim; i++ ) 
   {
      for ( j = 0; j < nullspace_dim; j++ ) 
         fprintf(fp, "%25.16e ", new_null[naggr*nullspace_dim*j+i]);
      fprintf(fp, "\n");
   }
   fclose(fp);
#endif

   /*-----------------------------------------------------------------
    * if damping factor for prolongator smoother = 0
    *-----------------------------------------------------------------*/

// if ( curr_level == 0 || P_weight == 0.0 )
   if ( P_weight == 0.0 )
   {
      /*--------------------------------------------------------------
       * create and initialize Pmat 
       *--------------------------------------------------------------*/

      row_lengths = new int[P_local_nrows];
      for ( i = 0; i < P_local_nrows; i++ ) row_lengths[i] = nullspace_dim;
      ierr = HYPRE_IJMatrixSetRowSizes(IJPmat, row_lengths);
      ierr = HYPRE_IJMatrixInitialize(IJPmat);
      assert(!ierr);
      delete [] row_lengths;

      /*-----------------------------------------------------------------
       * load and assemble Pmat 
       *-----------------------------------------------------------------*/

      col_ind = new int[nullspace_dim];
      col_val = new double[nullspace_dim];
      for ( irow = 0; irow < P_local_nrows; irow++ )
      {
         if ( P_cols[irow] >= 0 )
         {
            nzcnt = 0;
            for ( j = 0; j < nullspace_dim; j++ )
            {
               if ( P_vecs[j][irow] != 0.0 )
               {
                  col_ind[nzcnt] = P_cols[irow] + j;
                  col_val[nzcnt++] = P_vecs[j][irow];
               }
            }
            row_num = P_start_row + irow;
            HYPRE_IJMatrixSetValues(IJPmat, 1, &nzcnt, 
                             (const int *) &row_num, (const int *) col_ind, 
                             (const double *) col_val);
         }
      }
      ierr = HYPRE_IJMatrixAssemble(IJPmat);
      assert( !ierr );
      HYPRE_IJMatrixGetObject(IJPmat, (void **) &Pmat);
      hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) Pmat);
      comm_pkg = hypre_ParCSRMatrixCommPkg(Amat);
      if (!comm_pkg) hypre_MatvecCommPkgCreate(Amat);
      HYPRE_IJMatrixSetObjectType(IJPmat, -1);
      HYPRE_IJMatrixDestroy( IJPmat );
      delete [] col_ind;
      delete [] col_val;
   }

   /*-----------------------------------------------------------------
    * form prolongator by P = (I - alpha A) tentP
    *-----------------------------------------------------------------*/

   else
   {
      MLI_Matrix_FormJacobi(mli_Amat, alpha, &mli_Jmat);
      Jmat = (hypre_ParCSRMatrix *) mli_Jmat->getMatrix();
      row_lengths = new int[P_local_nrows];
      for ( i = 0; i < P_local_nrows; i++ ) row_lengths[i] = nullspace_dim;
      ierr = HYPRE_IJMatrixSetRowSizes(IJPmat, row_lengths);
      ierr = HYPRE_IJMatrixInitialize(IJPmat);
      assert(!ierr);
      delete [] row_lengths;
      col_ind = new int[nullspace_dim];
      col_val = new double[nullspace_dim];
      for ( irow = 0; irow < P_local_nrows; irow++ )
      {
         if ( P_cols[irow] >= 0 )
         {
            nzcnt = 0;
            for ( j = 0; j < nullspace_dim; j++ )
            {
               if ( P_vecs[j][irow] != 0.0 )
               {
                  col_ind[nzcnt] = P_cols[irow] + j;
                  col_val[nzcnt++] = P_vecs[j][irow];
               }
            }
            row_num = P_start_row + irow;
            HYPRE_IJMatrixSetValues(IJPmat, 1, &nzcnt, 
                             (const int *) &row_num, (const int *) col_ind, 
                             (const double *) col_val);
         }
      }
      ierr = HYPRE_IJMatrixAssemble(IJPmat);
      assert( !ierr );
      HYPRE_IJMatrixGetObject(IJPmat, (void **) &Pmat2);
      HYPRE_IJMatrixSetObjectType(IJPmat, -1);
      HYPRE_IJMatrixDestroy( IJPmat );
      delete [] col_ind;
      delete [] col_val;
      Pmat = hypre_ParMatmul( Jmat, Pmat2);
      hypre_ParCSRMatrixOwnsRowStarts(Jmat) = 0; 
      hypre_ParCSRMatrixOwnsColStarts(Pmat2) = 0;
      hypre_ParCSRMatrixDestroy(Pmat2);
      delete mli_Jmat;
   }

   /*-----------------------------------------------------------------
    * clean up
    *-----------------------------------------------------------------*/

   if ( P_cols != NULL ) delete [] P_cols;
   if ( P_vecs != NULL ) 
   {
      for (i = 0; i < nullspace_dim; i++) 
         if ( P_vecs[i] != NULL ) delete [] P_vecs[i];
      delete [] P_vecs;
   }
   delete [] eqn2aggr;

   /*-----------------------------------------------------------------
    * set up and return the Pmat 
    *-----------------------------------------------------------------*/

   func_ptr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(func_ptr);
   sprintf(param_string, "HYPRE_ParCSR" ); 
   mli_Pmat = new MLI_Matrix( Pmat, param_string, func_ptr );
   (*Pmat_out) = mli_Pmat;
   delete func_ptr;
   return max_eigen;
}

/* ********************************************************************* *
 * local coarsening scheme (Given a graph, aggregate on the local subgraph)
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::coarsenLocal(hypre_ParCSRMatrix *hypre_graph,
                                   int *mli_aggr_leng, int **mli_aggr_array)
{
   MPI_Comm  comm;
   int       mypid, num_procs, *partition, start_row, end_row;
   int       local_nrows, naggr=0, *node2aggr, *aggr_size;
   int       irow, icol, col_num, row_num, row_leng, *cols, global_nrows;
   int       *node_stat, select_flag, nselected=0, count;
   int       ibuf[2], itmp[2];

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   comm = hypre_ParCSRMatrixComm(hypre_graph);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&num_procs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypre_graph, 
                                        &partition);
   start_row   = partition[mypid];
   end_row     = partition[mypid+1] - 1;
   free( partition );
   local_nrows = end_row - start_row + 1;
   MPI_Allreduce(&local_nrows, &global_nrows, 1, MPI_INT, MPI_SUM, comm);
   if ( mypid == 0 && output_level > 1 )
   {
      printf("\t*** Aggregation(U) : total nodes to aggregate = %d\n",
             global_nrows);
   }

   /*-----------------------------------------------------------------
    * this array is used to determine which row has been aggregated
    *-----------------------------------------------------------------*/

   if ( local_nrows > 0 )
   {
      node2aggr = new int[local_nrows];
      aggr_size = new int[local_nrows];
      node_stat = new int[local_nrows];
      for ( irow = 0; irow < local_nrows; irow++ ) 
      {
         aggr_size[irow] = 0;
         node2aggr[irow] = -1;
         node_stat[irow] = MLI_METHOD_AMGSA_READY;
      }
   }
   else node2aggr = aggr_size = node_stat = NULL;

   /*-----------------------------------------------------------------
    * Phase 1 : form aggregates
    *-----------------------------------------------------------------*/

   for ( irow = 0; irow < local_nrows; irow++ )
   {
      if ( node_stat[irow] == MLI_METHOD_AMGSA_READY )
      {
         row_num = start_row + irow;
         hypre_ParCSRMatrixGetRow(hypre_graph,row_num,&row_leng,&cols,NULL);
         select_flag = 1;
         count       = 1;
         for ( icol = 0; icol < row_leng; icol++ )
         {
            col_num = cols[icol] - start_row;
            if ( col_num >= 0 && col_num < local_nrows )
            {
               if ( node_stat[col_num] != MLI_METHOD_AMGSA_READY )
               {
                  select_flag = 0;
                  break;
               }
               else count++;
            }
         }
         if ( select_flag == 1 && count > 1 )
         {
            nselected++;
            node2aggr[irow]  = naggr;
            aggr_size[naggr] = 1;
            node_stat[irow]  = MLI_METHOD_AMGSA_SELECTED;
            for ( icol = 0; icol < row_leng; icol++ )
            {
               col_num = cols[icol] - start_row;
               if ( col_num >= 0 && col_num < local_nrows )
               {
                  node2aggr[col_num] = naggr;
                  node_stat[col_num] = MLI_METHOD_AMGSA_SELECTED;
                  aggr_size[naggr]++;
                  nselected++;
               }
            }
            naggr++;
         }
         hypre_ParCSRMatrixRestoreRow(hypre_graph,row_num,&row_leng,&cols,NULL);
      }
   }
   itmp[0] = naggr;
   itmp[1] = nselected;
   if (output_level > 1) MPI_Allreduce(itmp, ibuf, 2, MPI_INT, MPI_SUM, comm);
   if ( mypid == 0 && output_level > 1 )
   {
      printf("\t*** Aggregation(U) P1 : no. of aggregates     = %d\n",ibuf[0]);
      printf("\t*** Aggregation(U) P1 : no. nodes aggregated  = %d\n",ibuf[1]);
   }

   /*-----------------------------------------------------------------
    * Phase 2 : put the rest into one of the existing aggregates
    *-----------------------------------------------------------------*/

   if ( nselected < local_nrows )
   {
      for ( irow = 0; irow < local_nrows; irow++ )
      {
         if ( node_stat[irow] == MLI_METHOD_AMGSA_READY )
         {
            row_num = start_row + irow;
            hypre_ParCSRMatrixGetRow(hypre_graph,row_num,&row_leng,&cols,NULL);
            for ( icol = 0; icol < row_leng; icol++ )
            {
               col_num = cols[icol] - start_row;
               if ( col_num >= 0 && col_num < local_nrows )
               {
                  if ( node_stat[col_num] == MLI_METHOD_AMGSA_SELECTED )
                  {
                  if ( node_stat[col_num] == MLI_METHOD_AMGSA_SELECTED )
                     node2aggr[irow] = node2aggr[col_num];
                     node_stat[irow] = MLI_METHOD_AMGSA_PENDING;
                     aggr_size[node2aggr[col_num]]++;
                     break;
                  }
               }
            }
            hypre_ParCSRMatrixRestoreRow(hypre_graph,row_num,&row_leng,&cols,
                                         NULL);
         }
      }
      for ( irow = 0; irow < local_nrows; irow++ )
      {
         if ( node_stat[irow] == MLI_METHOD_AMGSA_PENDING )
         {
            node_stat[irow] = MLI_METHOD_AMGSA_SELECTED;
            nselected++;
         }
      } 
   }
   itmp[0] = naggr;
   itmp[1] = nselected;
   if (output_level > 1) MPI_Allreduce(itmp,ibuf,2,MPI_INT,MPI_SUM,comm);
   if ( mypid == 0 && output_level > 1 )
   {
      printf("\t*** Aggregation(U) P2 : no. of aggregates     = %d\n",ibuf[0]);
      printf("\t*** Aggregation(U) P2 : no. nodes aggregated  = %d\n",ibuf[1]);
   }

   /*-----------------------------------------------------------------
    * Phase 3 : form aggregates for all other rows
    *-----------------------------------------------------------------*/

   if ( nselected < local_nrows )
   {
      for ( irow = 0; irow < local_nrows; irow++ )
      {
         if ( node_stat[irow] == MLI_METHOD_AMGSA_READY )
         {
            row_num = start_row + irow;
            hypre_ParCSRMatrixGetRow(hypre_graph,row_num,&row_leng,&cols,NULL);
            count = 1;
            for ( icol = 0; icol < row_leng; icol++ )
            {
               col_num = cols[icol] - start_row;
               if ( col_num >= 0 && col_num < local_nrows )
               {
                  if ( node_stat[col_num] == MLI_METHOD_AMGSA_READY ) count++;
               }
            }
            if ( count > 1 )
            {
               node2aggr[irow]  = naggr;
               node_stat[irow]  = MLI_METHOD_AMGSA_SELECTED;
               aggr_size[naggr] = 1;
               nselected++;
               for ( icol = 0; icol < row_leng; icol++ )
               {
                  col_num = cols[icol] - start_row;
                  if ( col_num >= 0 && col_num < local_nrows )
                  {
                     if ( node_stat[col_num] == MLI_METHOD_AMGSA_READY )
                     {
                        node_stat[col_num] = MLI_METHOD_AMGSA_SELECTED;
                        node2aggr[col_num] = naggr;
                        aggr_size[naggr]++;
                        nselected++;
                     }
                  }
               }
               naggr++;
            }
            hypre_ParCSRMatrixRestoreRow(hypre_graph,row_num,&row_leng,&cols,
                                         NULL);
         }
      }
   }
   itmp[0] = naggr;
   itmp[1] = nselected;
   if (output_level > 1) MPI_Allreduce(itmp,ibuf,2,MPI_INT,MPI_SUM,comm);
   if ( mypid == 0 && output_level > 1 )
   {
      printf("\t*** Aggregation(U) P3 : no. of aggregates     = %d\n",ibuf[0]);
      printf("\t*** Aggregation(U) P3 : no. nodes aggregated  = %d\n",ibuf[1]);
   }

   /*-----------------------------------------------------------------
    * Phase 4 : finally put all lone rows into some neighbor aggregate
    *-----------------------------------------------------------------*/

   if ( nselected < local_nrows )
   {
      for ( irow = 0; irow < local_nrows; irow++ )
      {
         if ( node_stat[irow] == MLI_METHOD_AMGSA_READY )
         {
            row_num = start_row + irow;
            hypre_ParCSRMatrixGetRow(hypre_graph,row_num,&row_leng,&cols,NULL);
            for ( icol = 0; icol < row_leng; icol++ )
            {
               if ( col_num < 0 ) col_num = - col_num - 1;
               col_num = cols[icol] - start_row;
               if ( col_num >= 0 && col_num < local_nrows )
               {
                  if ( node_stat[col_num] == MLI_METHOD_AMGSA_SELECTED )
                  {
                     node2aggr[irow] = node2aggr[col_num];
                     node_stat[irow] = MLI_METHOD_AMGSA_SELECTED;
                     aggr_size[node2aggr[col_num]]++;
                     break;
                  }
               }
            }
            hypre_ParCSRMatrixRestoreRow(hypre_graph,row_num,&row_leng,&cols,
                                         NULL);
         }
      }
   }
   itmp[0] = naggr;
   itmp[1] = nselected;
   if ( output_level > 1 ) MPI_Allreduce(itmp,ibuf,2,MPI_INT,MPI_SUM,comm);
   if ( mypid == 0 && output_level > 1 )
   {
      printf("\t*** Aggregation(U) P4 : no. of aggregates     = %d\n",ibuf[0]);
      printf("\t*** Aggregation(U) P4 : no. nodes aggregated  = %d\n",ibuf[1]);
   }
   if ( nselected < local_nrows )
   {
      for ( irow = 0; irow < local_nrows; irow++ )
         if ( node_stat[irow] != MLI_METHOD_AMGSA_SELECTED )
         {
            node2aggr[irow] = -1;
            node_stat[irow] = MLI_METHOD_AMGSA_SELECTED;
         }
   }

   /*-----------------------------------------------------------------
    * diagnostics
    *-----------------------------------------------------------------*/

   if ( nselected < local_nrows )
   {
      for ( irow = 0; irow < local_nrows; irow++ )
      {
         if ( node_stat[irow] != MLI_METHOD_AMGSA_SELECTED )
         {
            row_num = start_row + irow;
#ifdef MLI_DEBUG_DETAILED
            printf("%5d : unaggregated node = %8d\n", mypid, row_num);
#endif
            hypre_ParCSRMatrixGetRow(hypre_graph,row_num,&row_leng,&cols,NULL);
            for ( icol = 0; icol < row_leng; icol++ )
            {
               //if ( col_num < 0 ) col_num = - col_num - 1;
               //col_num = cols[icol] - start_row;
               col_num = cols[icol];
               printf("ERROR : neighbor of unselected node %9d = %9d\n", 
                     row_num, col_num);
            }
         }
      }
   }

   /*-----------------------------------------------------------------
    * clean up and initialize the output arrays 
    *-----------------------------------------------------------------*/

   if ( local_nrows > 0 ) delete [] aggr_size; 
   if ( local_nrows > 0 ) delete [] node_stat; 
   if ( local_nrows == 1 && naggr == 0 )
   {
      node2aggr[0] = 0;
      naggr = 1;
   }
   (*mli_aggr_array) = node2aggr;
   (*mli_aggr_leng)  = naggr;
   return 0;
}

/***********************************************************************
 * form graph from matrix (internal subroutine)
 * ------------------------------------------------------------------- */

int MLI_Method_AMGSA::formLocalGraph( hypre_ParCSRMatrix *Amat,
                                      hypre_ParCSRMatrix **graph_in,
                                      int *local_labels)
{
   HYPRE_IJMatrix     IJGraph;
   hypre_CSRMatrix    *Adiag_block;
   hypre_ParCSRMatrix *graph;
   MPI_Comm           comm;
   int                i, j, jj, index, mypid, num_procs, *partition;
   int                start_row, end_row, *row_lengths;
   int                *Adiag_rptr, *Adiag_cols, Adiag_nrows, length;
   int                irow, max_row_nnz, ierr, *col_ind, labeli, labelj;
   double             *diag_data=NULL, *col_val;
   double             *Adiag_vals, dcomp1, dcomp2, epsilon;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   assert( Amat != NULL );
   comm = hypre_ParCSRMatrixComm(Amat);
   MPI_Comm_rank(comm,&mypid);
   MPI_Comm_size(comm,&num_procs);

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) Amat,&partition);
   start_row    = partition[mypid];
   end_row      = partition[mypid+1] - 1;
   free( partition );
   Adiag_block  = hypre_ParCSRMatrixDiag(Amat);
   Adiag_nrows  = hypre_CSRMatrixNumRows(Adiag_block);
   Adiag_rptr   = hypre_CSRMatrixI(Adiag_block);
   Adiag_cols   = hypre_CSRMatrixJ(Adiag_block);
   Adiag_vals   = hypre_CSRMatrixData(Adiag_block);
   
   /*-----------------------------------------------------------------
    * construct the diagonal array (diag_data) 
    *-----------------------------------------------------------------*/

   if ( threshold > 0.0 )
   {
      diag_data = new double[Adiag_nrows];

#define HYPRE_SMP_PRIVATE irow,j
#include "utilities/hypre_smp_forloop.h"
      for (irow = 0; irow < Adiag_nrows; irow++)
      {
         for (j = Adiag_rptr[irow]; j < Adiag_rptr[irow+1]; j++)
         {
            if ( Adiag_cols[j] == irow )
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

   ierr = HYPRE_IJMatrixCreate(comm, start_row, end_row, start_row,
                               end_row, &IJGraph);
   ierr = HYPRE_IJMatrixSetObjectType(IJGraph, HYPRE_PARCSR);
   assert(!ierr);

   /*-----------------------------------------------------------------
    * find and initialize the length of each row in the graph
    *-----------------------------------------------------------------*/

   epsilon = threshold;
   for ( i = 0; i < curr_level; i++ ) epsilon *= 0.5;
   if ( mypid == 0 && output_level > 1 )
   {
      printf("\t*** Aggregation(U) : strength threshold       = %8.2e\n",
             epsilon);
   }
   epsilon = epsilon * epsilon;
   row_lengths = new int[Adiag_nrows];

#define HYPRE_SMP_PRIVATE irow,j,jj,index,dcomp1,dcomp2
#include "utilities/hypre_smp_forloop.h"
   for ( irow = 0; irow < Adiag_nrows; irow++ )
   {
      row_lengths[irow] = 0;
      index = start_row + irow;
      if ( local_labels != NULL ) labeli = local_labels[irow];
      else                        labeli = 0;
      if ( epsilon > 0.0 )
      {
         for (j = Adiag_rptr[irow]; j < Adiag_rptr[irow+1]; j++)
         {
            jj = Adiag_cols[j];
            if ( local_labels != NULL ) labelj = local_labels[jj];
            else                        labelj = 0;
            if ( jj != irow )
            {
               dcomp1 = Adiag_vals[j] * Adiag_vals[j];
               if (dcomp1 > 0.0 && labeli == labelj) row_lengths[irow]++;
            }
         }
      }
      else 
      {
         for (j = Adiag_rptr[irow]; j < Adiag_rptr[irow+1]; j++)
         {
            jj = Adiag_cols[j];
            if ( local_labels != NULL ) labelj = local_labels[jj];
            else                        labelj = 0;
            if ( jj != irow && Adiag_vals[j] != 0.0 && labeli == labelj )
               row_lengths[irow]++;
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
   delete [] row_lengths;

   /*-----------------------------------------------------------------
    * load and assemble the graph
    *-----------------------------------------------------------------*/

   col_ind = new int[max_row_nnz];
   col_val = new double[max_row_nnz];
   for ( irow = 0; irow < Adiag_nrows; irow++ )
   {
      length = 0;
      index  = start_row + irow;
      if ( local_labels != NULL ) labeli = local_labels[irow];
      else                        labeli = 0;
      if ( epsilon > 0.0 )
      {
         for (j = Adiag_rptr[irow]; j < Adiag_rptr[irow+1]; j++)
         {
            jj = Adiag_cols[j];
            if ( local_labels != NULL ) labelj = local_labels[jj];
            else                        labelj = 0;
            if ( jj != irow )
            {
               dcomp1 = Adiag_vals[j] * Adiag_vals[j];
               if ( dcomp1 > 0.0 )
               {
                  dcomp2 = habs(diag_data[irow] * diag_data[jj]);
                  col_val[length] = dcomp2 / dcomp1;
                  if ( (dcomp2 >= epsilon * dcomp1) && (labeli == labelj) ) 
                     col_ind[length++] = jj + start_row;
                  else                              
                     col_ind[length++] = - (jj + start_row) - 1;
               }
            }
         }
      }
      else 
      {
         for (j = Adiag_rptr[irow]; j < Adiag_rptr[irow+1]; j++)
         {
            jj = Adiag_cols[j];
            if ( local_labels != NULL ) labelj = local_labels[jj];
            else                        labelj = 0;
            if ( jj != irow )
            {
               col_val[length] = Adiag_vals[j];
               if (Adiag_vals[j] != 0.0 && (labeli == labelj)) 
                    col_ind[length++] = jj + start_row;
               else col_ind[length++] = -(jj+start_row)-1;
            }
         }
      }
      HYPRE_IJMatrixSetValues(IJGraph, 1, &length, (const int *) &index, 
                              (const int *) col_ind, (const double *) col_val);
   }
   ierr = HYPRE_IJMatrixAssemble(IJGraph);
   assert(!ierr);

   /*-----------------------------------------------------------------
    * return the graph and clean up
    *-----------------------------------------------------------------*/

   HYPRE_IJMatrixGetObject(IJGraph, (void **) &graph);
   HYPRE_IJMatrixSetObjectType(IJGraph, -1);
   HYPRE_IJMatrixDestroy(IJGraph);
   (*graph_in) = graph;
   delete [] col_ind;
   delete [] col_val;
   if ( threshold > 0.0 ) delete [] diag_data;
   return 0;
}

#undef MLI_METHOD_AMGSA_READY
#undef MLI_METHOD_AMGSA_SELECTED
#undef MLI_METHOD_AMGSA_PENDING

