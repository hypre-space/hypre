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
#include <iostream.h>
#include <assert.h>
#include "HYPRE.h"
#include "utilities/utilities.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "seq_mv/seq_mv.h"
#include "parcsr_mv/parcsr_mv.h"

#include "amgs/mli_method_amgsa.h"
#include "util/mli_utils.h"
 
// *********************************************************************
// local defines
// ---------------------------------------------------------------------

#define MLI_METHOD_AMGSA_READY     -1
#define MLI_METHOD_AMGSA_SELECTED  -2
#define MLI_METHOD_AMGSA_PENDING   -3

#define dabs(x) ((x > 0 ) ? x : -(x))

// *********************************************************************
// external subroutines
// ---------------------------------------------------------------------

extern "C"
{
   void qsort1(int *, double *, int, int);
}

/* ********************************************************************* 
 * Purpose   : Given Amat and aggregation information, create the 
 *             corresponding Pmat using the local aggregation scheme 
 * ------------------------------------------------------------------- */

double MLI_Method_AMGSA::genPLocal(MLI_Matrix *mli_Amat,MLI_Matrix **Pmat_out,
                                   int init_count, int *init_aggr)
{
   HYPRE_IJMatrix         IJPmat;
   hypre_CSRMatrix        *A_offd, *J_diag, *J_offd;
   hypre_ParCSRMatrix     *Amat, *A2mat, *Pmat, *Gmat, *Jmat, *Pmat2;
   hypre_ParCSRCommPkg    *comm_pkg;
   hypre_ParCSRCommHandle *comm_handle;
   MLI_Matrix             *mli_Pmat, *mli_Jmat, *mli_A2mat;
   MLI_Function           *func_ptr;
   MPI_Comm  comm;
   int       i, j, mypid, num_procs, A_start_row, A_end_row, A_global_nrows;
   int       A_local_nrows, *partition, naggr, *node2aggr, *eqn2aggr, ierr;
   int       P_local_ncols, P_start_col, P_global_ncols, P_global_nrows;
   int       P_local_nrows, P_start_row, A_offd_ncols, *P_cols_ext, offset;
   int       k, irow, *col_ind, num_sends, send_leng, *send_ibuf, *P_cols;
   int       index, map_start_i, map_start_ip1, *row_lengths, J_ncols_ext;
   int       *J_diag_i, *J_diag_j, *J_offd_i, *J_offd_j, J_offd_ncols;
   int       *K_diag_j, *K_offd_j, row_size, row_num, J_start_row, nnz_cnt;
   int       J_local_nrows, cindex, max_nnz, max_nnz_diag, max_nnz_offd;
   int       *new_col_ind, *K_diag_i, *K_offd_i, old_index, old_offset;
   int       blk_size, max_agg_size, *agg_cnt_array, **agg_ind_array;
   int       agg_size, info;
   double    *col_val, **P_vecs_ext, *send_buf, **P_vecs, max_eigen=0, alpha;
   double    *J_diag_data, *J_offd_data, *K_diag_data, *K_offd_data, cvalue;
   double    *new_col_val, *new_col_itmp, *q_array, *new_null, *r_array;
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

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) Amat, &partition);
   A_start_row    = partition[mypid];
   A_end_row      = partition[mypid+1] - 1;
   A_global_nrows = partition[num_procs];
   free( partition );
   A_local_nrows = A_end_row - A_start_row + 1;

   /*-----------------------------------------------------------------
    * reduce Amat based on the block size information (if node_dofs > 1)
    *-----------------------------------------------------------------*/

   if ( init_aggr == NULL )
   {
      blk_size = curr_node_dofs;
      if (blk_size > 1) MLI_Matrix_Compress(mli_Amat, blk_size, &mli_A2mat);
      else              mli_A2mat = mli_Amat;
      A2mat = (hypre_ParCSRMatrix *) mli_A2mat->getMatrix();
   }

   /*-----------------------------------------------------------------
    * form aggregation graph by taking out weak edges
    *-----------------------------------------------------------------*/

   if ( init_aggr == NULL ) formLocalGraph(A2mat, &Gmat);

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
      if ( blk_size > 1 ) delete mli_A2mat;
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
   if ( P_global_ncols < min_coarse_size ) 
   {
      (*Pmat_out) = NULL;
      delete [] node2aggr;
      return 0.0;
   }
   P_global_nrows = A_global_nrows;
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
            cout << "Aggregation ERROR : underdetermined system in QR.\n";
            exit(1);
         }
          
         /* ------ put data into the temporary array ------ */

         for ( j = 0; j < agg_size; j++ ) 
         {
            for ( k = 0; k < nullspace_dim; k++ ) 
               q_array[agg_size*k+j] = P_vecs[k][agg_ind_array[i][j]]; 
         }
if (i == -1)
{
for ( j = 0; j < agg_size; j++ ) 
{
printf("%d, %d : ", mypid, agg_ind_array[i][j]);
for ( k = 0; k < nullspace_dim; k++ ) 
printf(" %10.3e", q_array[agg_size*k+j]);
printf("\n");
}
}

         /* ------ call QR function ------ */

         info = MLI_Utils_QR(q_array, r_array, agg_size, nullspace_dim); 
         if (info != 0)
         {
            cout << mypid << " : Aggregation ERROR : QR returned a non-zero " 
                 << i << endl;
            for ( j = 0; j < agg_size; j++ ) 
            {
               for ( k = 0; k < nullspace_dim; k++ ) 
                  printf(" %16.8e ", q_array[agg_size*k+j]);
               printf("\n");
            }
            exit(1);
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

   /*-----------------------------------------------------------------
    * if damping factor for prolongator smoother = 0
    *-----------------------------------------------------------------*/

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
            for ( j = 0; j < nullspace_dim; j++ )
            {
               col_ind[j] = P_cols[irow] + j;
               col_val[j] = P_vecs[j][irow];
            }
            row_num = P_start_row + irow;
            HYPRE_IJMatrixSetValues(IJPmat, 1, &nullspace_dim, 
                             (const int *) &row_num, (const int *) col_ind, 
                             (const double *) col_val);
         }
      }
      ierr = HYPRE_IJMatrixAssemble(IJPmat);
      assert( !ierr );
      HYPRE_IJMatrixGetObject(IJPmat, (void **) &Pmat);
      //hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) Pmat);
      HYPRE_IJMatrixSetObjectType(IJPmat, -1);
      HYPRE_IJMatrixDestroy( IJPmat );
      delete [] col_ind;
      delete [] col_val;
      if ( pre_smoother == MLI_SOLVER_MLS_ID ||
           postsmoother == MLI_SOLVER_MLS_ID )
         MLI_Utils_ComputeSpectralRadius(Amat, &max_eigen);
   }

   /*-----------------------------------------------------------------
    * form prolongator by P = (I - alpha A) tentP
    *-----------------------------------------------------------------*/

   else
   {
      /* ================================================================*/
      /* ================= old version (before Matmul debugged) ==========
       *--------------------------------------------------------------
       * fetch communication pattern of A and set up communication buffer 
       *--------------------------------------------------------------*

      comm_pkg = hypre_ParCSRMatrixCommPkg(Amat);
      if (!comm_pkg)
      {
         hypre_MatvecCommPkgCreate(Amat);
         comm_pkg = hypre_ParCSRMatrixCommPkg(Amat);
      }
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      send_leng = hypre_ParCSRCommPkgSendMapStart(comm_pkg,num_sends);
      if ( num_sends > 0 )
      {
         send_ibuf    = new int[send_leng];
         send_buf     = new double[send_leng];
         A_offd       = hypre_ParCSRMatrixOffd(Amat);
         A_offd_ncols = hypre_CSRMatrixNumCols(A_offd);
         P_cols_ext   = new int[A_offd_ncols];
         P_vecs_ext   = new double*[nullspace_dim];
         for (i = 0; i < nullspace_dim; i++) 
            P_vecs_ext[i] = new double[A_offd_ncols];
      }
      else
      {
         send_ibuf    = NULL;
         send_buf     = NULL;
         A_offd_ncols = 0;
         P_cols_ext   = NULL;
         P_vecs_ext   = NULL;
      }

      *-----------------------------------------------------------------
       * load communication buffer and send/receive
       * (to fetch the off-diagonal part of P_vecs)
       *-----------------------------------------------------------------*

      if ( num_sends > 0 )
      {
         index = 0;
         for (i = 0; i < num_sends; i++)
         {
            map_start_i   = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            map_start_ip1 = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1);
            for ( j = map_start_i; j < map_start_ip1; j++ )
               send_ibuf[index++]
                 = P_cols[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
         }
         comm_handle = hypre_ParCSRCommHandleCreate(11,comm_pkg,send_ibuf,
                                                    P_cols_ext);
         hypre_ParCSRCommHandleDestroy(comm_handle);
         comm_handle = NULL;
         for (k = 0; k < nullspace_dim; k++) 
         {
            index = 0;
            for (i = 0; i < num_sends; i++)
            {
               map_start_i   = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
               map_start_ip1 = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1);
               for ( j = map_start_i; j < map_start_ip1; j++ )
                  send_buf[index++]
                    = P_vecs[k][hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
            }
            comm_handle = hypre_ParCSRCommHandleCreate(1,comm_pkg,send_buf,
                                                       P_vecs_ext[k]);
            hypre_ParCSRCommHandleDestroy(comm_handle);
            comm_handle = NULL;
         }
      }

       *-----------------------------------------------------------------
       * compute smoothing factor for the prolongation smoother
       *-----------------------------------------------------------------*

      MLI_Utils_ComputeSpectralRadius(Amat, &max_eigen);
      if ( mypid == 0 && output_level > 1 )
         printf("\tEstimated spectral radius of A = %e\n", max_eigen);
      assert ( max_eigen > 0.0 );
      alpha = P_weight / max_eigen;

       *-----------------------------------------------------------------
       * compute the Jacobi matrix (I - alpha A) 
       *-----------------------------------------------------------------*

      MLI_Matrix_FormJacobi(mli_Amat, alpha, &mli_Jmat);
      Jmat = (hypre_ParCSRMatrix *) mli_Jmat->getMatrix();

       *-----------------------------------------------------------------
       * compute the smoothed prolongator (J * P)
       * Now :
       *  - P_vecs, P_cols has column numbers and values of tentative P
       *  - P_vecs_ext and P_cols_ext has column numbers and values of
       *    tentative P from external processors
       *  - P_cols and P_cols_ext have global column numbers
       *-----------------------------------------------------------------*

       * ----- fetch diagonal and off-diagonal blocks of J ----- *

      J_diag        = hypre_ParCSRMatrixDiag(Jmat);
      J_offd        = hypre_ParCSRMatrixOffd(Jmat);
      J_diag_i      = hypre_CSRMatrixI(J_diag);
      J_diag_j      = hypre_CSRMatrixJ(J_diag);
      J_diag_data   = hypre_CSRMatrixData(J_diag);
      J_offd_ncols  = hypre_CSRMatrixNumCols(J_offd);
      J_offd_i      = hypre_CSRMatrixI(J_offd);
      J_offd_j      = hypre_CSRMatrixJ(J_offd);
      J_offd_data   = hypre_CSRMatrixData(J_offd);
      J_start_row   = A_start_row;
      J_local_nrows = A_local_nrows;
      K_offd_i      = new int[J_local_nrows+1];
      K_offd_j      = new int[J_offd_i[J_local_nrows]];
      K_offd_data   = new double[J_offd_i[J_local_nrows]*nullspace_dim];
      K_diag_i      = new int[J_local_nrows+1];
      K_diag_j      = new int[J_diag_i[J_local_nrows]];
      K_diag_data   = new double[J_diag_i[J_local_nrows]*nullspace_dim];

       * ----- loop all rows of the J matrix ----- *

      max_nnz = 0;
      for ( irow = 0; irow < J_local_nrows; irow++ )
      {
         for ( i = J_diag_i[irow]; i < J_diag_i[irow+1]; i++ )
         {
            cindex      = J_diag_j[i];
            cvalue      = J_diag_data[i];
            K_diag_j[i] = P_cols[cindex]; 
            offset      = i * nullspace_dim;
            for ( j = 0; j < nullspace_dim; j++ )
               K_diag_data[offset+j] = cvalue * P_vecs[j][cindex]; 
         }
         max_nnz_diag = J_diag_i[irow+1] - J_diag_i[irow];
         for ( i = J_offd_i[irow]; i < J_offd_i[irow+1]; i++ )
         {
            cindex      = J_offd_j[i];
            cvalue      = J_offd_data[i];
            K_offd_j[i] = P_cols_ext[cindex]; 
            offset      = i * nullspace_dim;
            for ( j = 0; j < nullspace_dim; j++ )
               K_offd_data[offset+j] = cvalue * P_vecs_ext[j][cindex]; 
         }
         max_nnz_offd  = J_offd_i[irow+1] - J_offd_i[irow];
         if ( (max_nnz_diag + max_nnz_offd) > max_nnz )
            max_nnz = max_nnz_diag + max_nnz_offd;
      }

       * ----- allocate temporary storage space ----- *

      max_nnz = max_nnz * nullspace_dim;
      new_col_ind  = new int[max_nnz];
      new_col_itmp = new double[max_nnz];
      new_col_val  = new double[max_nnz];
      row_lengths  = new int[P_local_nrows];

       * ----- sum up each row to remove duplicate column indices ----- *

      K_diag_i[0] = 0;
      K_offd_i[0] = 0;
      for ( irow = 0; irow < J_local_nrows; irow++ )
      {
         * ----- handle the diagonal part ----- *

         k = 0;
         for ( i = J_diag_i[irow]; i < J_diag_i[irow+1]; i++ )
            new_col_ind[k++] = K_diag_j[i];
         for ( i = 0; i < k; i++ ) new_col_itmp[i] = (double) i;
         qsort1(new_col_ind, new_col_itmp, 0, k-1);
         for ( i = 0; i < k; i++ )
         {
            offset = (int) new_col_itmp[i];
            offset = ( J_diag_i[irow] + offset ) * nullspace_dim;
            for ( j = 0; j < nullspace_dim; j++ )
               new_col_val[i*nullspace_dim+j] = K_diag_data[offset+j];
         }
         old_index  = 0;
         old_offset = 0;
         for ( i = 1; i < k; i++ )
         {
            if ( new_col_ind[i] == new_col_ind[old_index] )
            {
               offset = i * nullspace_dim;
               for ( j = 0; j < nullspace_dim; j++ )
                  new_col_val[old_offset+j] += new_col_val[offset+j];
               new_col_ind[i] = -1;
            }
            else 
            {
               old_index = i;
               old_offset = old_index * nullspace_dim;
            }
         }
         nnz_cnt = K_diag_i[irow];
         for ( i = 0; i < k; i++ )
         {
            if ( new_col_ind[i] >= 0 )
            {
               K_diag_j[nnz_cnt] = new_col_ind[i];
               for ( j = 0; j < nullspace_dim; j++ )
                  K_diag_data[nnz_cnt*nullspace_dim+j] = 
                     new_col_val[i*nullspace_dim+j];
               nnz_cnt++;
            }
         }
         row_lengths[irow] = nnz_cnt - K_diag_i[irow];
         K_diag_i[irow+1] = nnz_cnt;
 
         * ----- handle the off-diagonal part ----- *

         k = 0; 
         for ( i = J_offd_i[irow]; i < J_offd_i[irow+1]; i++ )
            new_col_ind[k++] = K_offd_j[i];
         for ( i = 0; i < k; i++ ) new_col_itmp[i] = (double) i;
         qsort1(new_col_ind, new_col_itmp, 0, k-1);
         for ( i = 0; i < k; i++ )
         {
            offset = (int) new_col_itmp[i];
            offset = ( J_offd_i[irow] + offset ) * nullspace_dim;
            for ( j = 0; j < nullspace_dim; j++ )
               new_col_val[i*nullspace_dim+j] = K_offd_data[offset+j];
         }
         old_offset = 0;
         old_index  = 0;
         for ( i = 1; i < k; i++ )
         {
            if ( new_col_ind[i] == new_col_ind[old_index] )
            {
               offset = (int) new_col_itmp[i];
               for ( j = 0; j < nullspace_dim; j++ )
                  new_col_val[old_offset+j] += new_col_val[offset+j];
               new_col_ind[i] = -1;
            }
            else 
            {
               old_index = i;
               old_offset = old_index * nullspace_dim;
            }
         }
         nnz_cnt = K_offd_i[irow];
         for ( i = 0; i < k; i++ )
         {
            if ( new_col_ind[i] >= 0 )
            {
               K_offd_j[nnz_cnt] = new_col_ind[i];
               for ( j = 0; j < nullspace_dim; j++ )
                  K_offd_data[nnz_cnt*nullspace_dim+j] = 
                     new_col_val[i*nullspace_dim+j];
               nnz_cnt++;
            }
         }
         row_lengths[irow] += (nnz_cnt - K_offd_i[irow]);
         row_lengths[irow] *= nullspace_dim;
         K_offd_i[irow+1] = nnz_cnt;
      }

      * set up the row sizes of the P matrix *

      ierr = HYPRE_IJMatrixSetRowSizes(IJPmat, row_lengths);
      ierr = HYPRE_IJMatrixInitialize(IJPmat);
      assert(!ierr);
      delete [] row_lengths;
      delete [] new_col_itmp;

      * ----- now load the smoothed prolongator ----- *

      for ( irow = 0; irow < J_local_nrows; irow++ )
      {
         k = 0;
         for ( i = K_diag_i[irow]; i < K_diag_i[irow+1]; i++ )
         {
            for ( j = 0; j < nullspace_dim; j++ )
            {
               new_col_ind[k] = K_diag_j[i] + j;
               new_col_val[k++] = K_diag_data[i*nullspace_dim+j];
            }
         }
         for ( i = K_offd_i[irow]; i < K_offd_i[irow+1]; i++ )
         {
            for ( j = 0; j < nullspace_dim; j++ )
            {
               new_col_ind[k] = K_offd_j[i] + j;
               new_col_val[k++] = K_offd_data[i*nullspace_dim+j];
            }
         }
         row_num = P_start_row + irow;
         HYPRE_IJMatrixSetValues(IJPmat, 1, &k, (const int *) &row_num, 
                  (const int *) new_col_ind, (const double *) new_col_val);
      }
      ierr = HYPRE_IJMatrixAssemble(IJPmat);
      assert( !ierr );
      HYPRE_IJMatrixGetObject(IJPmat, (void **) &Pmat);
      //hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) Pmat);
      HYPRE_IJMatrixSetObjectType(IJPmat, -1);
      HYPRE_IJMatrixDestroy( IJPmat );
      //sprintf( param_string, "Pmat" );
      //MLI_Utils_HypreMatrixPrint(Pmat, param_string);

       *-----------------------------------------------------------------
       * clean up 
       *-----------------------------------------------------------------*

      delete [] new_col_ind;
      delete [] new_col_val;
      delete [] send_ibuf;
      delete [] send_buf;
      if ( P_cols_ext != NULL ) delete [] P_cols_ext;
      if ( P_vecs_ext != NULL ) 
      {
         for (i = 0; i < nullspace_dim; i++) 
            if ( P_vecs_ext[i] != NULL ) delete [] P_vecs_ext[i];
         delete [] P_vecs_ext;
      }
      delete [] K_diag_i;
      delete [] K_diag_j;
      delete [] K_diag_data;
      delete [] K_offd_i;
      delete [] K_offd_j;
      delete [] K_offd_data;

       * ================= old version (before Matmul debugged) =========*/
      /* ================================================================*/

      MLI_Utils_ComputeSpectralRadius(Amat, &max_eigen);
      if ( mypid == 0 && output_level > 1 )
         printf("\tEstimated spectral radius of A = %e\n", max_eigen);
      assert ( max_eigen > 0.0 );
      alpha = P_weight / max_eigen;

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
            for ( j = 0; j < nullspace_dim; j++ )
            {
               col_ind[j] = P_cols[irow] + j;
               col_val[j] = P_vecs[j][irow];
            }
            row_num = P_start_row + irow;
            HYPRE_IJMatrixSetValues(IJPmat, 1, &nullspace_dim, 
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
    * set the block size of the next coarsening 
    *-----------------------------------------------------------------*/

   func_ptr = new MLI_Function();
   MLI_Utils_HypreMatrixGetDestroyFunc(func_ptr);
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
   int       i, local_nrows, naggr=0, *node2aggr, *aggr_size;
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
   MPI_Allreduce(itmp, ibuf, 2, MPI_INT, MPI_SUM, comm);
   if ( mypid == 0 && output_level > 1 )
   {
      printf("\t*** Aggregation(U) P1 : no. of aggregates     = %d\n",ibuf[0]);
      printf("\t*** Aggregation(U) P1 : no. nodes aggregated  = %d\n",ibuf[1]);
   }

   /*-----------------------------------------------------------------
    * Phase 2 : put the rest into one of the existing aggregates
    *-----------------------------------------------------------------*/

   if ( ibuf[1] < global_nrows )
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
      itmp[0] = naggr;
      itmp[1] = nselected;
      MPI_Allreduce(itmp, ibuf, 2, MPI_INT, MPI_SUM, comm);
      if ( mypid == 0 && output_level > 1 )
      {
         printf("\t*** Aggregation(U) P2 : no. of aggregates     = %d\n",ibuf[0]);
         printf("\t*** Aggregation(U) P2 : no. nodes aggregated  = %d\n",ibuf[1]);
      }
   }

   /*-----------------------------------------------------------------
    * Phase 3 : form aggregates for all other rows
    *-----------------------------------------------------------------*/

   if ( ibuf[1] < global_nrows )
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
      itmp[0] = naggr;
      itmp[1] = nselected;
      MPI_Allreduce(itmp, ibuf, 2, MPI_INT, MPI_SUM, comm);
      if ( mypid == 0 && output_level > 1 )
      {
         printf("\t*** Aggregation(U) P3 : no. of aggregates     = %d\n",ibuf[0]);
         printf("\t*** Aggregation(U) P3 : no. nodes aggregated  = %d\n",ibuf[1]);
      }
   }

   /*-----------------------------------------------------------------
    * Phase 4 : finally put all lone rows into some neighbor aggregate
    *-----------------------------------------------------------------*/

   if ( ibuf[1] < global_nrows )
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
      itmp[0] = naggr;
      itmp[1] = nselected;
      MPI_Allreduce(itmp, ibuf, 2, MPI_INT, MPI_SUM, comm);
      if ( mypid == 0 && output_level > 1 )
      {
         printf("\t*** Aggregation(U) P4 : no. of aggregates     = %d\n",ibuf[0]);
         printf("\t*** Aggregation(U) P4 : no. nodes aggregated  = %d\n",ibuf[1]);
      }
   }
   if ( ibuf[1] < global_nrows )
   {
      for ( irow = 0; irow < local_nrows; irow++ )
         if ( node_stat[irow] != MLI_METHOD_AMGSA_SELECTED )
         {
            node2aggr[irow] = -1;
            node_stat[irow] != MLI_METHOD_AMGSA_SELECTED;
         }
   }

   /*-----------------------------------------------------------------
    * diagnostics
    *-----------------------------------------------------------------*/

   if ( ibuf[1] < global_nrows )
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
   (*mli_aggr_array) = node2aggr;
   (*mli_aggr_leng)  = naggr;
   return 0;
}

/***********************************************************************
 * form graph from matrix (internal subroutine)
 * ------------------------------------------------------------------- */

int MLI_Method_AMGSA::formLocalGraph( hypre_ParCSRMatrix *Amat,
                               hypre_ParCSRMatrix **graph_in)
{
   HYPRE_IJMatrix     IJGraph;
   hypre_CSRMatrix    *Adiag_block;
   hypre_ParCSRMatrix *graph;
   MPI_Comm           comm;
   int                i, j, jj, index, mypid, num_procs, *partition;
   int                start_row, end_row, local_nrow, *row_lengths;
   int                *Adiag_rptr, *Adiag_cols, Adiag_nrows, length;
   int                Aoffd_nrows, global_nrows, global_ncols;
   int                irow, max_row_nnz, ierr, *col_ind;
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
   global_nrows = hypre_ParCSRMatrixGlobalNumRows(Amat);
   global_ncols = hypre_ParCSRMatrixGlobalNumCols(Amat);
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
      if ( epsilon > 0.0 )
      {
         for (j = Adiag_rptr[irow]; j < Adiag_rptr[irow+1]; j++)
         {
            jj = Adiag_cols[j];
            if ( jj != irow )
            {
               dcomp1 = Adiag_vals[j] * Adiag_vals[j];
               if ( dcomp1 > 0.0 ) row_lengths[irow]++;
            }
         }
      }
      else 
      {
         for (j = Adiag_rptr[irow]; j < Adiag_rptr[irow+1]; j++)
         {
            jj = Adiag_cols[j];
            if ( jj != irow && Adiag_vals[j] != 0.0 ) row_lengths[irow]++;
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
      if ( epsilon > 0.0 )
      {
         for (j = Adiag_rptr[irow]; j < Adiag_rptr[irow+1]; j++)
         {
            jj = Adiag_cols[j];
            if ( jj != irow )
            {
               dcomp1 = Adiag_vals[j] * Adiag_vals[j];
               if ( dcomp1 > 0.0 )
               {
                  dcomp2 = dabs(diag_data[irow] * diag_data[jj]);
                  col_val[length] = dcomp2 / dcomp1;
                  if ( dcomp2 >= epsilon * dcomp1 ) 
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
            if ( jj != irow )
            {
               col_val[length] = Adiag_vals[j];
               if (Adiag_vals[j] != 0.0) col_ind[length++] = jj + start_row;
               else                      col_ind[length++] = -(jj+start_row)-1;
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

