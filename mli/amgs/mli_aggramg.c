/**************************************************************************
 *
 *************************************************************************/

#include "parcsr_mv.h"
#include "mli_matrix.h"
#include "mli_amg_sa.h"
 
/* ********************************************************************* *
 * Function   : MLI_AggrAMGCreate                                        *
 * --------------------------------------------------------------------- */

MLI_AggrAMG *MLI_AggrAMGCreate()
{
   MLI_AggrAMG *object;

   strcpy(method_name, "aggregation");
   object = (MLI_AggrAMG *) calloc( MLI_AMG_SA, 1);
   object->max_levels        = 1;
   object->debug_level       = 0;
   object->node_dofs         = 1;
   object->threshold         = 0.08;
   object->nullspace_dim     = 1;
   object->nullspace_vec     = NULL;
   object->P_weight          = 4.0/3.0;
   object->matrix_rowsums    = NULL;           /* matrix rowsum norms    */
   object->matrix_sizes      = NULL;           /* matrix dimensions      */
   object->sa_data           = NULL;           /* aggregate information  */
   object->spectral_norms    = NULL;           /* calculated max eigen   */
   object->calc_norm_scheme  = 0;              /* use matrix rowsum norm */
   object->min_coarse_size   = 100;            /* smallest coarse grid   */
   object->coarsen_scheme    = MLI_AGGRAMG_LOCAL;
   object->mat_complexity    = NULL;
   object->oper_complexity   = NULL;
   object->pre_smoothers     = 0;
   object->postsmoothers     = 0;
   object->pre_smoother_num  = 0;
   object->postsmoother_num  = 0;
   object->pre_smoother_wgt  = 0.0;
   object->postsmoother_wgt  = 0.0;
   object->coarse_solver     = 0;
   object->coarse_solver_num = 1;
   object->coarse_solver_wgt = 1.0;
   object->mpi_comm          = (MPI_Comm) 0;
   return object;
}

/* ********************************************************************* *
 * Function   : MLI_AggrAMGDestroy                                       *
 * --------------------------------------------------------------------- */

int MLI_AggrAMGDestroy( MLI_AMG_SA *object )
{
   if ( object->nullspace_vec != NULL )
   {
      free( object->nullspace_vec );
      object->nullspace_vec = NULL;
   } 
   if ( object->matrix_rowsums != NULL )
   {
      free( object->matrix_rowsums );
      object->matrix_rowsums = NULL;
   }
   if ( object->matrix_sizes != NULL )
   {
      free( object->matrix_sizes );
      object->matrix_sizes = NULL;
   }
   if ( object->spectral_norms != NULL )
   {
      free( object->spectral_norms );
      object->spectral_norms = NULL;
   }
   if ( object->sa_data != NULL )
   {
      for ( i = object->max_levels-1; i >= 0; i-- )
      {
         if ( object->sa_data[i] != NULL )
            free( object->sa_data[i] );
         else
            break;
      }
      free( object->sa_data );
      object->sa_data = NULL;
   }
   free( object );
   return 0;
}

/* ********************************************************************* *
 * Function   : MLI_AggrAMGSetDebugLevel                                 *
 * --------------------------------------------------------------------- */

int MLI_AggrAMGSetDebugLevel( MLI_AggrAMG *object, int debug_level )
{
   object->debug_level = debug_level;
   return 0;
}

/* ********************************************************************* *
 * Function   : MLI_AggrAMGSetMinCoarseSize                              *
 * --------------------------------------------------------------------- */

int MLI_AggrAMGSetMinCoarseSize( MLI_AggrAMG *object, int coarse_size  )
{
   object->min_coarse_size = size;
   return 0;
}

/* ********************************************************************* *
 * Function   : MLI_AggrAMGSetCoarsenSchemeLocal                         *
 * --------------------------------------------------------------------- */

int MLI_AggrAMGSetCoarsenSchemeLocal( MLI_AggrAMG *object )
{
   object->coarsen_scheme = MLI_AGGRAMG_LOCAL;
   return 0;
}

/* ********************************************************************* *
 * Function   : MLI_AggrAMGSetThreshold                                  *
 * --------------------------------------------------------------------- */

int MLI_AggrAMGSetThreshold( MLI_AggrAMG *object, double thresh )
{
   if ( thresh > 0.0 ) object->threshold = thresh;
   else                object->threshold = 0.0;
   return 0;
}

/* ********************************************************************* *
 * Function   : MLI_AggrAMGSetPweight                                    *
 * --------------------------------------------------------------------- */

int MLI_AggrAMGSetPweight( MLI_AggrAMG *object, double weight )
{
   if ( weight >= 0.0 && weight <= 2.0 ) object->P_weight = weight;
   return 0;
}

/* ********************************************************************* *
 * Function   : MLI_AggrAMGSetCalcSpectralNorm                           *
 * --------------------------------------------------------------------- */

int ML_AggrAMGSetCalcSpectralNorm( MLI_AggrAMG *object )
{
   object->spectral_radius_scheme = 1;
   return 0;
}

/* ********************************************************************* *
 * Function   : MLI_AggrAMGSetNullSpace                                  *
 * --------------------------------------------------------------------- */

int ML_AggrAMGSetNullSpace( MLI_AggrAMG *object, int node_dofs,
                            int null_dim, double *null_vec, int leng )
{
   int i;

   if ( (null_vec == NULL) && (node_dofs != null_dim) )
   {
      printf("WARNING:  When no nullspace vector is specified, the nodal\n");
      printf("DOFS must be equal to the nullspace dimension.\n");
      null_dim = node_dofs;
   }
   object->node_dofs     = node_dofs;
   object->nullspace_dim = null_dim;
   if ( object->nullspace_vec != NULL ) free( object->nullspace_vec );
   if ( null_vect != NULL )
   {
      object->nullspace_vec = (double *) calloc(double, leng * null_dim );
      for ( i = 0; i < leng*null_dim; i++ )
         (object->nullspace_vec)[i] = null_vect[i];
   }
   else object->nullspace_vec = NULL;
   return 0;
}

/***********************************************************************
 * Function  : MLI_AggrAMGGenProlongators
 * Purpose   : create a prolongator matrix from Amat 
 * Inputs    : Amat (in Amat_array)
 **********************************************************************/

int MLI_AggrAMGGenProlongators(MLI_AggrAMG *mli_aggr, 
                               MLI_Matrix **Amat_array)
                               MLI_Matrix **Pmat_array)
{
   hypre_ParCSRMatrix *curr_Amat;
   int                coarsen_scheme;

   /* --------------------------------------------------------------- */
   /* fetch aggregation information                                   */
   /* --------------------------------------------------------------- */

   coarsen_scheme = mli_aggr->coarsen_scheme;
   nlevels        = mli_aggr->nlevels;
   curr_Amat      = (hypre_ParCSRMatrix *) Amat_array[nlevels-1]->matrix;
   assert( curr_Amat );

   /* --------------------------------------------------------------- */
   /* traverse all levels                                             */
   /* --------------------------------------------------------------- */

   for (level = nlevels-1; level > 0 ; level-- )
   {
      switch ( coarsen_scheme )
      {
         case MLI_AGGR_LOCAL :
              MLI_AMG_SA_GenPLocal(mli_aggr, Amat, Pmat); 
              break;
      }
      if ( Pmat == NULL ) break;
      hypre_BoomerAMGBuildCoarseOperator( Pmat, curr_Amat, Pmat, &cAmat );
      Pmat_array[i-1] = Pmat;
      Amat_array[i-1] = cAmat;
      curr_Amat = cAmat;
   }

   /* --------------------------------------------------------------- */
   /* return the coarsest grid level number                           */
   /* --------------------------------------------------------------- */

   return level;
}

/***********************************************************************
 * Function  : MLI_AMG_SA_FormGraph
 * Purpose   : Form graph from Amat (specific to aggregation scheme) 
 **********************************************************************/

void MLI_AMG_SA_FormGraph(MLI_Aggregation *mli_aggr, 
                          hypre_ParCSRMatrix *Amat;
                          hypre_ParCSRMatrix *graph)
{
   HYPRE_IJMatrix         IJGraph;
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

   assert( Amat == NULL );
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
    * return the graph and clean up
    *-----------------------------------------------------------------*/

   graph = HYPRE_IJMatrixGetLocalStorageType(IJGraph, HYPRE_PARCSR);
   free( col_ind );
   free( col_val );
   if ( threshold > 0.0 ) free( diag_data );
   return 0;
}

