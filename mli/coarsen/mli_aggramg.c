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

