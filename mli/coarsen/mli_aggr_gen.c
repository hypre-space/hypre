/***********************************************************************
 *
 **********************************************************************/

#include "parcsr_mv.h"
#include "mli_matrix.h"
#include "mli_aggregation.h"
 
/***********************************************************************
 * Function  : MLI_AMG_SA_GenProlongator
 * Purpose   : create a prolongator matrix from Amat 
 * Inputs    : Amat, aggregation information 
 **********************************************************************/

int MLI_AMG_SA_GenAllProlongator(MLI_Aggregation *mli_aggr, 
                                 hypre_ParCSRMatrix **Amat_array)
                                 hypre_ParCSRMatrix **Pmat_array)
{
   hypre_ParCSRMatrix *curr_Amat;
   int                coarsen_scheme;

   /* --------------------------------------------------------------- */
   /* fetch aggregation information                                   */
   /* --------------------------------------------------------------- */

   coarsen_scheme = mli_aggr->coarsen_scheme;
   nlevels        = mli_aggr->nlevels;
   curr_Amat      = Amat_array[nlevels-1];
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

