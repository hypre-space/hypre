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

void MLI_AMG_SA_GenProlongator(MLI_Aggregation *mli_aggr, 
                               hypre_ParCSRMatrix *Amat, 
                               hypre_ParCSRMatrix *Pmat)
{
   hypre_ParCSRMatrix *A2mat;
   int                blk_size;

   blk_size = mli_aggr->num_PDEs;
   MLI_Matrix_DoBlocking(Amat, A2mat, blksize);
   mli_aggr->matrix = A2mat;
   MLI_AMG_SA_FormGraph(mli_aggr);
   MLI_AMG_SA_CoarsenLocal(mli_aggr);
   hypre_ParCSRMatrixDestroy(A2mat);
   free( A2mat );
   mli_aggr->matrix = Amat;
   MLI_AMG_SA_GenPmatLocal(mli_aggr, Pmat);
   return;
}

