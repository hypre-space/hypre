#include "mli_include.h"

main()
{

   int                max_levels=40;
   hypre_ParCSRMatrix *hypre_Amat;
   MPI_Comm           mpi_comm;

   MLI         *mli;
   MLI_Matrix  *mli_mat;
   MLI_AggrAMG *mli_sa;

   mli_mat = MLI_Matrix_Create(hypre_Amat, "HYPRE ParCSR", NULL );
   mli_sa  = MLI_AggrAMGCreate();
   MLI_AggrAMGSetThreshold( mli_sa, 0.08 );
   MLI_AggrAMGSetNullSpace( mli_sa, 3, 3, NULL, 0 );
   MLI_Create( mli, mpi_comm, max_levels );
   MLI_SetAmat( mli, max_levels-1, mli_mat );
   MLI_SetMethod( mli, MLI_AGGRAMG, mli_sa );
   MLI_Setup( mli );
}

