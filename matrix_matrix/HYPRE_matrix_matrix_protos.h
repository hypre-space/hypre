
#include "HYPRE_distributed_matrix_types.h"

#ifdef PETSC_AVAILABLE
/* HYPRE_ConvertPETScMatrixToDistributedMatrix.c */
int HYPRE_ConvertPETScMatrixToDistributedMatrix (Mat PETSc_matrix , HYPRE_DistributedMatrix *DistributedMatrix );
#endif

/* HYPRE_ConvertParCSRMatrixToDistributedMatrix.c */
int HYPRE_ConvertParCSRMatrixToDistributedMatrix (HYPRE_ParCSRMatrix parcsr_matrix , HYPRE_DistributedMatrix *DistributedMatrix );

#include "HYPRE_IJ_mv.h"
/* HYPRE_BuildIJMatrixFromDistributedMatrix.c */
int HYPRE_BuildIJMatrixFromDistributedMatrix (HYPRE_DistributedMatrix DistributedMatrix, HYPRE_IJMatrix *ij_matrix );
