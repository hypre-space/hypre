# define	P(s) s

#include "../distributed_matrix/HYPRE_distributed_matrix_types.h"

#ifdef PETSC_AVAILABLE
/* HYPRE_ConvertPETScMatrixToDistributedMatrix.c */
int HYPRE_ConvertPETScMatrixToDistributedMatrix P((Mat PETSc_matrix , HYPRE_DistributedMatrix *DistributedMatrix ));
#endif

/* HYPRE_ConvertParCSRMatrixToDistributedMatrix.c */
int HYPRE_ConvertParCSRMatrixToDistributedMatrix P((HYPRE_ParCSRMatrix parcsr_matrix , HYPRE_DistributedMatrix *DistributedMatrix ));

#include "../IJ_matrix_vector/HYPRE_IJMatrix.h
/* HYPRE_BuildIJMatrixFromDistributedMatrix.c */
int HYPRE_BuildIJMatrixFromDistributedMatrix P((HYPRE_DistributedMatrix DistributedMatrix, HYPRE_IJMatrix *ij_matrix ));
#undef P
