
#include "HYPRE_distributed_matrix_types.h"
# define	P(s) s

#ifdef PETSC_AVAILABLE
/* HYPRE_ConvertPETScMatrixToDistributedMatrix.c */
int HYPRE_ConvertPETScMatrixToDistributedMatrix P((Mat PETSc_matrix , HYPRE_DistributedMatrix *DistributedMatrix ));
#endif

/* HYPRE_ConvertParCSRMatrixToDistributedMatrix.c */
int HYPRE_ConvertParCSRMatrixToDistributedMatrix P((HYPRE_ParCSRMatrix parcsr_matrix , HYPRE_DistributedMatrix *DistributedMatrix ));

#include "HYPRE_IJ_mv.h"
# define	P(s) s
/* HYPRE_BuildIJMatrixFromDistributedMatrix.c */
int HYPRE_BuildIJMatrixFromDistributedMatrix P((HYPRE_DistributedMatrix DistributedMatrix, HYPRE_IJMatrix *ij_matrix ));
#undef P
