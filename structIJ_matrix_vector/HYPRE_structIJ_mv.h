#include "../struct_matrix_vector/struct_matrix_vector.h"

#ifndef HYPRE_structIJ_MV_HEADER
#define HYPRE_structIJ_MV_HEADER

typedef struct {int opaque;} *HYPRE_StructIJMatrix;
typedef struct {int opaque;} *HYPRE_StructIJVector;

# define	P(s) s


/* HYPRE_structIJ_matrix.c */
int HYPRE_StructIJMatrixCreate P((MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructStencil stencil , HYPRE_StructIJMatrix *matrix ));
int HYPRE_StructIJMatrixDestroy P((HYPRE_StructIJMatrix matrix ));
int HYPRE_StructIJMatrixInitialize P((HYPRE_StructIJMatrix matrix ));
int HYPRE_StructIJMatrixAssemble P((HYPRE_StructIJMatrix matrix ));
int HYPRE_StructIJMatrixSetBoxValues P((HYPRE_StructIJMatrix matrix , int *lower_grid_index , int *upper_grid_index , int num_stencil_indices , int *stencil_indices , double *coeffs ));
int HYPRE_StructIJMatrixSetSymmetric P((HYPRE_StructIJMatrix matrix , int symmetric ));
void *HYPRE_StructIJMatrixGetLocalStorage P((HYPRE_StructIJMatrix matrix ));

/* HYPRE_structIJ_vector.c */
int HYPRE_StructIJVectorCreate P((MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructStencil stencil , HYPRE_StructIJVector *vector ));
int HYPRE_StructIJVectorDestroy P((HYPRE_StructIJVector vector ));
int HYPRE_StructIJVectorInitialize P((HYPRE_StructIJVector vector ));
int HYPRE_StructIJVectorAssemble P((HYPRE_StructIJVector vector ));
int HYPRE_StructIJVectorSetBoxValues P((HYPRE_StructIJVector vector , int *lower_grid_index , int *upper_grid_index , double *coeffs ));
void *HYPRE_StructIJVectorGetLocalStorage P((HYPRE_StructIJVector in_vector ));
int HYPRE_StructIJVectorSetPartitioning P((HYPRE_StructIJVector vector , const int *partitioning ));

#undef P

#endif
