#include "../struct_mv/struct_mv.h"

#ifndef HYPRE_structIJ_MV_HEADER
#define HYPRE_structIJ_MV_HEADER

struct hypre_StructIJMatrix_struct;
typedef struct hypre_StructIJMatrix_struct *HYPRE_StructIJMatrix;
struct hypre_StructIJVector_struct;
typedef struct hypre_StructIJVector_struct *HYPRE_StructIJVector;


/* HYPRE_structIJ_matrix.c */
int HYPRE_StructIJMatrixCreate( MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructStencil stencil , HYPRE_StructIJMatrix *matrix );
int HYPRE_StructIJMatrixDestroy( HYPRE_StructIJMatrix matrix );
int HYPRE_StructIJMatrixInitialize( HYPRE_StructIJMatrix matrix );
int HYPRE_StructIJMatrixAssemble( HYPRE_StructIJMatrix matrix );
int HYPRE_StructIJMatrixSetBoxValues( HYPRE_StructIJMatrix matrix , int *lower_grid_index , int *upper_grid_index , int num_stencil_indices , int *stencil_indices , double *coeffs );
int HYPRE_StructIJMatrixSetSymmetric( HYPRE_StructIJMatrix matrix , int symmetric );
void *HYPRE_StructIJMatrixGetLocalStorage( HYPRE_StructIJMatrix matrix );

/* HYPRE_structIJ_vector.c */
int HYPRE_StructIJVectorCreate( MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructStencil stencil , HYPRE_StructIJVector *vector );
int HYPRE_StructIJVectorDestroy( HYPRE_StructIJVector vector );
int HYPRE_StructIJVectorInitialize( HYPRE_StructIJVector vector );
int HYPRE_StructIJVectorAssemble( HYPRE_StructIJVector vector );
int HYPRE_StructIJVectorSetBoxValues( HYPRE_StructIJVector vector , int *lower_grid_index , int *upper_grid_index , double *coeffs );
void *HYPRE_StructIJVectorGetLocalStorage( HYPRE_StructIJVector in_vector );
int HYPRE_StructIJVectorSetPartitioning( HYPRE_StructIJVector vector , const int *partitioning );

#endif
