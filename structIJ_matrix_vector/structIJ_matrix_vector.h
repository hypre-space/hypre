#include "HYPRE_structIJ_mv.h"
#include "HYPRE_IJ_mv.h"
#include "../IJ_matrix_vector/IJ_matrix_vector.h"
#include "../CI_struct_matrix_vector/CI_struct_matrix_vector.h"
#include "utilities.h"

#ifndef hypre_structIJ_MV_HEADER
#define hypre_structIJ_MV_HEADER

#include "HYPRE.h"

/******************************************************************************
 *
 * Header info for the hypre_StructIJMatrix structures
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_StructIJMatrix:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm              comm;
   hypre_StructGrid     *grid;
   hypre_StructStencil  *stencil;
   int                   symmetric;
   void     	        *translator;  /* holds GridToCoord table */
   HYPRE_IJMatrix        IJmatrix;
   int                   ref_count;

} hypre_StructIJMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructIJMatrix
 *--------------------------------------------------------------------------*/

#define hypre_StructIJMatrixComm(matrix)        ((matrix) -> comm)
#define hypre_StructIJMatrixGrid(matrix)        ((matrix) -> grid)
#define hypre_StructIJMatrixStencil(matrix)     ((matrix) -> stencil)
#define hypre_StructIJMatrixSymmetric(matrix)   ((matrix) -> symmetric)
#define hypre_StructIJMatrixTranslator(matrix)  ((matrix) -> translator)
#define hypre_StructIJMatrixIJMatrix(matrix)    ((matrix) -> IJmatrix)
#define hypre_StructIJMatrixRefCount(matrix)    ((matrix) -> ref_count)


/******************************************************************************
 *
 * Header info for the hypre_StructIJVector structures
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_StructIJVector:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm             comm;
   hypre_StructGrid    *grid;
   hypre_StructStencil *stencil;
   HYPRE_IJVector       IJvector;
   void     	       *translator;  /* holds GridToCoord table */
   int                  ref_count;

} hypre_StructIJVector;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_StructIJVector
 *--------------------------------------------------------------------------*/

#define hypre_StructIJVectorComm(vector)      ((vector) -> comm)
#define hypre_StructIJVectorGrid(vector)      ((vector) -> grid)
#define hypre_StructIJVectorStencil(vector)   ((vector) -> stencil)
#define hypre_StructIJVectorIJVector(vector)  ((vector) -> IJvector)
#define hypre_StructIJVectorTranslator(vector)  ((vector) -> translator)
#define hypre_StructIJVectorRefCount(vector)  ((vector) -> ref_count)

# define P(s) ()

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

/* structIJ_matrix.c */
hypre_StructIJMatrix *hypre_StructIJMatrixCreate P((MPI_Comm comm , hypre_StructGrid *grid , hypre_StructStencil *stencil ));
int hypre_StructIJMatrixDestroy P((hypre_StructIJMatrix *matrix ));
int hypre_StructIJMatrixInitialize P((hypre_StructIJMatrix *matrix ));
int hypre_StructIJMatrixAssemble P((hypre_StructIJMatrix *matrix ));
int hypre_StructIJMatrixSetBoxValues P((hypre_StructIJMatrix *matrix , hypre_Index lower_grid_index , hypre_Index upper_grid_index , int num_stencil_indices , int *stencil_indices , double *coeffs ));
int hypre_StructIJMatrixSetValues P((hypre_StructIJMatrix *matrix , hypre_Index index , int num_stencil_indices , int *stencil_indices , double *coeffs ));

/* structIJ_vector.c */
hypre_StructIJVector *hypre_StructIJVectorCreate P((MPI_Comm comm , hypre_StructGrid *grid , hypre_StructStencil *stencil ));
int hypre_StructIJVectorDestroy P((hypre_StructIJVector *vector ));
int hypre_StructIJVectorInitialize P((hypre_StructIJVector *vector ));
int hypre_StructIJVectorSetBoxValues P((hypre_StructIJVector *vector , hypre_Index lower_grid_index , hypre_Index upper_grid_index , double *coeffs ));
int hypre_StructIJVectorSetValue P((hypre_StructIJVector *vector , hypre_Index index , double value ));
int hypre_StructIJVectorAssemble P((hypre_StructIJVector *vector ));

#undef P

#endif
