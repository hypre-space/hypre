#include "../struct_mv/struct_mv.h"

#ifndef HYPRE_CI_MV_HEADER
#define HYPRE_CI_MV_HEADER

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void *HYPRE_StructInterfaceMatrix;
typedef void *HYPRE_StructInterfaceVector;

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

# define	P(s) s

/* HYPRE_struct_grid.c */
HYPRE_StructGrid HYPRE_NewStructGrid P((MPI_Comm context , int dim ));
void HYPRE_FreeStructGrid P((HYPRE_StructGrid grid ));
void HYPRE_SetStructGridExtents P((HYPRE_StructGrid grid , int *ilower , int *iupper ));
void HYPRE_AssembleStructGrid P((HYPRE_StructGrid grid ));

/* HYPRE_struct_matrix.c */
HYPRE_StructInterfaceMatrix HYPRE_NewStructInterfaceMatrix P((MPI_Comm context , HYPRE_StructGrid grid , HYPRE_StructStencil stencil ));
int HYPRE_FreeStructInterfaceMatrix P((HYPRE_StructInterfaceMatrix struct_matrix ));
int HYPRE_SetStructInterfaceMatrixCoeffs P((HYPRE_StructInterfaceMatrix matrix , int *grid_index , double *coeffs ));
int HYPRE_SetStructInterfaceMatrixBoxValues P((HYPRE_StructInterfaceMatrix matrix , int *lower_grid_index , int *upper_grid_index , int num_stencil_indices , int *stencil_indices , double *coeffs ));
int HYPRE_InitializeStructInterfaceMatrix P((HYPRE_StructInterfaceMatrix matrix ));
int HYPRE_AssembleStructInterfaceMatrix P((HYPRE_StructInterfaceMatrix matrix ));
void *HYPRE_StructInterfaceMatrixGetData P((HYPRE_StructInterfaceMatrix matrix ));
int HYPRE_PrintStructInterfaceMatrix P((HYPRE_StructInterfaceMatrix matrix ));
int HYPRE_SetStructInterfaceMatrixStorageType P((HYPRE_StructInterfaceMatrix struct_matrix , int type ));
int HYPRE_SetStructInterfaceMatrixSymmetric P((HYPRE_StructInterfaceMatrix struct_matrix , int type ));
int HYPRE_SetStructInterfaceMatrixNumGhost P((HYPRE_StructInterfaceMatrix struct_matrix , int *num_ghost ));

/* HYPRE_struct_stencil.c */
HYPRE_StructStencil HYPRE_NewStructStencil P((int dim , int size ));
void HYPRE_SetStructStencilElement P((HYPRE_StructStencil stencil , int element_index , int *offset ));
void HYPRE_FreeStructStencil P((HYPRE_StructStencil stencil ));

/* HYPRE_struct_vector.c */
HYPRE_StructInterfaceVector HYPRE_NewStructInterfaceVector P((MPI_Comm context , HYPRE_StructGrid grid , HYPRE_StructStencil stencil ));
int HYPRE_FreeStructInterfaceVector P((HYPRE_StructInterfaceVector struct_vector ));
int HYPRE_SetStructInterfaceVectorCoeffs P((HYPRE_StructInterfaceVector vector , int *grid_index , double *coeffs ));
int HYPRE_SetStructInterfaceVectorBoxValues P((HYPRE_StructInterfaceVector vector , int *lower_grid_index , int *upper_grid_index , double *coeffs ));
int HYPRE_SetStructInterfaceVector P((HYPRE_StructInterfaceVector vector , double *val ));
int HYPRE_InitializeStructInterfaceVector P((HYPRE_StructInterfaceVector vector ));
int HYPRE_AssembleStructInterfaceVector P((HYPRE_StructInterfaceVector vector ));
int HYPRE_SetStructInterfaceVectorStorageType P((HYPRE_StructInterfaceVector struct_vector , int type ));
void *HYPRE_StructInterfaceVectorGetData P((HYPRE_StructInterfaceVector vector ));
int HYPRE_PrintStructInterfaceVector P((HYPRE_StructInterfaceVector vector ));
int HYPRE_RetrievalOnStructInterfaceVector P((HYPRE_StructInterfaceVector vector ));
int HYPRE_RetrievalOffStructInterfaceVector P((HYPRE_StructInterfaceVector vector ));
int HYPRE_GetStructInterfaceVectorValue P((HYPRE_StructInterfaceVector vector , int *index , double *value ));

#undef P

#endif
