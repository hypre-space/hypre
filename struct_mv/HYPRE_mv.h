/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header file for HYPRE_mv library
 *
 *****************************************************************************/

#ifndef HYPRE_MV_HEADER
#define HYPRE_MV_HEADER

#include "HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HYPRE_USE_PTHREADS
#define NO_PTHREAD_MANGLING
#endif

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {int opaque;} *HYPRE_StructStencilBase;
typedef struct {int opaque;} *HYPRE_StructGridBase;
typedef struct {int opaque;} *HYPRE_StructMatrixBase;
typedef struct {int opaque;} *HYPRE_StructVectorBase;
typedef struct {int opaque;} *HYPRE_CommPkgBase;

#ifdef NO_PTHREAD_MANGLING
typedef HYPRE_StructStencilBase HYPRE_StructStencil;
typedef HYPRE_StructGridBase    HYPRE_StructGrid;
typedef HYPRE_StructMatrixBase  HYPRE_StructMatrix;
typedef HYPRE_StructVectorBase  HYPRE_StructVector;
typedef HYPRE_CommPkgBase       HYPRE_CommPkg;
#else
typedef HYPRE_StructStencilBase HYPRE_StructStencil[hypre_MAX_THREADS];
typedef HYPRE_StructGridBase    HYPRE_StructGrid[hypre_MAX_THREADS];
typedef HYPRE_StructMatrixBase  HYPRE_StructMatrix[hypre_MAX_THREADS];
typedef HYPRE_StructVectorBase  HYPRE_StructVector[hypre_MAX_THREADS];
typedef HYPRE_CommPkgBase       HYPRE_CommPkg[hypre_MAX_THREADS];
#endif

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* HYPRE_struct_grid.c */
int HYPRE_NewStructGrid P((MPI_Comm comm , int dim , HYPRE_StructGrid *grid ));
int HYPRE_FreeStructGrid P((HYPRE_StructGrid grid ));
int HYPRE_SetStructGridExtents P((HYPRE_StructGrid grid , int *ilower , int *iupper ));
int HYPRE_AssembleStructGrid P((HYPRE_StructGrid grid ));

/* HYPRE_struct_matrix.c */
int HYPRE_NewStructMatrix P((MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructStencil stencil , HYPRE_StructMatrix *matrix ));
int HYPRE_FreeStructMatrix P((HYPRE_StructMatrix matrix ));
int HYPRE_InitializeStructMatrix P((HYPRE_StructMatrix matrix ));
int HYPRE_SetStructMatrixValues P((HYPRE_StructMatrix matrix , int *grid_index , int num_stencil_indices , int *stencil_indices , double *values ));
int HYPRE_SetStructMatrixBoxValues P((HYPRE_StructMatrix matrix , int *ilower , int *iupper , int num_stencil_indices , int *stencil_indices , double *values ));
int HYPRE_AssembleStructMatrix P((HYPRE_StructMatrix matrix ));
int HYPRE_SetStructMatrixNumGhost P((HYPRE_StructMatrix matrix , int *num_ghost ));
HYPRE_StructGrid HYPRE_StructMatrixGrid P((HYPRE_StructMatrix matrix ));
int HYPRE_SetStructMatrixSymmetric P((HYPRE_StructMatrix matrix , int symmetric ));
int HYPRE_PrintStructMatrix P((char *filename , HYPRE_StructMatrix matrix , int all ));

/* HYPRE_struct_stencil.c */
int HYPRE_NewStructStencil P((int dim , int size , HYPRE_StructStencil *stencil ));
int HYPRE_SetStructStencilElement P((HYPRE_StructStencil stencil , int element_index , int *offset ));
int HYPRE_FreeStructStencil P((HYPRE_StructStencil stencil ));

/* HYPRE_struct_vector.c */
int HYPRE_NewStructVector P((MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructStencil stencil , HYPRE_StructVector *vector ));
int HYPRE_FreeStructVector P((HYPRE_StructVector struct_vector ));
int HYPRE_InitializeStructVector P((HYPRE_StructVector vector ));
int HYPRE_SetStructVectorValues P((HYPRE_StructVector vector , int *grid_index , double values ));
int HYPRE_GetStructVectorValues P((HYPRE_StructVector vector , int *grid_index , double *values_ptr ));
int HYPRE_SetStructVectorBoxValues P((HYPRE_StructVector vector , int *ilower , int *iupper , double *values ));
int HYPRE_GetStructVectorBoxValues P((HYPRE_StructVector vector , int *ilower , int *iupper , double **values_ptr ));
int HYPRE_AssembleStructVector P((HYPRE_StructVector vector ));
int HYPRE_PrintStructVector P((char *filename , HYPRE_StructVector vector , int all ));
int HYPRE_SetStructVectorNumGhost P((HYPRE_StructVector vector , int *num_ghost ));
int HYPRE_SetStructVectorConstantValues P((HYPRE_StructVector vector , double values ));
HYPRE_CommPkg HYPRE_GetMigrateStructVectorCommPkg P((HYPRE_StructVector from_vector , HYPRE_StructVector to_vector ));
int HYPRE_MigrateStructVector P((HYPRE_CommPkg comm_pkg , HYPRE_StructVector from_vector , HYPRE_StructVector to_vector ));
int HYPRE_FreeCommPkg P((HYPRE_CommPkg comm_pkg ));

#undef P
#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* thread_wrappers.c */
void HYPRE_NewStructGridVoidPtr P((void *argptr ));
int HYPRE_NewStructGridPush P((MPI_Comm comm , int dim , HYPRE_StructGrid *grid ));
void HYPRE_FreeStructGridVoidPtr P((void *argptr ));
int HYPRE_FreeStructGridPush P((HYPRE_StructGrid grid ));
void HYPRE_SetStructGridExtentsVoidPtr P((void *argptr ));
int HYPRE_SetStructGridExtentsPush P((HYPRE_StructGrid grid , int *ilower , int *iupper ));
void HYPRE_AssembleStructGridVoidPtr P((void *argptr ));
int HYPRE_AssembleStructGridPush P((HYPRE_StructGrid grid ));
void HYPRE_NewStructMatrixVoidPtr P((void *argptr ));
int HYPRE_NewStructMatrixPush P((MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructStencil stencil , HYPRE_StructMatrix *matrix ));
void HYPRE_FreeStructMatrixVoidPtr P((void *argptr ));
int HYPRE_FreeStructMatrixPush P((HYPRE_StructMatrix matrix ));
void HYPRE_InitializeStructMatrixVoidPtr P((void *argptr ));
int HYPRE_InitializeStructMatrixPush P((HYPRE_StructMatrix matrix ));
void HYPRE_SetStructMatrixValuesVoidPtr P((void *argptr ));
int HYPRE_SetStructMatrixValuesPush P((HYPRE_StructMatrix matrix , int *grid_index , int num_stencil_indices , int *stencil_indices , double *values ));
void HYPRE_SetStructMatrixBoxValuesVoidPtr P((void *argptr ));
int HYPRE_SetStructMatrixBoxValuesPush P((HYPRE_StructMatrix matrix , int *ilower , int *iupper , int num_stencil_indices , int *stencil_indices , double *values ));
void HYPRE_AssembleStructMatrixVoidPtr P((void *argptr ));
int HYPRE_AssembleStructMatrixPush P((HYPRE_StructMatrix matrix ));
void HYPRE_SetStructMatrixNumGhostVoidPtr P((void *argptr ));
int HYPRE_SetStructMatrixNumGhostPush P((HYPRE_StructMatrix matrix , int *num_ghost ));
void HYPRE_StructMatrixGridVoidPtr P((void *argptr ));
HYPRE_StructGrid HYPRE_StructMatrixGridPush P((HYPRE_StructMatrix matrix ));
void HYPRE_SetStructMatrixSymmetricVoidPtr P((void *argptr ));
int HYPRE_SetStructMatrixSymmetricPush P((HYPRE_StructMatrix matrix , int symmetric ));
void HYPRE_PrintStructMatrixVoidPtr P((void *argptr ));
int HYPRE_PrintStructMatrixPush P((char *filename , HYPRE_StructMatrix matrix , int all ));
void HYPRE_NewStructStencilVoidPtr P((void *argptr ));
int HYPRE_NewStructStencilPush P((int dim , int size , HYPRE_StructStencil *stencil ));
void HYPRE_SetStructStencilElementVoidPtr P((void *argptr ));
int HYPRE_SetStructStencilElementPush P((HYPRE_StructStencil stencil , int element_index , int *offset ));
void HYPRE_FreeStructStencilVoidPtr P((void *argptr ));
int HYPRE_FreeStructStencilPush P((HYPRE_StructStencil stencil ));
void HYPRE_NewStructVectorVoidPtr P((void *argptr ));
int HYPRE_NewStructVectorPush P((MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructStencil stencil , HYPRE_StructVector *vector ));
void HYPRE_FreeStructVectorVoidPtr P((void *argptr ));
int HYPRE_FreeStructVectorPush P((HYPRE_StructVector struct_vector ));
void HYPRE_InitializeStructVectorVoidPtr P((void *argptr ));
int HYPRE_InitializeStructVectorPush P((HYPRE_StructVector vector ));
void HYPRE_SetStructVectorValuesVoidPtr P((void *argptr ));
int HYPRE_SetStructVectorValuesPush P((HYPRE_StructVector vector , int *grid_index , double values ));
void HYPRE_GetStructVectorValuesVoidPtr P((void *argptr ));
int HYPRE_GetStructVectorValuesPush P((HYPRE_StructVector vector , int *grid_index , double *values_ptr ));
void HYPRE_SetStructVectorBoxValuesVoidPtr P((void *argptr ));
int HYPRE_SetStructVectorBoxValuesPush P((HYPRE_StructVector vector , int *ilower , int *iupper , double *values ));
void HYPRE_GetStructVectorBoxValuesVoidPtr P((void *argptr ));
int HYPRE_GetStructVectorBoxValuesPush P((HYPRE_StructVector vector , int *ilower , int *iupper , double **values_ptr ));
void HYPRE_AssembleStructVectorVoidPtr P((void *argptr ));
int HYPRE_AssembleStructVectorPush P((HYPRE_StructVector vector ));
void HYPRE_PrintStructVectorVoidPtr P((void *argptr ));
int HYPRE_PrintStructVectorPush P((char *filename , HYPRE_StructVector vector , int all ));
void HYPRE_SetStructVectorNumGhostVoidPtr P((void *argptr ));
int HYPRE_SetStructVectorNumGhostPush P((HYPRE_StructVector vector , int *num_ghost ));
void HYPRE_SetStructVectorConstantValuesVoidPtr P((void *argptr ));
int HYPRE_SetStructVectorConstantValuesPush P((HYPRE_StructVector vector , double values ));
void HYPRE_GetMigrateStructVectorCommPkgVoidPtr P((void *argptr ));
HYPRE_CommPkg HYPRE_GetMigrateStructVectorCommPkgPush P((HYPRE_StructVector from_vector , HYPRE_StructVector to_vector ));
void HYPRE_MigrateStructVectorVoidPtr P((void *argptr ));
int HYPRE_MigrateStructVectorPush P((HYPRE_CommPkg comm_pkg , HYPRE_StructVector from_vector , HYPRE_StructVector to_vector ));
void HYPRE_FreeCommPkgVoidPtr P((void *argptr ));
int HYPRE_FreeCommPkgPush P((HYPRE_CommPkg comm_pkg ));

#undef P

#ifdef __cplusplus
}
#endif

#endif
