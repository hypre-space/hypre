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
typedef HYPRE_StructStencilBase HYPRE_StructStencil[NUM_THREADS];
typedef HYPRE_StructGridBase    HYPRE_StructGrid[NUM_THREADS];
typedef HYPRE_StructMatrixBase  HYPRE_StructMatrix[NUM_THREADS];
typedef HYPRE_StructVectorBase  HYPRE_StructVector[NUM_THREADS];
typedef HYPRE_CommPkgBase       HYPRE_CommPkg[NUM_THREADS];
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
HYPRE_StructGrid HYPRE_NewStructGrid P((MPI_Comm comm , int dim ));
void HYPRE_FreeStructGrid P((HYPRE_StructGrid grid ));
void HYPRE_SetStructGridExtents P((HYPRE_StructGrid grid , int *ilower , int *iupper ));
void HYPRE_AssembleStructGrid P((HYPRE_StructGrid grid ));

/* HYPRE_struct_matrix.c */
HYPRE_StructMatrix HYPRE_NewStructMatrix P((MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructStencil stencil ));
int HYPRE_FreeStructMatrix P((HYPRE_StructMatrix matrix ));
int HYPRE_InitializeStructMatrix P((HYPRE_StructMatrix matrix ));
int HYPRE_SetStructMatrixValues P((HYPRE_StructMatrix matrix , int *grid_index , int num_stencil_indices , int *stencil_indices , double *values ));
int HYPRE_SetStructMatrixBoxValues P((HYPRE_StructMatrix matrix , int *ilower , int *iupper , int num_stencil_indices , int *stencil_indices , double *values ));
int HYPRE_AssembleStructMatrix P((HYPRE_StructMatrix matrix ));
void HYPRE_SetStructMatrixNumGhost P((HYPRE_StructMatrix matrix , int *num_ghost ));
HYPRE_StructGrid HYPRE_StructMatrixGrid P((HYPRE_StructMatrix matrix ));
void HYPRE_SetStructMatrixSymmetric P((HYPRE_StructMatrix matrix , int symmetric ));
void HYPRE_PrintStructMatrix P((char *filename , HYPRE_StructMatrix matrix , int all ));

/* HYPRE_struct_stencil.c */
HYPRE_StructStencil HYPRE_NewStructStencil P((int dim , int size ));
void HYPRE_SetStructStencilElement P((HYPRE_StructStencil stencil , int element_index , int *offset ));
void HYPRE_FreeStructStencil P((HYPRE_StructStencil stencil ));

/* HYPRE_struct_vector.c */
HYPRE_StructVector HYPRE_NewStructVector P((MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructStencil stencil ));
int HYPRE_FreeStructVector P((HYPRE_StructVector struct_vector ));
int HYPRE_InitializeStructVector P((HYPRE_StructVector vector ));
int HYPRE_SetStructVectorValues P((HYPRE_StructVector vector , int *grid_index , double values ));
int HYPRE_GetStructVectorValues P((HYPRE_StructVector vector , int *grid_index , double *values_ptr ));
int HYPRE_SetStructVectorBoxValues P((HYPRE_StructVector vector , int *ilower , int *iupper , double *values ));
int HYPRE_GetStructVectorBoxValues P((HYPRE_StructVector vector , int *ilower , int *iupper , double **values_ptr ));
int HYPRE_AssembleStructVector P((HYPRE_StructVector vector ));
void HYPRE_PrintStructVector P((char *filename , HYPRE_StructVector vector , int all ));
void HYPRE_SetStructVectorNumGhost P((HYPRE_StructVector vector , int *num_ghost ));
int HYPRE_SetStructVectorConstantValues P((HYPRE_StructVector vector , double values ));
HYPRE_CommPkg HYPRE_GetMigrateStructVectorCommPkg P((HYPRE_StructVector from_vector , HYPRE_StructVector to_vector ));
int HYPRE_MigrateStructVector P((HYPRE_CommPkg comm_pkg , HYPRE_StructVector from_vector , HYPRE_StructVector to_vector ));
void HYPRE_FreeCommPkg P((HYPRE_CommPkg comm_pkg ));

#ifdef HYPRE_USE_PTHREADS
/* HYPRE_push_voidptr.c */
void HYPRE_InitializeStructMatrixVoidPtr P((void *argptr ));
int HYPRE_InitializeStructMatrixPush P((HYPRE_StructMatrix matrix ));
void HYPRE_SetStructMatrixBoxValuesVoidPtr P((void *argptr ));
int HYPRE_SetStructMatrixBoxValuesPush P((HYPRE_StructMatrix matrix , int *ilower , int *iupper , int num_stencil_indices , int *stencil_indices , double *values ));
void HYPRE_PrintStructMatrixVoidPtr P((void *argptr ));
void HYPRE_PrintStructMatrixPush P((char *filename , HYPRE_StructMatrix matrix , int all ));
void HYPRE_SetStructVectorBoxValuesVoidPtr P((void *argptr ));
int HYPRE_SetStructVectorBoxValuesPush P((HYPRE_StructVector vector , int *ilower , int *iupper , double *values ));
void HYPRE_GetStructVectorBoxValuesVoidPtr P((void *argptr ));
int HYPRE_GetStructVectorBoxValuesPush P((HYPRE_StructVector vector , int *ilower , int *iupper , double **values_ptr ));
void HYPRE_SetStructVectorConstantValuesVoidPtr P((void *argptr ));
int HYPRE_SetStructVectorConstantValuesPush P((HYPRE_StructVector vector , double values ));

#ifndef NO_PTHREAD_MANGLING
#define HYPRE_InitializeStructMatrix HYPRE_InitializeStructMatrixPush
#define HYPRE_SetStructMatrixBoxValues HYPRE_SetStructMatrixBoxValuesPush
#define HYPRE_PrintStructMatrix HYPRE_PrintStructMatrixPush
#define HYPRE_SetStructVectorBoxValues HYPRE_SetStructVectorBoxValuesPush
#define HYPRE_GetStructVectorBoxValues HYPRE_GetStructVectorBoxValuesPush
#define HYPRE_SetStructVectorConstantValues HYPRE_SetStructVectorConstantValuesPush
#endif
#endif


#undef P

#ifdef __cplusplus
}
#endif

#endif
