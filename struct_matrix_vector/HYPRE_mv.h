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

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {int opaque;} *HYPRE_StructStencilBase;
typedef struct {int opaque;} *HYPRE_StructGridBase;
typedef struct {int opaque;} *HYPRE_StructMatrixBase;
typedef struct {int opaque;} *HYPRE_StructVectorBase;
typedef struct {int opaque;} *HYPRE_CommPkgBase;

#ifndef HYPRE_USE_PTHREADS
#define hypre_MAX_THREADS 1
#ifndef HYPRE_NO_PTHREAD_MANGLING
#define HYPRE_NO_PTHREAD_MANGLING
#endif
#endif

typedef HYPRE_StructStencilBase   HYPRE_StructStencilArray[hypre_MAX_THREADS];
typedef HYPRE_StructGridBase      HYPRE_StructGridArray[hypre_MAX_THREADS];
typedef HYPRE_StructMatrixBase    HYPRE_StructMatrixArray[hypre_MAX_THREADS];
typedef HYPRE_StructVectorBase    HYPRE_StructVectorArray[hypre_MAX_THREADS];
typedef HYPRE_CommPkgBase         HYPRE_CommPkgArray[hypre_MAX_THREADS];


#ifdef HYPRE_NO_PTHREAD_MANGLING
typedef HYPRE_StructStencilBase  HYPRE_StructStencil;
typedef HYPRE_StructGridBase     HYPRE_StructGrid;
typedef HYPRE_StructMatrixBase   HYPRE_StructMatrix;
typedef HYPRE_StructVectorBase   HYPRE_StructVector;
typedef HYPRE_CommPkgBase        HYPRE_CommPkg;
#else
typedef HYPRE_StructStencilArray HYPRE_StructStencil;
typedef HYPRE_StructGridArray    HYPRE_StructGrid;
typedef HYPRE_StructMatrixArray  HYPRE_StructMatrix;
typedef HYPRE_StructVectorArray  HYPRE_StructVector;
typedef HYPRE_CommPkgArray       HYPRE_CommPkg;
#endif

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_NO_PTHREAD_MANGLING

#define HYPRE_NewStructGrid HYPRE_NewStructGridPush
#define HYPRE_FreeStructGrid HYPRE_FreeStructGridPush
#define HYPRE_SetStructGridExtents HYPRE_SetStructGridExtentsPush
#define HYPRE_SetStructGridPeriodic HYPRE_SetStructGridPeriodicPush
#define HYPRE_AssembleStructGrid HYPRE_AssembleStructGridPush
#define HYPRE_NewStructMatrix HYPRE_NewStructMatrixPush
#define HYPRE_FreeStructMatrix HYPRE_FreeStructMatrixPush
#define HYPRE_InitializeStructMatrix HYPRE_InitializeStructMatrixPush
#define HYPRE_SetStructMatrixValues HYPRE_SetStructMatrixValuesPush
#define HYPRE_SetStructMatrixBoxValues HYPRE_SetStructMatrixBoxValuesPush
#define HYPRE_AssembleStructMatrix HYPRE_AssembleStructMatrixPush
#define HYPRE_SetStructMatrixNumGhost HYPRE_SetStructMatrixNumGhostPush
#define HYPRE_StructMatrixGrid HYPRE_StructMatrixGridPush
#define HYPRE_SetStructMatrixSymmetric HYPRE_SetStructMatrixSymmetricPush
#define HYPRE_PrintStructMatrix HYPRE_PrintStructMatrixPush
#define HYPRE_NewStructStencil HYPRE_NewStructStencilPush
#define HYPRE_SetStructStencilElement HYPRE_SetStructStencilElementPush
#define HYPRE_FreeStructStencil HYPRE_FreeStructStencilPush
#define HYPRE_NewStructVector HYPRE_NewStructVectorPush
#define HYPRE_FreeStructVector HYPRE_FreeStructVectorPush
#define HYPRE_InitializeStructVector HYPRE_InitializeStructVectorPush
#define HYPRE_SetStructVectorValues HYPRE_SetStructVectorValuesPush
#define HYPRE_GetStructVectorValues HYPRE_GetStructVectorValuesPush
#define HYPRE_SetStructVectorBoxValues HYPRE_SetStructVectorBoxValuesPush
#define HYPRE_GetStructVectorBoxValues HYPRE_GetStructVectorBoxValuesPush
#define HYPRE_AssembleStructVector HYPRE_AssembleStructVectorPush
#define HYPRE_PrintStructVector HYPRE_PrintStructVectorPush
#define HYPRE_SetStructVectorNumGhost HYPRE_SetStructVectorNumGhostPush
#define HYPRE_SetStructVectorConstantValues HYPRE_SetStructVectorConstantValuesPush
#define HYPRE_GetMigrateStructVectorCommPkg HYPRE_GetMigrateStructVectorCommPkgPush
#define HYPRE_MigrateStructVector HYPRE_MigrateStructVectorPush
#define HYPRE_FreeCommPkg HYPRE_FreeCommPkgPush

#endif

# define	P(s) s

/* HYPRE_struct_grid.c */
int HYPRE_NewStructGrid P((MPI_Comm comm , int dim , HYPRE_StructGrid *grid ));
int HYPRE_FreeStructGrid P((HYPRE_StructGrid grid ));
int HYPRE_SetStructGridExtents P((HYPRE_StructGrid grid , int *ilower , int *iupper ));
int HYPRE_SetStructGridPeriodic P((HYPRE_StructGrid grid , int *periodic ));
int HYPRE_AssembleStructGrid P((HYPRE_StructGrid grid ));

/* HYPRE_struct_matrix.c */
int HYPRE_NewStructMatrix P((MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructStencil stencil , HYPRE_StructMatrix *matrix ));
int HYPRE_FreeStructMatrix P((HYPRE_StructMatrix matrix ));
int HYPRE_InitializeStructMatrix P((HYPRE_StructMatrix matrix ));
int HYPRE_SetStructMatrixValues P((HYPRE_StructMatrix matrix , int *grid_index , int num_stencil_indices , int *stencil_indices , double *values ));
int HYPRE_SetStructMatrixBoxValues P((HYPRE_StructMatrix matrix , int *ilower , int *iupper , int num_stencil_indices , int *stencil_indices , double *values ));
int HYPRE_AssembleStructMatrix P((HYPRE_StructMatrix matrix ));
int HYPRE_SetStructMatrixNumGhost P((HYPRE_StructMatrix matrix , int *num_ghost ));
int HYPRE_StructMatrixGrid P((HYPRE_StructMatrix matrix , HYPRE_StructGrid *grid ));
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
int HYPRE_GetStructVectorBoxValues P((HYPRE_StructVector vector , int *ilower , int *iupper , double *values ));
int HYPRE_AssembleStructVector P((HYPRE_StructVector vector ));
int HYPRE_PrintStructVector P((char *filename , HYPRE_StructVector vector , int all ));
int HYPRE_SetStructVectorNumGhost P((HYPRE_StructVector vector , int *num_ghost ));
int HYPRE_SetStructVectorConstantValues P((HYPRE_StructVector vector , double values ));
int HYPRE_GetMigrateStructVectorCommPkg P((HYPRE_StructVector from_vector , HYPRE_StructVector to_vector , HYPRE_CommPkg *comm_pkg ));
int HYPRE_MigrateStructVector P((HYPRE_CommPkg comm_pkg , HYPRE_StructVector from_vector , HYPRE_StructVector to_vector ));
int HYPRE_FreeCommPkg P((HYPRE_CommPkg comm_pkg ));

#undef P

#ifdef __cplusplus
}
#endif

#endif



