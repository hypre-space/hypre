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

#include <HYPRE_config.h>

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

#define HYPRE_StructGridCreate HYPRE_StructGridCreatePush
#define HYPRE_StructGridDestroy HYPRE_StructGridDestroyPush
#define HYPRE_StructGridSetExtents HYPRE_StructGridSetExtentsPush
#define HYPRE_StructGridSetPeriodic HYPRE_StructGridSetPeriodicPush
#define HYPRE_StructGridAssemble HYPRE_StructGridAssemblePush
#define HYPRE_StructMatrixCreate HYPRE_StructMatrixCreatePush
#define HYPRE_StructMatrixDestroy HYPRE_StructMatrixDestroyPush
#define HYPRE_StructMatrixInitialize HYPRE_StructMatrixInitializePush
#define HYPRE_StructMatrixSetValues HYPRE_StructMatrixSetValuesPush
#define HYPRE_StructMatrixSetBoxValues HYPRE_StructMatrixSetBoxValuesPush
#define HYPRE_StructMatrixAddToValues HYPRE_StructMatrixAddToValuesPush
#define HYPRE_StructMatrixAddToBoxValues HYPRE_StructMatrixAddToBoxValuesPush
#define HYPRE_StructMatrixAssemble HYPRE_StructMatrixAssemblePush
#define HYPRE_StructMatrixSetNumGhost HYPRE_StructMatrixSetNumGhostPush
#define HYPRE_StructMatrixGetGrid HYPRE_StructMatrixGetGridPush
#define HYPRE_StructMatrixSetSymmetric HYPRE_StructMatrixSetSymmetricPush
#define HYPRE_StructMatrixPrint HYPRE_StructMatrixPrintPush
#define HYPRE_StructStencilCreate HYPRE_StructStencilCreatePush
#define HYPRE_StructStencilSetElement HYPRE_StructStencilSetElementPush
#define HYPRE_StructStencilDestroy HYPRE_StructStencilDestroyPush
#define HYPRE_StructVectorCreate HYPRE_StructVectorCreatePush
#define HYPRE_StructVectorDestroy HYPRE_StructVectorDestroyPush
#define HYPRE_StructVectorInitialize HYPRE_StructVectorInitializePush
#define HYPRE_StructVectorSetValues HYPRE_StructVectorSetValuesPush
#define HYPRE_StructVectorSetBoxValues HYPRE_StructVectorSetBoxValuesPush
#define HYPRE_StructVectorAddToValues HYPRE_StructVectorAddToValuesPush
#define HYPRE_StructVectorAddToBoxValues HYPRE_StructVectorAddToBoxValuesPush
#define HYPRE_StructVectorGetValues HYPRE_StructVectorGetValuesPush
#define HYPRE_StructVectorGetBoxValues HYPRE_StructVectorGetBoxValuesPush
#define HYPRE_StructVectorAssemble HYPRE_StructVectorAssemblePush
#define HYPRE_StructVectorPrint HYPRE_StructVectorPrintPush
#define HYPRE_StructVectorSetNumGhost HYPRE_StructVectorSetNumGhostPush
#define HYPRE_StructVectorSetConstantValues HYPRE_StructVectorSetConstantValuesPush
#define HYPRE_StructVectorGetMigrateCommPkg HYPRE_StructVectorGetMigrateCommPkgPush
#define HYPRE_StructVectorMigrate HYPRE_StructVectorMigratePush
#define HYPRE_CommPkgDestroy HYPRE_CommPkgDestroyPush

#endif


/* HYPRE_struct_grid.c */
int HYPRE_StructGridCreate( MPI_Comm comm , int dim , HYPRE_StructGrid *grid );
int HYPRE_StructGridDestroy( HYPRE_StructGrid grid );
int HYPRE_StructGridSetExtents( HYPRE_StructGrid grid , int *ilower , int *iupper );
int HYPRE_StructGridSetPeriodic( HYPRE_StructGrid grid , int *periodic );
int HYPRE_StructGridAssemble( HYPRE_StructGrid grid );

/* HYPRE_struct_matrix.c */
int HYPRE_StructMatrixCreate( MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructStencil stencil , HYPRE_StructMatrix *matrix );
int HYPRE_StructMatrixDestroy( HYPRE_StructMatrix matrix );
int HYPRE_StructMatrixInitialize( HYPRE_StructMatrix matrix );
int HYPRE_StructMatrixSetValues( HYPRE_StructMatrix matrix , int *grid_index , int num_stencil_indices , int *stencil_indices , double *values );
int HYPRE_StructMatrixSetBoxValues( HYPRE_StructMatrix matrix , int *ilower , int *iupper , int num_stencil_indices , int *stencil_indices , double *values );
int HYPRE_StructMatrixAddToValues( HYPRE_StructMatrix matrix , int *grid_index , int num_stencil_indices , int *stencil_indices , double *values );
int HYPRE_StructMatrixAddToBoxValues( HYPRE_StructMatrix matrix , int *ilower , int *iupper , int num_stencil_indices , int *stencil_indices , double *values );
int HYPRE_StructMatrixAssemble( HYPRE_StructMatrix matrix );
int HYPRE_StructMatrixSetNumGhost( HYPRE_StructMatrix matrix , int *num_ghost );
int HYPRE_StructMatrixGetGrid( HYPRE_StructMatrix matrix , HYPRE_StructGrid *grid );
int HYPRE_StructMatrixSetSymmetric( HYPRE_StructMatrix matrix , int symmetric );
int HYPRE_StructMatrixPrint( char *filename , HYPRE_StructMatrix matrix , int all );

/* HYPRE_struct_stencil.c */
int HYPRE_StructStencilCreate( int dim , int size , HYPRE_StructStencil *stencil );
int HYPRE_StructStencilSetElement( HYPRE_StructStencil stencil , int element_index , int *offset );
int HYPRE_StructStencilDestroy( HYPRE_StructStencil stencil );

/* HYPRE_struct_vector.c */
int HYPRE_StructVectorCreate( MPI_Comm comm , HYPRE_StructGrid grid , HYPRE_StructStencil stencil , HYPRE_StructVector *vector );
int HYPRE_StructVectorDestroy( HYPRE_StructVector struct_vector );
int HYPRE_StructVectorInitialize( HYPRE_StructVector vector );
int HYPRE_StructVectorSetValues( HYPRE_StructVector vector , int *grid_index , double values );
int HYPRE_StructVectorSetBoxValues( HYPRE_StructVector vector , int *ilower , int *iupper , double *values );
int HYPRE_StructVectorAddToValues( HYPRE_StructVector vector , int *grid_index , double values );
int HYPRE_StructVectorAddToBoxValues( HYPRE_StructVector vector , int *ilower , int *iupper , double *values );
int HYPRE_StructVectorGetValues( HYPRE_StructVector vector , int *grid_index , double *values_ptr );
int HYPRE_StructVectorGetBoxValues( HYPRE_StructVector vector , int *ilower , int *iupper , double *values );
int HYPRE_StructVectorAssemble( HYPRE_StructVector vector );
int HYPRE_StructVectorPrint( char *filename , HYPRE_StructVector vector , int all );
int HYPRE_StructVectorSetNumGhost( HYPRE_StructVector vector , int *num_ghost );
int HYPRE_StructVectorSetConstantValues( HYPRE_StructVector vector , double values );
int HYPRE_StructVectorGetMigrateCommPkg( HYPRE_StructVector from_vector , HYPRE_StructVector to_vector , HYPRE_CommPkg *comm_pkg );
int HYPRE_StructVectorMigrate( HYPRE_CommPkg comm_pkg , HYPRE_StructVector from_vector , HYPRE_StructVector to_vector );
int HYPRE_CommPkgDestroy( HYPRE_CommPkg comm_pkg );

#ifdef __cplusplus
}
#endif

#endif



