
#include "HYPRE_mv.h"
#include "utilities.h"

#ifdef HYPRE_USE_PTHREADS


/*----------------------------------------------------------------
 * HYPRE_StructGridCreate thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   MPI_Comm comm;
   int dim;
   HYPRE_StructGridArray *grid;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructGridCreateArgs;

void
HYPRE_StructGridCreateVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructGridCreateArgs *localargs =
      (HYPRE_StructGridCreateArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructGridCreate(
         localargs -> comm,
         localargs -> dim,
         &(*(localargs -> grid))[threadid] );
}

int 
HYPRE_StructGridCreatePush(
   MPI_Comm comm,
   int dim,
   HYPRE_StructGridArray *grid )
{
   HYPRE_StructGridCreateArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.comm = comm;
   pushargs.dim = dim;
   pushargs.grid = (HYPRE_StructGridArray *)grid;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructGridCreateVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructGridDestroy thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructGridArray *grid;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructGridDestroyArgs;

void
HYPRE_StructGridDestroyVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructGridDestroyArgs *localargs =
      (HYPRE_StructGridDestroyArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructGridDestroy(
         (*(localargs -> grid))[threadid] );
}

int 
HYPRE_StructGridDestroyPush(
   HYPRE_StructGridArray grid )
{
   HYPRE_StructGridDestroyArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.grid = (HYPRE_StructGridArray *)grid;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructGridDestroyVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructGridSetExtents thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructGridArray *grid;
   int *ilower;
   int *iupper;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructGridSetExtentsArgs;

void
HYPRE_StructGridSetExtentsVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructGridSetExtentsArgs *localargs =
      (HYPRE_StructGridSetExtentsArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructGridSetExtents(
         (*(localargs -> grid))[threadid],
         localargs -> ilower,
         localargs -> iupper );
}

int 
HYPRE_StructGridSetExtentsPush(
   HYPRE_StructGridArray grid,
   int *ilower,
   int *iupper )
{
   HYPRE_StructGridSetExtentsArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.grid = (HYPRE_StructGridArray *)grid;
   pushargs.ilower = ilower;
   pushargs.iupper = iupper;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructGridSetExtentsVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructGridSetPeriodic thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructGridArray *grid;
   int *periodic;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructGridSetPeriodicArgs;

void
HYPRE_StructGridSetPeriodicVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructGridSetPeriodicArgs *localargs =
      (HYPRE_StructGridSetPeriodicArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructGridSetPeriodic(
         (*(localargs -> grid))[threadid],
         localargs -> periodic );
}

int 
HYPRE_StructGridSetPeriodicPush(
   HYPRE_StructGridArray grid,
   int *periodic )
{
   HYPRE_StructGridSetPeriodicArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.grid = (HYPRE_StructGridArray *)grid;
   pushargs.periodic = periodic;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructGridSetPeriodicVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructGridAssemble thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructGridArray *grid;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructGridAssembleArgs;

void
HYPRE_StructGridAssembleVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructGridAssembleArgs *localargs =
      (HYPRE_StructGridAssembleArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructGridAssemble(
         (*(localargs -> grid))[threadid] );
}

int 
HYPRE_StructGridAssemblePush(
   HYPRE_StructGridArray grid )
{
   HYPRE_StructGridAssembleArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.grid = (HYPRE_StructGridArray *)grid;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructGridAssembleVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructMatrixCreate thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   MPI_Comm comm;
   HYPRE_StructGridArray *grid;
   HYPRE_StructStencilArray *stencil;
   HYPRE_StructMatrixArray *matrix;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructMatrixCreateArgs;

void
HYPRE_StructMatrixCreateVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructMatrixCreateArgs *localargs =
      (HYPRE_StructMatrixCreateArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructMatrixCreate(
         localargs -> comm,
         (*(localargs -> grid))[threadid],
         (*(localargs -> stencil))[threadid],
         &(*(localargs -> matrix))[threadid] );
}

int 
HYPRE_StructMatrixCreatePush(
   MPI_Comm comm,
   HYPRE_StructGridArray grid,
   HYPRE_StructStencilArray stencil,
   HYPRE_StructMatrixArray *matrix )
{
   HYPRE_StructMatrixCreateArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.comm = comm;
   pushargs.grid = (HYPRE_StructGridArray *)grid;
   pushargs.stencil = (HYPRE_StructStencilArray *)stencil;
   pushargs.matrix = (HYPRE_StructMatrixArray *)matrix;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructMatrixCreateVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructMatrixDestroy thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructMatrixArray *matrix;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructMatrixDestroyArgs;

void
HYPRE_StructMatrixDestroyVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructMatrixDestroyArgs *localargs =
      (HYPRE_StructMatrixDestroyArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructMatrixDestroy(
         (*(localargs -> matrix))[threadid] );
}

int 
HYPRE_StructMatrixDestroyPush(
   HYPRE_StructMatrixArray matrix )
{
   HYPRE_StructMatrixDestroyArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.matrix = (HYPRE_StructMatrixArray *)matrix;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructMatrixDestroyVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructMatrixInitialize thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructMatrixArray *matrix;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructMatrixInitializeArgs;

void
HYPRE_StructMatrixInitializeVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructMatrixInitializeArgs *localargs =
      (HYPRE_StructMatrixInitializeArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructMatrixInitialize(
         (*(localargs -> matrix))[threadid] );
}

int 
HYPRE_StructMatrixInitializePush(
   HYPRE_StructMatrixArray matrix )
{
   HYPRE_StructMatrixInitializeArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.matrix = (HYPRE_StructMatrixArray *)matrix;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructMatrixInitializeVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructMatrixSetValues thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructMatrixArray *matrix;
   int *grid_index;
   int num_stencil_indices;
   int *stencil_indices;
   double *values;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructMatrixSetValuesArgs;

void
HYPRE_StructMatrixSetValuesVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructMatrixSetValuesArgs *localargs =
      (HYPRE_StructMatrixSetValuesArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructMatrixSetValues(
         (*(localargs -> matrix))[threadid],
         localargs -> grid_index,
         localargs -> num_stencil_indices,
         localargs -> stencil_indices,
         localargs -> values );
}

int 
HYPRE_StructMatrixSetValuesPush(
   HYPRE_StructMatrixArray matrix,
   int *grid_index,
   int num_stencil_indices,
   int *stencil_indices,
   double *values )
{
   HYPRE_StructMatrixSetValuesArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.matrix = (HYPRE_StructMatrixArray *)matrix;
   pushargs.grid_index = grid_index;
   pushargs.num_stencil_indices = num_stencil_indices;
   pushargs.stencil_indices = stencil_indices;
   pushargs.values = values;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructMatrixSetValuesVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructMatrixSetBoxValues thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructMatrixArray *matrix;
   int *ilower;
   int *iupper;
   int num_stencil_indices;
   int *stencil_indices;
   double *values;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructMatrixSetBoxValuesArgs;

void
HYPRE_StructMatrixSetBoxValuesVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructMatrixSetBoxValuesArgs *localargs =
      (HYPRE_StructMatrixSetBoxValuesArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructMatrixSetBoxValues(
         (*(localargs -> matrix))[threadid],
         localargs -> ilower,
         localargs -> iupper,
         localargs -> num_stencil_indices,
         localargs -> stencil_indices,
         localargs -> values );
}

int 
HYPRE_StructMatrixSetBoxValuesPush(
   HYPRE_StructMatrixArray matrix,
   int *ilower,
   int *iupper,
   int num_stencil_indices,
   int *stencil_indices,
   double *values )
{
   HYPRE_StructMatrixSetBoxValuesArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.matrix = (HYPRE_StructMatrixArray *)matrix;
   pushargs.ilower = ilower;
   pushargs.iupper = iupper;
   pushargs.num_stencil_indices = num_stencil_indices;
   pushargs.stencil_indices = stencil_indices;
   pushargs.values = values;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructMatrixSetBoxValuesVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructMatrixAssemble thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructMatrixArray *matrix;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructMatrixAssembleArgs;

void
HYPRE_StructMatrixAssembleVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructMatrixAssembleArgs *localargs =
      (HYPRE_StructMatrixAssembleArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructMatrixAssemble(
         (*(localargs -> matrix))[threadid] );
}

int 
HYPRE_StructMatrixAssemblePush(
   HYPRE_StructMatrixArray matrix )
{
   HYPRE_StructMatrixAssembleArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.matrix = (HYPRE_StructMatrixArray *)matrix;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructMatrixAssembleVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructMatrixSetNumGhost thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructMatrixArray *matrix;
   int *num_ghost;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructMatrixSetNumGhostArgs;

void
HYPRE_StructMatrixSetNumGhostVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructMatrixSetNumGhostArgs *localargs =
      (HYPRE_StructMatrixSetNumGhostArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructMatrixSetNumGhost(
         (*(localargs -> matrix))[threadid],
         localargs -> num_ghost );
}

int 
HYPRE_StructMatrixSetNumGhostPush(
   HYPRE_StructMatrixArray matrix,
   int *num_ghost )
{
   HYPRE_StructMatrixSetNumGhostArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.matrix = (HYPRE_StructMatrixArray *)matrix;
   pushargs.num_ghost = num_ghost;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructMatrixSetNumGhostVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructMatrixGetGrid thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructMatrixArray *matrix;
   HYPRE_StructGridArray *grid;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructMatrixGetGridArgs;

void
HYPRE_StructMatrixGetGridVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructMatrixGetGridArgs *localargs =
      (HYPRE_StructMatrixGetGridArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructMatrixGetGrid(
         (*(localargs -> matrix))[threadid],
         &(*(localargs -> grid))[threadid] );
}

int 
HYPRE_StructMatrixGetGridPush(
   HYPRE_StructMatrixArray matrix,
   HYPRE_StructGridArray *grid )
{
   HYPRE_StructMatrixGetGridArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.matrix = (HYPRE_StructMatrixArray *)matrix;
   pushargs.grid = (HYPRE_StructGridArray *)grid;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructMatrixGetGridVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructMatrixSetSymmetric thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructMatrixArray *matrix;
   int symmetric;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructMatrixSetSymmetricArgs;

void
HYPRE_StructMatrixSetSymmetricVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructMatrixSetSymmetricArgs *localargs =
      (HYPRE_StructMatrixSetSymmetricArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructMatrixSetSymmetric(
         (*(localargs -> matrix))[threadid],
         localargs -> symmetric );
}

int 
HYPRE_StructMatrixSetSymmetricPush(
   HYPRE_StructMatrixArray matrix,
   int symmetric )
{
   HYPRE_StructMatrixSetSymmetricArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.matrix = (HYPRE_StructMatrixArray *)matrix;
   pushargs.symmetric = symmetric;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructMatrixSetSymmetricVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructMatrixPrint thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   char *filename;
   HYPRE_StructMatrixArray *matrix;
   int all;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructMatrixPrintArgs;

void
HYPRE_StructMatrixPrintVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructMatrixPrintArgs *localargs =
      (HYPRE_StructMatrixPrintArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructMatrixPrint(
         localargs -> filename,
         (*(localargs -> matrix))[threadid],
         localargs -> all );
}

int 
HYPRE_StructMatrixPrintPush(
   char *filename,
   HYPRE_StructMatrixArray matrix,
   int all )
{
   HYPRE_StructMatrixPrintArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.filename = filename;
   pushargs.matrix = (HYPRE_StructMatrixArray *)matrix;
   pushargs.all = all;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructMatrixPrintVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructStencilCreate thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   int dim;
   int size;
   HYPRE_StructStencilArray *stencil;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructStencilCreateArgs;

void
HYPRE_StructStencilCreateVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructStencilCreateArgs *localargs =
      (HYPRE_StructStencilCreateArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructStencilCreate(
         localargs -> dim,
         localargs -> size,
         &(*(localargs -> stencil))[threadid] );
}

int 
HYPRE_StructStencilCreatePush(
   int dim,
   int size,
   HYPRE_StructStencilArray *stencil )
{
   HYPRE_StructStencilCreateArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.dim = dim;
   pushargs.size = size;
   pushargs.stencil = (HYPRE_StructStencilArray *)stencil;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructStencilCreateVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructStencilSetElement thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructStencilArray *stencil;
   int element_index;
   int *offset;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructStencilSetElementArgs;

void
HYPRE_StructStencilSetElementVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructStencilSetElementArgs *localargs =
      (HYPRE_StructStencilSetElementArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructStencilSetElement(
         (*(localargs -> stencil))[threadid],
         localargs -> element_index,
         localargs -> offset );
}

int 
HYPRE_StructStencilSetElementPush(
   HYPRE_StructStencilArray stencil,
   int element_index,
   int *offset )
{
   HYPRE_StructStencilSetElementArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.stencil = (HYPRE_StructStencilArray *)stencil;
   pushargs.element_index = element_index;
   pushargs.offset = offset;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructStencilSetElementVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructStencilDestroy thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructStencilArray *stencil;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructStencilDestroyArgs;

void
HYPRE_StructStencilDestroyVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructStencilDestroyArgs *localargs =
      (HYPRE_StructStencilDestroyArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructStencilDestroy(
         (*(localargs -> stencil))[threadid] );
}

int 
HYPRE_StructStencilDestroyPush(
   HYPRE_StructStencilArray stencil )
{
   HYPRE_StructStencilDestroyArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.stencil = (HYPRE_StructStencilArray *)stencil;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructStencilDestroyVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructVectorCreate thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   MPI_Comm comm;
   HYPRE_StructGridArray *grid;
   HYPRE_StructStencilArray *stencil;
   HYPRE_StructVectorArray *vector;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructVectorCreateArgs;

void
HYPRE_StructVectorCreateVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructVectorCreateArgs *localargs =
      (HYPRE_StructVectorCreateArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructVectorCreate(
         localargs -> comm,
         (*(localargs -> grid))[threadid],
         (*(localargs -> stencil))[threadid],
         &(*(localargs -> vector))[threadid] );
}

int 
HYPRE_StructVectorCreatePush(
   MPI_Comm comm,
   HYPRE_StructGridArray grid,
   HYPRE_StructStencilArray stencil,
   HYPRE_StructVectorArray *vector )
{
   HYPRE_StructVectorCreateArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.comm = comm;
   pushargs.grid = (HYPRE_StructGridArray *)grid;
   pushargs.stencil = (HYPRE_StructStencilArray *)stencil;
   pushargs.vector = (HYPRE_StructVectorArray *)vector;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructVectorCreateVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructVectorDestroy thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructVectorArray *struct_vector;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructVectorDestroyArgs;

void
HYPRE_StructVectorDestroyVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructVectorDestroyArgs *localargs =
      (HYPRE_StructVectorDestroyArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructVectorDestroy(
         (*(localargs -> struct_vector))[threadid] );
}

int 
HYPRE_StructVectorDestroyPush(
   HYPRE_StructVectorArray struct_vector )
{
   HYPRE_StructVectorDestroyArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.struct_vector = (HYPRE_StructVectorArray *)struct_vector;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructVectorDestroyVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructVectorInitialize thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructVectorArray *vector;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructVectorInitializeArgs;

void
HYPRE_StructVectorInitializeVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructVectorInitializeArgs *localargs =
      (HYPRE_StructVectorInitializeArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructVectorInitialize(
         (*(localargs -> vector))[threadid] );
}

int 
HYPRE_StructVectorInitializePush(
   HYPRE_StructVectorArray vector )
{
   HYPRE_StructVectorInitializeArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.vector = (HYPRE_StructVectorArray *)vector;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructVectorInitializeVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructVectorSetValues thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructVectorArray *vector;
   int *grid_index;
   double values;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructVectorSetValuesArgs;

void
HYPRE_StructVectorSetValuesVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructVectorSetValuesArgs *localargs =
      (HYPRE_StructVectorSetValuesArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructVectorSetValues(
         (*(localargs -> vector))[threadid],
         localargs -> grid_index,
         localargs -> values );
}

int 
HYPRE_StructVectorSetValuesPush(
   HYPRE_StructVectorArray vector,
   int *grid_index,
   double values )
{
   HYPRE_StructVectorSetValuesArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.vector = (HYPRE_StructVectorArray *)vector;
   pushargs.grid_index = grid_index;
   pushargs.values = values;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructVectorSetValuesVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructVectorGetValues thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructVectorArray *vector;
   int *grid_index;
   double *values_ptr;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructVectorGetValuesArgs;

void
HYPRE_StructVectorGetValuesVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructVectorGetValuesArgs *localargs =
      (HYPRE_StructVectorGetValuesArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructVectorGetValues(
         (*(localargs -> vector))[threadid],
         localargs -> grid_index,
         localargs -> values_ptr );
}

int 
HYPRE_StructVectorGetValuesPush(
   HYPRE_StructVectorArray vector,
   int *grid_index,
   double *values_ptr )
{
   HYPRE_StructVectorGetValuesArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.vector = (HYPRE_StructVectorArray *)vector;
   pushargs.grid_index = grid_index;
   pushargs.values_ptr = values_ptr;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructVectorGetValuesVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructVectorSetBoxValues thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructVectorArray *vector;
   int *ilower;
   int *iupper;
   double *values;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructVectorSetBoxValuesArgs;

void
HYPRE_StructVectorSetBoxValuesVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructVectorSetBoxValuesArgs *localargs =
      (HYPRE_StructVectorSetBoxValuesArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructVectorSetBoxValues(
         (*(localargs -> vector))[threadid],
         localargs -> ilower,
         localargs -> iupper,
         localargs -> values );
}

int 
HYPRE_StructVectorSetBoxValuesPush(
   HYPRE_StructVectorArray vector,
   int *ilower,
   int *iupper,
   double *values )
{
   HYPRE_StructVectorSetBoxValuesArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.vector = (HYPRE_StructVectorArray *)vector;
   pushargs.ilower = ilower;
   pushargs.iupper = iupper;
   pushargs.values = values;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructVectorSetBoxValuesVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructVectorGetBoxValues thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructVectorArray *vector;
   int *ilower;
   int *iupper;
   double *values;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructVectorGetBoxValuesArgs;

void
HYPRE_StructVectorGetBoxValuesVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructVectorGetBoxValuesArgs *localargs =
      (HYPRE_StructVectorGetBoxValuesArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructVectorGetBoxValues(
         (*(localargs -> vector))[threadid],
         localargs -> ilower,
         localargs -> iupper,
         localargs -> values );
}

int 
HYPRE_StructVectorGetBoxValuesPush(
   HYPRE_StructVectorArray vector,
   int *ilower,
   int *iupper,
   double *values )
{
   HYPRE_StructVectorGetBoxValuesArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.vector = (HYPRE_StructVectorArray *)vector;
   pushargs.ilower = ilower;
   pushargs.iupper = iupper;
   pushargs.values = values;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructVectorGetBoxValuesVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructVectorAssemble thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructVectorArray *vector;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructVectorAssembleArgs;

void
HYPRE_StructVectorAssembleVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructVectorAssembleArgs *localargs =
      (HYPRE_StructVectorAssembleArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructVectorAssemble(
         (*(localargs -> vector))[threadid] );
}

int 
HYPRE_StructVectorAssemblePush(
   HYPRE_StructVectorArray vector )
{
   HYPRE_StructVectorAssembleArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.vector = (HYPRE_StructVectorArray *)vector;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructVectorAssembleVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructVectorPrint thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   char *filename;
   HYPRE_StructVectorArray *vector;
   int all;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructVectorPrintArgs;

void
HYPRE_StructVectorPrintVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructVectorPrintArgs *localargs =
      (HYPRE_StructVectorPrintArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructVectorPrint(
         localargs -> filename,
         (*(localargs -> vector))[threadid],
         localargs -> all );
}

int 
HYPRE_StructVectorPrintPush(
   char *filename,
   HYPRE_StructVectorArray vector,
   int all )
{
   HYPRE_StructVectorPrintArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.filename = filename;
   pushargs.vector = (HYPRE_StructVectorArray *)vector;
   pushargs.all = all;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructVectorPrintVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructVectorSetNumGhost thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructVectorArray *vector;
   int *num_ghost;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructVectorSetNumGhostArgs;

void
HYPRE_StructVectorSetNumGhostVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructVectorSetNumGhostArgs *localargs =
      (HYPRE_StructVectorSetNumGhostArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructVectorSetNumGhost(
         (*(localargs -> vector))[threadid],
         localargs -> num_ghost );
}

int 
HYPRE_StructVectorSetNumGhostPush(
   HYPRE_StructVectorArray vector,
   int *num_ghost )
{
   HYPRE_StructVectorSetNumGhostArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.vector = (HYPRE_StructVectorArray *)vector;
   pushargs.num_ghost = num_ghost;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructVectorSetNumGhostVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructVectorSetConstantValues thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructVectorArray *vector;
   double values;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructVectorSetConstantValuesArgs;

void
HYPRE_StructVectorSetConstantValuesVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructVectorSetConstantValuesArgs *localargs =
      (HYPRE_StructVectorSetConstantValuesArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructVectorSetConstantValues(
         (*(localargs -> vector))[threadid],
         localargs -> values );
}

int 
HYPRE_StructVectorSetConstantValuesPush(
   HYPRE_StructVectorArray vector,
   double values )
{
   HYPRE_StructVectorSetConstantValuesArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.vector = (HYPRE_StructVectorArray *)vector;
   pushargs.values = values;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructVectorSetConstantValuesVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructVectorGetMigrateCommPkg thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructVectorArray *from_vector;
   HYPRE_StructVectorArray *to_vector;
   HYPRE_CommPkgArray *comm_pkg;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructVectorGetMigrateCommPkgArgs;

void
HYPRE_StructVectorGetMigrateCommPkgVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructVectorGetMigrateCommPkgArgs *localargs =
      (HYPRE_StructVectorGetMigrateCommPkgArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructVectorGetMigrateCommPkg(
         (*(localargs -> from_vector))[threadid],
         (*(localargs -> to_vector))[threadid],
         &(*(localargs -> comm_pkg))[threadid] );
}

int 
HYPRE_StructVectorGetMigrateCommPkgPush(
   HYPRE_StructVectorArray from_vector,
   HYPRE_StructVectorArray to_vector,
   HYPRE_CommPkgArray *comm_pkg )
{
   HYPRE_StructVectorGetMigrateCommPkgArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.from_vector = (HYPRE_StructVectorArray *)from_vector;
   pushargs.to_vector = (HYPRE_StructVectorArray *)to_vector;
   pushargs.comm_pkg = (HYPRE_CommPkgArray *)comm_pkg;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructVectorGetMigrateCommPkgVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructVectorMigrate thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_CommPkgArray *comm_pkg;
   HYPRE_StructVectorArray *from_vector;
   HYPRE_StructVectorArray *to_vector;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructVectorMigrateArgs;

void
HYPRE_StructVectorMigrateVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructVectorMigrateArgs *localargs =
      (HYPRE_StructVectorMigrateArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructVectorMigrate(
         (*(localargs -> comm_pkg))[threadid],
         (*(localargs -> from_vector))[threadid],
         (*(localargs -> to_vector))[threadid] );
}

int 
HYPRE_StructVectorMigratePush(
   HYPRE_CommPkgArray comm_pkg,
   HYPRE_StructVectorArray from_vector,
   HYPRE_StructVectorArray to_vector )
{
   HYPRE_StructVectorMigrateArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.comm_pkg = (HYPRE_CommPkgArray *)comm_pkg;
   pushargs.from_vector = (HYPRE_StructVectorArray *)from_vector;
   pushargs.to_vector = (HYPRE_StructVectorArray *)to_vector;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructVectorMigrateVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_CommPkgDestroy thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_CommPkgArray *comm_pkg;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_CommPkgDestroyArgs;

void
HYPRE_CommPkgDestroyVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_CommPkgDestroyArgs *localargs =
      (HYPRE_CommPkgDestroyArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_CommPkgDestroy(
         (*(localargs -> comm_pkg))[threadid] );
}

int 
HYPRE_CommPkgDestroyPush(
   HYPRE_CommPkgArray comm_pkg )
{
   HYPRE_CommPkgDestroyArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.comm_pkg = (HYPRE_CommPkgArray *)comm_pkg;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_CommPkgDestroyVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

#else

/* this is used only to eliminate compiler warnings */
int hypre_empty;

#endif

