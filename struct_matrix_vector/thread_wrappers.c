
#include "HYPRE_mv.h"
#include "utilities.h"
#ifdef HYPRE_USE_PTHREADS


/*----------------------------------------------------------------
 * HYPRE_CreateStructGrid thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   MPI_Comm comm;
   int dim;
   HYPRE_StructGridArray *grid;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_CreateStructGridArgs;

void
HYPRE_CreateStructGridVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_CreateStructGridArgs *localargs =
      (HYPRE_CreateStructGridArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_CreateStructGrid(
         localargs -> comm,
         localargs -> dim,
         &(*(localargs -> grid))[threadid] );
}

int 
HYPRE_CreateStructGridPush(
   MPI_Comm comm,
   int dim,
   HYPRE_StructGridArray *grid )
{
   HYPRE_CreateStructGridArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.comm = comm;
   pushargs.dim = dim;
   pushargs.grid = (HYPRE_StructGridArray *)grid;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_CreateStructGridVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_DestroyStructGrid thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructGridArray *grid;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_DestroyStructGridArgs;

void
HYPRE_DestroyStructGridVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_DestroyStructGridArgs *localargs =
      (HYPRE_DestroyStructGridArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_DestroyStructGrid(
         (*(localargs -> grid))[threadid] );
}

int 
HYPRE_DestroyStructGridPush(
   HYPRE_StructGridArray grid )
{
   HYPRE_DestroyStructGridArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.grid = (HYPRE_StructGridArray *)grid;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_DestroyStructGridVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_SetStructGridExtents thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructGridArray *grid;
   int *ilower;
   int *iupper;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_SetStructGridExtentsArgs;

void
HYPRE_SetStructGridExtentsVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_SetStructGridExtentsArgs *localargs =
      (HYPRE_SetStructGridExtentsArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_SetStructGridExtents(
         (*(localargs -> grid))[threadid],
         localargs -> ilower,
         localargs -> iupper );
}

int 
HYPRE_SetStructGridExtentsPush(
   HYPRE_StructGridArray grid,
   int *ilower,
   int *iupper )
{
   HYPRE_SetStructGridExtentsArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.grid = (HYPRE_StructGridArray *)grid;
   pushargs.ilower = ilower;
   pushargs.iupper = iupper;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_SetStructGridExtentsVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_SetStructGridPeriodic thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructGridArray *grid;
   int *periodic;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_SetStructGridPeriodicArgs;

void
HYPRE_SetStructGridPeriodicVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_SetStructGridPeriodicArgs *localargs =
      (HYPRE_SetStructGridPeriodicArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_SetStructGridPeriodic(
         (*(localargs -> grid))[threadid],
         localargs -> periodic );
}

int 
HYPRE_SetStructGridPeriodicPush(
   HYPRE_StructGridArray grid,
   int *periodic )
{
   HYPRE_SetStructGridPeriodicArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.grid = (HYPRE_StructGridArray *)grid;
   pushargs.periodic = periodic;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_SetStructGridPeriodicVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_AssembleStructGrid thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructGridArray *grid;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_AssembleStructGridArgs;

void
HYPRE_AssembleStructGridVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_AssembleStructGridArgs *localargs =
      (HYPRE_AssembleStructGridArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_AssembleStructGrid(
         (*(localargs -> grid))[threadid] );
}

int 
HYPRE_AssembleStructGridPush(
   HYPRE_StructGridArray grid )
{
   HYPRE_AssembleStructGridArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.grid = (HYPRE_StructGridArray *)grid;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_AssembleStructGridVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_CreateStructMatrix thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   MPI_Comm comm;
   HYPRE_StructGridArray *grid;
   HYPRE_StructStencilArray *stencil;
   HYPRE_StructMatrixArray *matrix;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_CreateStructMatrixArgs;

void
HYPRE_CreateStructMatrixVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_CreateStructMatrixArgs *localargs =
      (HYPRE_CreateStructMatrixArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_CreateStructMatrix(
         localargs -> comm,
         (*(localargs -> grid))[threadid],
         (*(localargs -> stencil))[threadid],
         &(*(localargs -> matrix))[threadid] );
}

int 
HYPRE_CreateStructMatrixPush(
   MPI_Comm comm,
   HYPRE_StructGridArray grid,
   HYPRE_StructStencilArray stencil,
   HYPRE_StructMatrixArray *matrix )
{
   HYPRE_CreateStructMatrixArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.comm = comm;
   pushargs.grid = (HYPRE_StructGridArray *)grid;
   pushargs.stencil = (HYPRE_StructStencilArray *)stencil;
   pushargs.matrix = (HYPRE_StructMatrixArray *)matrix;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_CreateStructMatrixVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_DestroyStructMatrix thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructMatrixArray *matrix;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_DestroyStructMatrixArgs;

void
HYPRE_DestroyStructMatrixVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_DestroyStructMatrixArgs *localargs =
      (HYPRE_DestroyStructMatrixArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_DestroyStructMatrix(
         (*(localargs -> matrix))[threadid] );
}

int 
HYPRE_DestroyStructMatrixPush(
   HYPRE_StructMatrixArray matrix )
{
   HYPRE_DestroyStructMatrixArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.matrix = (HYPRE_StructMatrixArray *)matrix;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_DestroyStructMatrixVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_InitializeStructMatrix thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructMatrixArray *matrix;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_InitializeStructMatrixArgs;

void
HYPRE_InitializeStructMatrixVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_InitializeStructMatrixArgs *localargs =
      (HYPRE_InitializeStructMatrixArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_InitializeStructMatrix(
         (*(localargs -> matrix))[threadid] );
}

int 
HYPRE_InitializeStructMatrixPush(
   HYPRE_StructMatrixArray matrix )
{
   HYPRE_InitializeStructMatrixArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.matrix = (HYPRE_StructMatrixArray *)matrix;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_InitializeStructMatrixVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_SetStructMatrixValues thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructMatrixArray *matrix;
   int *grid_index;
   int num_stencil_indices;
   int *stencil_indices;
   double *values;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_SetStructMatrixValuesArgs;

void
HYPRE_SetStructMatrixValuesVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_SetStructMatrixValuesArgs *localargs =
      (HYPRE_SetStructMatrixValuesArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_SetStructMatrixValues(
         (*(localargs -> matrix))[threadid],
         localargs -> grid_index,
         localargs -> num_stencil_indices,
         localargs -> stencil_indices,
         localargs -> values );
}

int 
HYPRE_SetStructMatrixValuesPush(
   HYPRE_StructMatrixArray matrix,
   int *grid_index,
   int num_stencil_indices,
   int *stencil_indices,
   double *values )
{
   HYPRE_SetStructMatrixValuesArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.matrix = (HYPRE_StructMatrixArray *)matrix;
   pushargs.grid_index = grid_index;
   pushargs.num_stencil_indices = num_stencil_indices;
   pushargs.stencil_indices = stencil_indices;
   pushargs.values = values;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_SetStructMatrixValuesVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_SetStructMatrixBoxValues thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructMatrixArray *matrix;
   int *ilower;
   int *iupper;
   int num_stencil_indices;
   int *stencil_indices;
   double *values;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_SetStructMatrixBoxValuesArgs;

void
HYPRE_SetStructMatrixBoxValuesVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_SetStructMatrixBoxValuesArgs *localargs =
      (HYPRE_SetStructMatrixBoxValuesArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_SetStructMatrixBoxValues(
         (*(localargs -> matrix))[threadid],
         localargs -> ilower,
         localargs -> iupper,
         localargs -> num_stencil_indices,
         localargs -> stencil_indices,
         localargs -> values );
}

int 
HYPRE_SetStructMatrixBoxValuesPush(
   HYPRE_StructMatrixArray matrix,
   int *ilower,
   int *iupper,
   int num_stencil_indices,
   int *stencil_indices,
   double *values )
{
   HYPRE_SetStructMatrixBoxValuesArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.matrix = (HYPRE_StructMatrixArray *)matrix;
   pushargs.ilower = ilower;
   pushargs.iupper = iupper;
   pushargs.num_stencil_indices = num_stencil_indices;
   pushargs.stencil_indices = stencil_indices;
   pushargs.values = values;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_SetStructMatrixBoxValuesVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_AssembleStructMatrix thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructMatrixArray *matrix;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_AssembleStructMatrixArgs;

void
HYPRE_AssembleStructMatrixVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_AssembleStructMatrixArgs *localargs =
      (HYPRE_AssembleStructMatrixArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_AssembleStructMatrix(
         (*(localargs -> matrix))[threadid] );
}

int 
HYPRE_AssembleStructMatrixPush(
   HYPRE_StructMatrixArray matrix )
{
   HYPRE_AssembleStructMatrixArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.matrix = (HYPRE_StructMatrixArray *)matrix;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_AssembleStructMatrixVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_SetStructMatrixNumGhost thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructMatrixArray *matrix;
   int *num_ghost;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_SetStructMatrixNumGhostArgs;

void
HYPRE_SetStructMatrixNumGhostVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_SetStructMatrixNumGhostArgs *localargs =
      (HYPRE_SetStructMatrixNumGhostArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_SetStructMatrixNumGhost(
         (*(localargs -> matrix))[threadid],
         localargs -> num_ghost );
}

int 
HYPRE_SetStructMatrixNumGhostPush(
   HYPRE_StructMatrixArray matrix,
   int *num_ghost )
{
   HYPRE_SetStructMatrixNumGhostArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.matrix = (HYPRE_StructMatrixArray *)matrix;
   pushargs.num_ghost = num_ghost;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_SetStructMatrixNumGhostVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructMatrixGrid thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructMatrixArray *matrix;
   HYPRE_StructGridArray *grid;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructMatrixGridArgs;

void
HYPRE_StructMatrixGridVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructMatrixGridArgs *localargs =
      (HYPRE_StructMatrixGridArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructMatrixGrid(
         (*(localargs -> matrix))[threadid],
         &(*(localargs -> grid))[threadid] );
}

int 
HYPRE_StructMatrixGridPush(
   HYPRE_StructMatrixArray matrix,
   HYPRE_StructGridArray *grid )
{
   HYPRE_StructMatrixGridArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.matrix = (HYPRE_StructMatrixArray *)matrix;
   pushargs.grid = (HYPRE_StructGridArray *)grid;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructMatrixGridVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_SetStructMatrixSymmetric thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructMatrixArray *matrix;
   int symmetric;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_SetStructMatrixSymmetricArgs;

void
HYPRE_SetStructMatrixSymmetricVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_SetStructMatrixSymmetricArgs *localargs =
      (HYPRE_SetStructMatrixSymmetricArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_SetStructMatrixSymmetric(
         (*(localargs -> matrix))[threadid],
         localargs -> symmetric );
}

int 
HYPRE_SetStructMatrixSymmetricPush(
   HYPRE_StructMatrixArray matrix,
   int symmetric )
{
   HYPRE_SetStructMatrixSymmetricArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.matrix = (HYPRE_StructMatrixArray *)matrix;
   pushargs.symmetric = symmetric;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_SetStructMatrixSymmetricVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_PrintStructMatrix thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   char *filename;
   HYPRE_StructMatrixArray *matrix;
   int all;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_PrintStructMatrixArgs;

void
HYPRE_PrintStructMatrixVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_PrintStructMatrixArgs *localargs =
      (HYPRE_PrintStructMatrixArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_PrintStructMatrix(
         localargs -> filename,
         (*(localargs -> matrix))[threadid],
         localargs -> all );
}

int 
HYPRE_PrintStructMatrixPush(
   char *filename,
   HYPRE_StructMatrixArray matrix,
   int all )
{
   HYPRE_PrintStructMatrixArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.filename = filename;
   pushargs.matrix = (HYPRE_StructMatrixArray *)matrix;
   pushargs.all = all;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_PrintStructMatrixVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_CreateStructStencil thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   int dim;
   int size;
   HYPRE_StructStencilArray *stencil;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_CreateStructStencilArgs;

void
HYPRE_CreateStructStencilVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_CreateStructStencilArgs *localargs =
      (HYPRE_CreateStructStencilArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_CreateStructStencil(
         localargs -> dim,
         localargs -> size,
         &(*(localargs -> stencil))[threadid] );
}

int 
HYPRE_CreateStructStencilPush(
   int dim,
   int size,
   HYPRE_StructStencilArray *stencil )
{
   HYPRE_CreateStructStencilArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.dim = dim;
   pushargs.size = size;
   pushargs.stencil = (HYPRE_StructStencilArray *)stencil;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_CreateStructStencilVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_SetStructStencilElement thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructStencilArray *stencil;
   int element_index;
   int *offset;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_SetStructStencilElementArgs;

void
HYPRE_SetStructStencilElementVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_SetStructStencilElementArgs *localargs =
      (HYPRE_SetStructStencilElementArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_SetStructStencilElement(
         (*(localargs -> stencil))[threadid],
         localargs -> element_index,
         localargs -> offset );
}

int 
HYPRE_SetStructStencilElementPush(
   HYPRE_StructStencilArray stencil,
   int element_index,
   int *offset )
{
   HYPRE_SetStructStencilElementArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.stencil = (HYPRE_StructStencilArray *)stencil;
   pushargs.element_index = element_index;
   pushargs.offset = offset;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_SetStructStencilElementVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_DestroyStructStencil thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructStencilArray *stencil;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_DestroyStructStencilArgs;

void
HYPRE_DestroyStructStencilVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_DestroyStructStencilArgs *localargs =
      (HYPRE_DestroyStructStencilArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_DestroyStructStencil(
         (*(localargs -> stencil))[threadid] );
}

int 
HYPRE_DestroyStructStencilPush(
   HYPRE_StructStencilArray stencil )
{
   HYPRE_DestroyStructStencilArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.stencil = (HYPRE_StructStencilArray *)stencil;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_DestroyStructStencilVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_CreateStructVector thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   MPI_Comm comm;
   HYPRE_StructGridArray *grid;
   HYPRE_StructStencilArray *stencil;
   HYPRE_StructVectorArray *vector;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_CreateStructVectorArgs;

void
HYPRE_CreateStructVectorVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_CreateStructVectorArgs *localargs =
      (HYPRE_CreateStructVectorArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_CreateStructVector(
         localargs -> comm,
         (*(localargs -> grid))[threadid],
         (*(localargs -> stencil))[threadid],
         &(*(localargs -> vector))[threadid] );
}

int 
HYPRE_CreateStructVectorPush(
   MPI_Comm comm,
   HYPRE_StructGridArray grid,
   HYPRE_StructStencilArray stencil,
   HYPRE_StructVectorArray *vector )
{
   HYPRE_CreateStructVectorArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.comm = comm;
   pushargs.grid = (HYPRE_StructGridArray *)grid;
   pushargs.stencil = (HYPRE_StructStencilArray *)stencil;
   pushargs.vector = (HYPRE_StructVectorArray *)vector;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_CreateStructVectorVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_DestroyStructVector thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructVectorArray *struct_vector;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_DestroyStructVectorArgs;

void
HYPRE_DestroyStructVectorVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_DestroyStructVectorArgs *localargs =
      (HYPRE_DestroyStructVectorArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_DestroyStructVector(
         (*(localargs -> struct_vector))[threadid] );
}

int 
HYPRE_DestroyStructVectorPush(
   HYPRE_StructVectorArray struct_vector )
{
   HYPRE_DestroyStructVectorArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.struct_vector = (HYPRE_StructVectorArray *)struct_vector;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_DestroyStructVectorVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_InitializeStructVector thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructVectorArray *vector;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_InitializeStructVectorArgs;

void
HYPRE_InitializeStructVectorVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_InitializeStructVectorArgs *localargs =
      (HYPRE_InitializeStructVectorArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_InitializeStructVector(
         (*(localargs -> vector))[threadid] );
}

int 
HYPRE_InitializeStructVectorPush(
   HYPRE_StructVectorArray vector )
{
   HYPRE_InitializeStructVectorArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.vector = (HYPRE_StructVectorArray *)vector;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_InitializeStructVectorVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_SetStructVectorValues thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructVectorArray *vector;
   int *grid_index;
   double values;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_SetStructVectorValuesArgs;

void
HYPRE_SetStructVectorValuesVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_SetStructVectorValuesArgs *localargs =
      (HYPRE_SetStructVectorValuesArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_SetStructVectorValues(
         (*(localargs -> vector))[threadid],
         localargs -> grid_index,
         localargs -> values );
}

int 
HYPRE_SetStructVectorValuesPush(
   HYPRE_StructVectorArray vector,
   int *grid_index,
   double values )
{
   HYPRE_SetStructVectorValuesArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.vector = (HYPRE_StructVectorArray *)vector;
   pushargs.grid_index = grid_index;
   pushargs.values = values;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_SetStructVectorValuesVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_GetStructVectorValues thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructVectorArray *vector;
   int *grid_index;
   double *values_ptr;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_GetStructVectorValuesArgs;

void
HYPRE_GetStructVectorValuesVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_GetStructVectorValuesArgs *localargs =
      (HYPRE_GetStructVectorValuesArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_GetStructVectorValues(
         (*(localargs -> vector))[threadid],
         localargs -> grid_index,
         localargs -> values_ptr );
}

int 
HYPRE_GetStructVectorValuesPush(
   HYPRE_StructVectorArray vector,
   int *grid_index,
   double *values_ptr )
{
   HYPRE_GetStructVectorValuesArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.vector = (HYPRE_StructVectorArray *)vector;
   pushargs.grid_index = grid_index;
   pushargs.values_ptr = values_ptr;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_GetStructVectorValuesVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_SetStructVectorBoxValues thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructVectorArray *vector;
   int *ilower;
   int *iupper;
   double *values;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_SetStructVectorBoxValuesArgs;

void
HYPRE_SetStructVectorBoxValuesVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_SetStructVectorBoxValuesArgs *localargs =
      (HYPRE_SetStructVectorBoxValuesArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_SetStructVectorBoxValues(
         (*(localargs -> vector))[threadid],
         localargs -> ilower,
         localargs -> iupper,
         localargs -> values );
}

int 
HYPRE_SetStructVectorBoxValuesPush(
   HYPRE_StructVectorArray vector,
   int *ilower,
   int *iupper,
   double *values )
{
   HYPRE_SetStructVectorBoxValuesArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.vector = (HYPRE_StructVectorArray *)vector;
   pushargs.ilower = ilower;
   pushargs.iupper = iupper;
   pushargs.values = values;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_SetStructVectorBoxValuesVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_GetStructVectorBoxValues thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructVectorArray *vector;
   int *ilower;
   int *iupper;
   double *values;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_GetStructVectorBoxValuesArgs;

void
HYPRE_GetStructVectorBoxValuesVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_GetStructVectorBoxValuesArgs *localargs =
      (HYPRE_GetStructVectorBoxValuesArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_GetStructVectorBoxValues(
         (*(localargs -> vector))[threadid],
         localargs -> ilower,
         localargs -> iupper,
         localargs -> values );
}

int 
HYPRE_GetStructVectorBoxValuesPush(
   HYPRE_StructVectorArray vector,
   int *ilower,
   int *iupper,
   double *values )
{
   HYPRE_GetStructVectorBoxValuesArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.vector = (HYPRE_StructVectorArray *)vector;
   pushargs.ilower = ilower;
   pushargs.iupper = iupper;
   pushargs.values = values;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_GetStructVectorBoxValuesVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_AssembleStructVector thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructVectorArray *vector;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_AssembleStructVectorArgs;

void
HYPRE_AssembleStructVectorVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_AssembleStructVectorArgs *localargs =
      (HYPRE_AssembleStructVectorArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_AssembleStructVector(
         (*(localargs -> vector))[threadid] );
}

int 
HYPRE_AssembleStructVectorPush(
   HYPRE_StructVectorArray vector )
{
   HYPRE_AssembleStructVectorArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.vector = (HYPRE_StructVectorArray *)vector;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_AssembleStructVectorVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_PrintStructVector thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   char *filename;
   HYPRE_StructVectorArray *vector;
   int all;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_PrintStructVectorArgs;

void
HYPRE_PrintStructVectorVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_PrintStructVectorArgs *localargs =
      (HYPRE_PrintStructVectorArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_PrintStructVector(
         localargs -> filename,
         (*(localargs -> vector))[threadid],
         localargs -> all );
}

int 
HYPRE_PrintStructVectorPush(
   char *filename,
   HYPRE_StructVectorArray vector,
   int all )
{
   HYPRE_PrintStructVectorArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.filename = filename;
   pushargs.vector = (HYPRE_StructVectorArray *)vector;
   pushargs.all = all;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_PrintStructVectorVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_SetStructVectorNumGhost thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructVectorArray *vector;
   int *num_ghost;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_SetStructVectorNumGhostArgs;

void
HYPRE_SetStructVectorNumGhostVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_SetStructVectorNumGhostArgs *localargs =
      (HYPRE_SetStructVectorNumGhostArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_SetStructVectorNumGhost(
         (*(localargs -> vector))[threadid],
         localargs -> num_ghost );
}

int 
HYPRE_SetStructVectorNumGhostPush(
   HYPRE_StructVectorArray vector,
   int *num_ghost )
{
   HYPRE_SetStructVectorNumGhostArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.vector = (HYPRE_StructVectorArray *)vector;
   pushargs.num_ghost = num_ghost;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_SetStructVectorNumGhostVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_SetStructVectorConstantValues thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructVectorArray *vector;
   double values;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_SetStructVectorConstantValuesArgs;

void
HYPRE_SetStructVectorConstantValuesVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_SetStructVectorConstantValuesArgs *localargs =
      (HYPRE_SetStructVectorConstantValuesArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_SetStructVectorConstantValues(
         (*(localargs -> vector))[threadid],
         localargs -> values );
}

int 
HYPRE_SetStructVectorConstantValuesPush(
   HYPRE_StructVectorArray vector,
   double values )
{
   HYPRE_SetStructVectorConstantValuesArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.vector = (HYPRE_StructVectorArray *)vector;
   pushargs.values = values;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_SetStructVectorConstantValuesVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_GetMigrateStructVectorCommPkg thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructVectorArray *from_vector;
   HYPRE_StructVectorArray *to_vector;
   HYPRE_CommPkgArray *comm_pkg;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_GetMigrateStructVectorCommPkgArgs;

void
HYPRE_GetMigrateStructVectorCommPkgVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_GetMigrateStructVectorCommPkgArgs *localargs =
      (HYPRE_GetMigrateStructVectorCommPkgArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_GetMigrateStructVectorCommPkg(
         (*(localargs -> from_vector))[threadid],
         (*(localargs -> to_vector))[threadid],
         &(*(localargs -> comm_pkg))[threadid] );
}

int 
HYPRE_GetMigrateStructVectorCommPkgPush(
   HYPRE_StructVectorArray from_vector,
   HYPRE_StructVectorArray to_vector,
   HYPRE_CommPkgArray *comm_pkg )
{
   HYPRE_GetMigrateStructVectorCommPkgArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.from_vector = (HYPRE_StructVectorArray *)from_vector;
   pushargs.to_vector = (HYPRE_StructVectorArray *)to_vector;
   pushargs.comm_pkg = (HYPRE_CommPkgArray *)comm_pkg;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_GetMigrateStructVectorCommPkgVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_MigrateStructVector thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_CommPkgArray *comm_pkg;
   HYPRE_StructVectorArray *from_vector;
   HYPRE_StructVectorArray *to_vector;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_MigrateStructVectorArgs;

void
HYPRE_MigrateStructVectorVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_MigrateStructVectorArgs *localargs =
      (HYPRE_MigrateStructVectorArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_MigrateStructVector(
         (*(localargs -> comm_pkg))[threadid],
         (*(localargs -> from_vector))[threadid],
         (*(localargs -> to_vector))[threadid] );
}

int 
HYPRE_MigrateStructVectorPush(
   HYPRE_CommPkgArray comm_pkg,
   HYPRE_StructVectorArray from_vector,
   HYPRE_StructVectorArray to_vector )
{
   HYPRE_MigrateStructVectorArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.comm_pkg = (HYPRE_CommPkgArray *)comm_pkg;
   pushargs.from_vector = (HYPRE_StructVectorArray *)from_vector;
   pushargs.to_vector = (HYPRE_StructVectorArray *)to_vector;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_MigrateStructVectorVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_DestroyCommPkg thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_CommPkgArray *comm_pkg;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_DestroyCommPkgArgs;

void
HYPRE_DestroyCommPkgVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_DestroyCommPkgArgs *localargs =
      (HYPRE_DestroyCommPkgArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_DestroyCommPkg(
         (*(localargs -> comm_pkg))[threadid] );
}

int 
HYPRE_DestroyCommPkgPush(
   HYPRE_CommPkgArray comm_pkg )
{
   HYPRE_DestroyCommPkgArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.comm_pkg = (HYPRE_CommPkgArray *)comm_pkg;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_DestroyCommPkgVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

#else

/* this is used only to eliminate compiler warnings */
int hypre_empty;

#endif

