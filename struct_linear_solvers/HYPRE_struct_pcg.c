/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_StructPCG interface
 *
 *****************************************************************************/
#ifdef HYPRE_USE_PTHREADS
#define NO_PTHREAD_MANGLING
#endif

#include "headers.h"

#ifdef HYPRE_USE_PTHREADS
#include "box_pthreads.h"
#include "threading.h"
#endif
/*--------------------------------------------------------------------------
 * HYPRE_StructPCGInitialize
 *--------------------------------------------------------------------------*/

HYPRE_StructSolver
HYPRE_StructPCGInitialize( MPI_Comm comm )
{
   return ( (HYPRE_StructSolver) hypre_PCGInitialize( ) );
}

#ifdef HYPRE_USE_PTHREADS
typedef struct {
   MPI_Comm             comm;
   HYPRE_StructSolver  *returnvalue;
} HYPRE_StructPCGInitializeArgs;

void
HYPRE_StructPCGInitializeVoidPtr( void *argptr )
{
   HYPRE_StructPCGInitializeArgs *localargs =
                                (HYPRE_StructPCGInitializeArgs *) argptr;

   *(localargs->returnvalue) = HYPRE_StructPCGInitialize( localargs->comm );
}

HYPRE_StructSolver
HYPRE_StructPCGInitializePush( MPI_Comm comm )
{
   HYPRE_StructPCGInitializeArgs  pushargs;
   int                            i;
   HYPRE_StructSolver             returnvalue;

   pushargs.comm = comm;
   pushargs.returnvalue = 
                  (HYPRE_StructSolver*) malloc(sizeof(HYPRE_StructSolver));

   for (i=0; i<hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPCGInitializeVoidPtr, (void*)&pushargs );

   hypre_work_wait();

   returnvalue = *(pushargs.returnvalue);

   free( pushargs.returnvalue );

   return returnvalue;
}
#endif

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGFinalize
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructPCGFinalize( HYPRE_StructSolver solver )
{
   return( hypre_PCGFinalize( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructPCGSetup( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                      HYPRE_StructVector x      )
{
   return( hypre_PCGSetup( (void *) solver,
                           (void *) A,
                           (void *) b,
                           (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructPCGSolve( HYPRE_StructSolver solver,
                      HYPRE_StructMatrix A,
                      HYPRE_StructVector b,
                      HYPRE_StructVector x      )
{
   return( hypre_PCGSolve( (void *) solver,
                           (void *) A,
                           (void *) b,
                           (void *) x ) );
}

#ifdef HYPRE_USE_PTHREADS
typedef struct {
   HYPRE_StructSolver  solver;
   HYPRE_StructMatrix  A;
   HYPRE_StructVector  b;
   HYPRE_StructVector  x;
   int                *returnvalue;
} HYPRE_StructPCGSolveArgs;

void
HYPRE_StructPCGSolveVoidPtr( void *argptr )
{
   HYPRE_StructPCGSolveArgs *localargs = (HYPRE_StructPCGSolveArgs *) argptr;

   *(localargs->returnvalue) = HYPRE_StructPCGSolve( localargs->solver,
                                                     localargs->A,
                                                     localargs->b,
                                                     localargs->x );
}

int 
HYPRE_StructPCGSolvePush( HYPRE_StructSolver solver,
                          HYPRE_StructMatrix A,
                          HYPRE_StructVector b,
                          HYPRE_StructVector x      )
{
   HYPRE_StructPCGSolveArgs  pushargs;
   int                       i;
   int                       returnvalue;
   
   pushargs.solver = solver;
   pushargs.A      = A;
   pushargs.b      = b;
   pushargs.b      = b;
   pushargs.returnvalue = (int *) malloc(sizeof(int)); 
 
   for (i=0; i<hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPCGSolveVoidPtr, (void*)&pushargs );
   
   hypre_work_wait();
                      
   returnvalue = *(pushargs.returnvalue);  
                      
   free( pushargs.returnvalue );
   
   return returnvalue;
}
#endif

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPCGSetTol( HYPRE_StructSolver solver,
                       double             tol    )
{
   return( hypre_PCGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPCGSetMaxIter( HYPRE_StructSolver solver,
                           int                max_iter )
{
   return( hypre_PCGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetTwoNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPCGSetTwoNorm( HYPRE_StructSolver solver,
                           int                two_norm )
{
   return( hypre_PCGSetTwoNorm( (void *) solver, two_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetRelChange
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPCGSetRelChange( HYPRE_StructSolver solver,
                             int                rel_change )
{
   return( hypre_PCGSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPCGSetPrecond( HYPRE_StructSolver  solver,
                           int               (*precond)(),
                           int               (*precond_setup)(),
                           void               *precond_data )
{
   return( hypre_PCGSetPrecond( (void *) solver,
                                precond, precond_setup, precond_data ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPCGSetLogging( HYPRE_StructSolver solver,
                           int                logging )
{
   return( hypre_PCGSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPCGGetNumIterations( HYPRE_StructSolver  solver,
                                 int                *num_iterations )
{
   return( hypre_PCGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_StructPCGGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                             double             *norm   )
{
   return( hypre_PCGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructDiagScaleSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructDiagScaleSetup( HYPRE_StructSolver solver,
                            HYPRE_StructMatrix A,
                            HYPRE_StructVector y,
                            HYPRE_StructVector x      )
{
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructDiagScale
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructDiagScale( HYPRE_StructSolver solver,
                       HYPRE_StructMatrix HA,
                       HYPRE_StructVector Hy,
                       HYPRE_StructVector Hx      )
{
   hypre_StructMatrix   *A = (hypre_StructMatrix *) HA;
   hypre_StructVector   *y = (hypre_StructVector *) Hy;
   hypre_StructVector   *x = (hypre_StructVector *) Hx;

   hypre_BoxArray       *boxes;
   hypre_Box            *box;

   hypre_Box            *A_data_box;
   hypre_Box            *y_data_box;
   hypre_Box            *x_data_box;
                     
   double               *Ap;
   double               *yp;
   double               *xp;
                       
   int                   Ai;
   int                   yi;
   int                   xi;
                     
   hypre_Index           index;
   hypre_IndexRef        start;
   hypre_Index           stride;
   hypre_Index           loop_size;
                     
   int                   i;
   int                   loopi, loopj, loopk;

   int                   ierr = 0;
  
   /* x = D^{-1} y */
   hypre_SetIndex(stride, 1, 1, 1);
   boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
   hypre_ForBoxI(i, boxes)
      {
         box = hypre_BoxArrayBox(boxes, i);

         A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
         x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

         hypre_SetIndex(index, 0, 0, 0);
         Ap = hypre_StructMatrixExtractPointerByIndex(A, i, index);
         xp = hypre_StructVectorBoxData(x, i);
         yp = hypre_StructVectorBoxData(y, i);

         start  = hypre_BoxIMin(box);

         hypre_GetBoxSize(box, loop_size);
         hypre_BoxLoop3(loopi, loopj, loopk, loop_size,
                        A_data_box,  start,  stride,  Ai,
                        x_data_box,  start,  stride,  xi,
                        y_data_box,  start,  stride,  yi,
                        {
                           xp[xi] = yp[yi] / Ap[Ai];
                        });
      }

   return ierr;
}

#ifdef HYPRE_USE_PTHREADS
typedef struct {
   HYPRE_StructSolver  solver;
   HYPRE_StructMatrix  HA;
   HYPRE_StructVector  Hy;
   HYPRE_StructVector  Hx;
   int                *returnvalue;
} HYPRE_StructDiagScaleArgs;

void
HYPRE_StructDiagScaleVoidPtr ( void *argptr )
{

   HYPRE_StructDiagScaleArgs *localargs = (HYPRE_StructDiagScaleArgs *) argptr;

   *(localargs->returnvalue) = HYPRE_StructDiagScale( localargs->solver,
                                                       localargs->HA,
                                                       localargs->Hy,
                                                       localargs->Hx );
}

int 
HYPRE_StructDiagScalePush( HYPRE_StructSolver solver,
                           HYPRE_StructMatrix HA,
                           HYPRE_StructVector Hy,
                           HYPRE_StructVector Hx      )
{
   HYPRE_StructDiagScaleArgs  pushargs;
   int                        i;
   int                        returnvalue;

   pushargs.solver = solver;
   pushargs.HA     = HA;
   pushargs.Hy     = Hy;
   pushargs.Hx     = Hx;
   pushargs.returnvalue = (int *) malloc(sizeof(int));

   for (i=0; i<hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructDiagScaleVoidPtr, (void *)&pushargs);

   hypre_work_wait();

   returnvalue = *(pushargs.returnvalue);

   free( pushargs.returnvalue );

   return returnvalue;
}
#endif
