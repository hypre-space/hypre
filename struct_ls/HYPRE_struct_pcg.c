/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "headers.h"

#ifdef HYPRE_USE_PTHREADS
#include "box_pthreads.h"
#endif

/*==========================================================================*/
/** Returns a default PCG solver structure.

{\bf Input files:}
headers.h

@return integer 0

@param comm []
  MPI communicator
@param *solver []
  solver structure

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup */
/*--------------------------------------------------------------------------*/

int
HYPRE_StructPCGInitialize( MPI_Comm comm, HYPRE_StructSolver *solver )
{
   *solver = ( (HYPRE_StructSolver) hypre_PCGInitialize( ) );

   return 0;
}

/*==========================================================================*/
/*==========================================================================*/
/** Free a PCG solver structure.

{\bf Input files:}
headers.h

@return integer error code

@param solver []
  solver structure

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup */
/*--------------------------------------------------------------------------*/

int 
HYPRE_StructPCGFinalize( HYPRE_StructSolver solver )
{
   return( hypre_PCGFinalize( (void *) solver ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** Gets the matrix data. Will comlpete any remianing tasks needed to do the
    solve such as set up the NewVector arguments, initialize the Matvec,
    set up the precondioner, and allocate space for log info.

{\bf Input files:}
headers.h

@return integer error code

@param solver []
  solver structure
@param A []
  structured matrix
@param b []
  structured vector
@param x []
  structured vector

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup */
/*--------------------------------------------------------------------------*/

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

/*==========================================================================*/
/*==========================================================================*/
/** Performs the PCG linear solve.

{\bf Input files:}
headers.h

@return integer error code

@param solver []
  solver structure
@param A []
  structured matrix
@param b []
  structured vector
@param x []
  structured vector

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup */
/*--------------------------------------------------------------------------*/

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

/*==========================================================================*/
/*==========================================================================*/
/** An optional call to set the PCG solver tolerance.

{\bf Input files:}
headers.h

@return integer error code

@param solver []
  solver structure
@param tol []
  PCG solver tolerance

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup   */
/*--------------------------------------------------------------------------*/

int
HYPRE_StructPCGSetTol( HYPRE_StructSolver solver,
                       double             tol    )
{
   return( hypre_PCGSetTol( (void *) solver, tol ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** An optional call to set the PCG solver maximum number of iterations.

{\bf Input files:}
headers.h

@return integer error code

@param solver []
  solver structure
@param max_iter
  PCG solver maximum number of iterations

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup */
/*--------------------------------------------------------------------------*/

int
HYPRE_StructPCGSetMaxIter( HYPRE_StructSolver solver,
                           int                max_iter )
{
   return( hypre_PCGSetMaxIter( (void *) solver, max_iter ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** Optional call to set the PCG solver ???? .

{\bf Input files:}
headers.h

@return integer error code

@param solver []
  solver structure
@param two_norm []
  ????

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup */
/*--------------------------------------------------------------------------*/

int
HYPRE_StructPCGSetTwoNorm( HYPRE_StructSolver solver,
                           int                two_norm )
{
   return( hypre_PCGSetTwoNorm( (void *) solver, two_norm ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** Optional call to set the PCG solver relative change parameter.

{\bf Input files:}
headers.h

@return integer error code

@param solver []
  solver structure
@param rel_change
  PCG solver relative change

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup */
/*--------------------------------------------------------------------------*/

int
HYPRE_StructPCGSetRelChange( HYPRE_StructSolver solver,
                             int                rel_change )
{
   return( hypre_PCGSetRelChange( (void *) solver, rel_change ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** Optional call to apply a precondioner prior to the solve.

{\bf Input files:}
headers.h

@return integer error code

@param solver []
  solver structure
@param *precond() []
  the preconditioner to be applied
@param *precond_setup() []
  routine to setup the preconditioner
@param *precond_data []
  data needed by the preconditioner

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup*/
/*--------------------------------------------------------------------------*/

int
HYPRE_StructPCGSetPrecond( HYPRE_StructSolver  solver,
                           int               (*precond)(),
                           int               (*precond_setup)(),
                           HYPRE_StructSolver  precond_solver   )
{
   return( hypre_PCGSetPrecond( (void *) solver,
                                precond, precond_setup,
                                (void *) precond_solver ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** Optional call to ???? .

{\bf Input files:}
headers.h

@return integer error code

@param solver []
  solver structure
@param logging
  ????

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup */
/*--------------------------------------------------------------------------*/

int
HYPRE_StructPCGSetLogging( HYPRE_StructSolver solver,
                           int                logging )
{
   return( hypre_PCGSetLogging( (void *) solver, logging ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** Optional call to get the number of ierations of the PCG solve.

{\bf Input files:}
headers.h

@return integer error code

@param solver []
  solver structure
@param *num_iterations
  number of iterations of the PCG solve

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup */
/*--------------------------------------------------------------------------*/

int
HYPRE_StructPCGGetNumIterations( HYPRE_StructSolver  solver,
                                 int                *num_iterations )
{
   return( hypre_PCGGetNumIterations( (void *) solver, num_iterations ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** Optional call to retrieve the final relative residual norms of the
    PCG solve.

{\bf Input files:}
headers.h

@return integer error code

@param solver []
  solver structure
@param *norm []
  ????

@see HYPRE_StructPCGSolve, HYPRE_StructPCGSetup */
/*--------------------------------------------------------------------------*/

int
HYPRE_StructPCGGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                             double             *norm   )
{
   return( hypre_PCGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*==========================================================================*/
/*==========================================================================*/
/** Gets the matrix data.  Completes any remaining tasks needed for the
    DiagScale such as ????. 

{\bf Input files:}
headers.h

@return integer 0

@param solver []
  solver structure
@param A []
  structured matrix
@param b []
  structured vector
@param x []
  structured vector

@see HYPRE_StructDiagScale */
/*--------------------------------------------------------------------------*/

int 
HYPRE_StructDiagScaleSetup( HYPRE_StructSolver solver,
                            HYPRE_StructMatrix A,
                            HYPRE_StructVector y,
                            HYPRE_StructVector x      )
{
   return 0;
}

/*==========================================================================*/
/*==========================================================================*/
/** ????.

{\bf Input files:}
headers.h

@return integer error code

@param solver []
  solver structure
@param HA []
  structured solver
@param Hy []
  structured vector
@param Hx []
  structured vector

@see HYPRE_StructDiagScaleSetup */
/*--------------------------------------------------------------------------*/
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

