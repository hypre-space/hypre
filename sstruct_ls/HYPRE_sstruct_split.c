/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_SStructSplit solver interface
 *
 * This solver does the following iteration:
 *
 *    x_{k+1} = M^{-1} (b + N x_k) ,
 *
 * where A = M - N is a splitting of A, and M is the block-diagonal
 * matrix of structured intra-variable couplings.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SStructSolver data structure
 *--------------------------------------------------------------------------*/

typedef struct hypre_SStructSolver_struct
{
   hypre_SStructVector     *y;

   int                      nparts;
   int                     *nvars;

   void                 ****smatvec_data;

   int                  (***ssolver_solve)();
   int                  (***ssolver_destroy)();
   void                  ***ssolver_data;

   double                   tol;
   int                      max_iter;
   int                      zero_guess;
   int                      num_iterations;
   double                   rel_norm;
   int                      ssolver;

} hypre_SStructSolver;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructSplitCreate( MPI_Comm             comm,
                          HYPRE_SStructSolver *solver_ptr )
{
   int ierr = 0;
   hypre_SStructSolver *solver;

   solver = hypre_TAlloc(hypre_SStructSolver, 1);

   (solver -> y)               = NULL;
   (solver -> nparts)          = 0;
   (solver -> nvars)           = 0;
   (solver -> smatvec_data)    = NULL;
   (solver -> ssolver_solve)   = NULL;
   (solver -> ssolver_destroy) = NULL;
   (solver -> ssolver_data)    = NULL;
   (solver -> tol)             = 1.0e-06;
   (solver -> max_iter)        = 200;
   (solver -> zero_guess)      = 0;
   (solver -> num_iterations)  = 0;
   (solver -> rel_norm)        = 0;
   (solver -> ssolver)         = HYPRE_SMG;

   *solver_ptr = solver;

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructSplitDestroy( HYPRE_SStructSolver solver )
{
   int ierr = 0;

   hypre_SStructVector     *y;
   int                      nparts;
   int                     *nvars;
   void                 ****smatvec_data;
   int                  (***ssolver_solve)();
   int                  (***ssolver_destroy)();
   void                  ***ssolver_data;

   int                    (*sdestroy)();
   void                    *sdata;

   int                      part, vi, vj;

   if (solver)
   {
      y               = (solver -> y);
      nparts          = (solver -> nparts);
      nvars           = (solver -> nvars);
      smatvec_data    = (solver -> smatvec_data);
      ssolver_solve   = (solver -> ssolver_solve);
      ssolver_destroy = (solver -> ssolver_destroy);
      ssolver_data    = (solver -> ssolver_data);

      HYPRE_SStructVectorDestroy(y);
      for (part = 0; part < nparts; part++)
      {
         for (vi = 0; vi < nvars[part]; vi++)
         {
            for (vj = 0; vj < nvars[part]; vj++)
            {
               if (smatvec_data[part][vi][vj] != NULL)
               {
                  hypre_StructMatvecDestroy(smatvec_data[part][vi][vj]);
               }
            }
            hypre_TFree(smatvec_data[part][vi]);
            sdestroy = ssolver_destroy[part][vi];
            sdata = ssolver_data[part][vi];
            sdestroy(sdata);
         }
         hypre_TFree(smatvec_data[part]);
         hypre_TFree(ssolver_solve[part]);
         hypre_TFree(ssolver_destroy[part]);
         hypre_TFree(ssolver_data[part]);
      }
      hypre_TFree(nvars);
      hypre_TFree(smatvec_data);
      hypre_TFree(ssolver_solve);
      hypre_TFree(ssolver_destroy);
      hypre_TFree(ssolver_data);
      hypre_TFree(solver);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructSplitSetup( HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x )
{
   int ierr = 0;

   hypre_SStructVector     *y;
   int                      nparts;
   int                     *nvars;
   void                 ****smatvec_data;
   int                  (***ssolver_solve)();
   int                  (***ssolver_destroy)();
   void                  ***ssolver_data;
   int                      max_iter         = (solver -> max_iter);
   int                      zero_guess       = (solver -> zero_guess);
   int                      ssolver          = (solver -> ssolver);

   MPI_Comm                 comm;
   hypre_SStructGrid       *grid;
   hypre_SStructPMatrix    *pA;
   hypre_SStructPVector    *px;
   hypre_SStructPVector    *py;
   hypre_StructMatrix      *sA;
   hypre_StructVector      *sx;
   hypre_StructVector      *sy;
   int                    (*ssolve)();
   int                    (*sdestroy)();
   void                    *sdata;

   int                      part, vi, vj;

   comm = hypre_SStructVectorComm(b);
   grid = hypre_SStructVectorGrid(b);
   HYPRE_SStructVectorCreate(comm, grid, &y);
   HYPRE_SStructVectorInitialize(y);
   HYPRE_SStructVectorAssemble(y);

   nparts = hypre_SStructMatrixNParts(A);
   nvars = hypre_TAlloc(int, nparts);
   smatvec_data    = hypre_TAlloc(void ***, nparts);
   ssolver_solve   = (int (***)()) hypre_MAlloc((sizeof(int (**)()) * nparts));
   ssolver_destroy = (int (***)()) hypre_MAlloc((sizeof(int (**)()) * nparts));
   ssolver_data    = hypre_TAlloc(void **, nparts);
   for (part = 0; part < nparts; part++)
   {
      pA = hypre_SStructMatrixPMatrix(A, part);
      px = hypre_SStructVectorPVector(x, part);
      py = hypre_SStructVectorPVector(y, part);
      nvars[part] = hypre_SStructPMatrixNVars(pA);

      smatvec_data[part]    = hypre_TAlloc(void **, nvars[part]);
      ssolver_solve[part]   =
         (int (**)()) hypre_MAlloc((sizeof(int (*)()) * nvars[part]));
      ssolver_destroy[part] =
         (int (**)()) hypre_MAlloc((sizeof(int (*)()) * nvars[part]));
      ssolver_data[part]    = hypre_TAlloc(void *, nvars[part]);
      for (vi = 0; vi < nvars[part]; vi++)
      {
         smatvec_data[part][vi] = hypre_TAlloc(void *, nvars[part]);
         for (vj = 0; vj < nvars[part]; vj++)
         {
            sA = hypre_SStructPMatrixSMatrix(pA, vi, vj);
            sx = hypre_SStructPVectorSVector(px, vj);
            smatvec_data[part][vi][vj] = NULL;
            if (sA != NULL)
            {
               smatvec_data[part][vi][vj] = hypre_StructMatvecCreate();
               hypre_StructMatvecSetup(smatvec_data[part][vi][vj], sA, sx);
            }
         }

         sA = hypre_SStructPMatrixSMatrix(pA, vi, vi);
         sx = hypre_SStructPVectorSVector(px, vi);
         sy = hypre_SStructPVectorSVector(py, vi);
         switch(ssolver)
         {
            case HYPRE_SMG:
               HYPRE_StructSMGCreate(comm, (HYPRE_StructSolver *)&sdata);
               HYPRE_StructSMGSetMemoryUse(sdata, 0);
               HYPRE_StructSMGSetMaxIter(sdata, 1);
               HYPRE_StructSMGSetTol(sdata, 0.0);
               HYPRE_StructSMGSetZeroGuess(sdata);
               HYPRE_StructSMGSetNumPreRelax(sdata, 1);
               HYPRE_StructSMGSetNumPostRelax(sdata, 1);
               HYPRE_StructSMGSetLogging(sdata, 0);
               HYPRE_StructSMGSetup(sdata, sA, sy, sx);
               ssolve = HYPRE_StructSMGSolve;
               sdestroy = HYPRE_StructSMGDestroy;
               break;
            case HYPRE_PFMG:
               HYPRE_StructPFMGCreate(comm, (HYPRE_StructSolver *)&sdata);
               HYPRE_StructPFMGSetMaxIter(sdata, 1);
               HYPRE_StructPFMGSetTol(sdata, 0.0);
               HYPRE_StructPFMGSetZeroGuess(sdata);
               HYPRE_StructPFMGSetRelaxType(sdata, 1);
               HYPRE_StructPFMGSetNumPreRelax(sdata, 1);
               HYPRE_StructPFMGSetNumPostRelax(sdata, 1);
               HYPRE_StructPFMGSetLogging(sdata, 0);
               HYPRE_StructPFMGSetup(sdata, sA, sy, sx);
               ssolve = HYPRE_StructPFMGSolve;
               sdestroy = HYPRE_StructPFMGDestroy;
               break;
         }
         ssolver_solve[part][vi]   = ssolve;
         ssolver_destroy[part][vi] = sdestroy;
         ssolver_data[part][vi]    = sdata;
      }
   }

   (solver -> y)               = y;
   (solver -> nparts)          = nparts;
   (solver -> nvars)           = nvars;
   (solver -> smatvec_data)    = smatvec_data;
   (solver -> ssolver_solve)   = ssolver_solve;
   (solver -> ssolver_destroy) = ssolver_destroy;
   (solver -> ssolver_data)    = ssolver_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int 
HYPRE_SStructSplitSolve( HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x )
{
   int ierr = 0;

   hypre_SStructVector     *y                = (solver -> y);
   int                      nparts           = (solver -> nparts);
   int                     *nvars            = (solver -> nvars);
   void                 ****smatvec_data     = (solver -> smatvec_data);
   int                  (***ssolver_solve)() = (solver -> ssolver_solve);
   void                  ***ssolver_data     = (solver -> ssolver_data);
   int                      max_iter         = (solver -> max_iter);
   int                      zero_guess       = (solver -> zero_guess);

   hypre_SStructPMatrix    *pA;
   hypre_SStructPVector    *px;
   hypre_SStructPVector    *py;
   hypre_StructMatrix      *sA;
   hypre_StructVector      *sx;
   hypre_StructVector      *sy;
   int                    (*ssolve)();
   void                    *sdata;
   hypre_ParCSRMatrix      *parcsrA;
   hypre_ParVector         *parx;
   hypre_ParVector         *pary;

   int                      iter, part, vi, vj;

   for (iter = 0; iter < max_iter; iter++)
   {
      /* copy b into y */
      hypre_SStructCopy(b, y);

      /* compute y = y + Nx */
      if (!zero_guess)
      {
         for (part = 0; part < nparts; part++)
         {
            pA = hypre_SStructMatrixPMatrix(A, part);
            px = hypre_SStructVectorPVector(x, part);
            py = hypre_SStructVectorPVector(y, part);
            for (vi = 0; vi < nvars[part]; vi++)
            {
               for (vj = 0; vj < nvars[part]; vj++)
               {
                  sdata = smatvec_data[part][vi][vj];
                  sy = hypre_SStructPVectorSVector(py, vi);
                  if ((sdata != NULL) && (vj != vi))
                  {
                     sA = hypre_SStructPMatrixSMatrix(pA, vi, vj);
                     sx = hypre_SStructPVectorSVector(px, vj);
                     hypre_StructMatvecCompute(sdata, -1.0, sA, sx, 1.0, sy);
                  }
               }
            }
         }
         parcsrA = hypre_SStructMatrixParCSRMatrix(A);
         hypre_SStructVectorConvert(x, &parx);
         hypre_SStructVectorConvert(y, &pary);
         hypre_ParCSRMatrixMatvec(-1.0, parcsrA, parx, 1.0, pary);
         hypre_SStructVectorRestore(x, NULL);
         hypre_SStructVectorRestore(y, pary);
      }

      /* compute x = M^{-1} y */
      for (part = 0; part < nparts; part++)
      {
         pA = hypre_SStructMatrixPMatrix(A, part);
         px = hypre_SStructVectorPVector(x, part);
         py = hypre_SStructVectorPVector(y, part);
         for (vi = 0; vi < nvars[part]; vi++)
         {
            ssolve = ssolver_solve[part][vi];
            sdata  = ssolver_data[part][vi];
            sA = hypre_SStructPMatrixSMatrix(pA, vi, vi);
            sx = hypre_SStructPVectorSVector(px, vi);
            sy = hypre_SStructPVectorSVector(py, vi);
            ssolve(sdata, sA, sy, sx);
         }
      }
   }

   (solver -> num_iterations) = iter;

   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructSplitSetTol( HYPRE_SStructSolver solver,
                          double              tol )
{
   int ierr = 0;
   (solver -> tol) = tol;
   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructSplitSetMaxIter( HYPRE_SStructSolver solver,
                              int                 max_iter )
{
   int ierr = 0;
   (solver -> max_iter) = max_iter;
   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructSplitSetZeroGuess( HYPRE_SStructSolver solver )
{
   int ierr = 0;
   (solver -> zero_guess) = 1;
   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructSplitSetNonZeroGuess( HYPRE_SStructSolver solver )
{
   int ierr = 0;
   (solver -> zero_guess) = 0;
   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructSplitSetStructSolver( HYPRE_SStructSolver solver,
                                   int                 ssolver )
{
   int ierr = 0;
   (solver -> ssolver) = ssolver;
   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructSplitGetNumIterations( HYPRE_SStructSolver  solver,
                                    int                 *num_iterations )
{
   int ierr = 0;
   *num_iterations = (solver -> num_iterations);
   return ierr;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

int
HYPRE_SStructSplitGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                                double              *norm )
{
   int ierr = 0;
   return ierr;
}
