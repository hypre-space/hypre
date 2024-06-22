/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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

#include "_hypre_sstruct_ls.h"

typedef HYPRE_Int (*HYPRE_PtrToVoid1Fcn)(void*);
typedef HYPRE_Int (*HYPRE_PtrToVoid4Fcn)(void*, void*, void*, void*);

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct hypre_SStructSolver_struct
{
   hypre_SStructVector     *y;

   HYPRE_Int                nparts;
   HYPRE_Int               *nvars;

   void                 ****smatvec_data;

   HYPRE_PtrToVoid1Fcn    **ssolver_destroy;
   HYPRE_PtrToVoid4Fcn    **ssolver_solve;
   void                  ***ssolver_data;

   HYPRE_Real               tol;
   HYPRE_Int                max_iter;
   HYPRE_Int                zero_guess;
   HYPRE_Int                num_iterations;
   HYPRE_Real               rel_norm;
   HYPRE_Int                ssolver;

   void                    *matvec_data;

} hypre_SStructSolver;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSplitCreate( MPI_Comm             comm,
                          HYPRE_SStructSolver *solver_ptr )
{
   HYPRE_UNUSED_VAR(comm);

   hypre_SStructSolver *solver;

   solver = hypre_TAlloc(hypre_SStructSolver,  1, HYPRE_MEMORY_HOST);

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
   (solver -> matvec_data)     = NULL;

   *solver_ptr = solver;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSplitDestroy( HYPRE_SStructSolver solver )
{
   hypre_SStructVector     *y;
   HYPRE_Int                nparts;
   HYPRE_Int               *nvars;
   void                 ****smatvec_data;
   HYPRE_PtrToVoid4Fcn    **ssolver_solve;
   HYPRE_PtrToVoid1Fcn    **ssolver_destroy;
   void                  ***ssolver_data;

   HYPRE_PtrToVoid1Fcn      sdestroy;
   void                    *sdata;

   HYPRE_Int                part, vi, vj;

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
            hypre_TFree(smatvec_data[part][vi], HYPRE_MEMORY_HOST);
            sdestroy = ssolver_destroy[part][vi];
            sdata = ssolver_data[part][vi];
            sdestroy(sdata);
         }
         hypre_TFree(smatvec_data[part], HYPRE_MEMORY_HOST);
         hypre_TFree(ssolver_solve[part], HYPRE_MEMORY_HOST);
         hypre_TFree(ssolver_destroy[part], HYPRE_MEMORY_HOST);
         hypre_TFree(ssolver_data[part], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(nvars, HYPRE_MEMORY_HOST);
      hypre_TFree(smatvec_data, HYPRE_MEMORY_HOST);
      hypre_TFree(ssolver_solve, HYPRE_MEMORY_HOST);
      hypre_TFree(ssolver_destroy, HYPRE_MEMORY_HOST);
      hypre_TFree(ssolver_data, HYPRE_MEMORY_HOST);
      hypre_SStructMatvecDestroy(solver -> matvec_data);
      hypre_TFree(solver, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSplitSetup( HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x )
{
   HYPRE_Int                ssolver = (solver -> ssolver);
   hypre_SStructVector     *y;
   HYPRE_Int                nparts;
   HYPRE_Int               *nvars;
   void                 ****smatvec_data;
   HYPRE_PtrToVoid4Fcn    **ssolver_solve;
   HYPRE_PtrToVoid1Fcn    **ssolver_destroy;
   void                  ***ssolver_data;

   MPI_Comm                 comm;
   hypre_SStructGrid       *grid;
   hypre_SStructPMatrix    *pA;
   hypre_SStructPVector    *px;
   hypre_SStructPVector    *py;
   hypre_StructMatrix      *sA;
   hypre_StructVector      *sx;
   hypre_StructVector      *sy;
   HYPRE_StructMatrix      sAH;
   HYPRE_StructVector      sxH;
   HYPRE_StructVector      syH;

   HYPRE_PtrToVoid4Fcn      ssolve;
   HYPRE_PtrToVoid1Fcn      sdestroy;
   void                    *sdata;

   HYPRE_Int                part, vi, vj;

   comm = hypre_SStructVectorComm(b);
   grid = hypre_SStructVectorGrid(b);
   HYPRE_SStructVectorCreate(comm, grid, &y);
   HYPRE_SStructVectorInitialize(y);
   HYPRE_SStructVectorAssemble(y);

   nparts = hypre_SStructMatrixNParts(A);
   nvars  = hypre_TAlloc(HYPRE_Int, nparts, HYPRE_MEMORY_HOST);

   smatvec_data    = hypre_TAlloc(void***, nparts, HYPRE_MEMORY_HOST);
   ssolver_solve   = hypre_TAlloc(HYPRE_PtrToVoid4Fcn*, nparts, HYPRE_MEMORY_HOST);
   ssolver_destroy = hypre_TAlloc(HYPRE_PtrToVoid1Fcn*, nparts, HYPRE_MEMORY_HOST);
   ssolver_data    = hypre_TAlloc(void**,  nparts, HYPRE_MEMORY_HOST);

   for (part = 0; part < nparts; part++)
   {
      pA = hypre_SStructMatrixPMatrix(A, part);
      px = hypre_SStructVectorPVector(x, part);
      py = hypre_SStructVectorPVector(y, part);
      nvars[part] = hypre_SStructPMatrixNVars(pA);

      smatvec_data[part]    = hypre_TAlloc(void**, nvars[part], HYPRE_MEMORY_HOST);
      ssolver_solve[part]   = hypre_TAlloc(HYPRE_PtrToVoid4Fcn, nvars[part], HYPRE_MEMORY_HOST);
      ssolver_destroy[part] = hypre_TAlloc(HYPRE_PtrToVoid1Fcn, nvars[part], HYPRE_MEMORY_HOST);
      ssolver_data[part]    = hypre_TAlloc(void*, nvars[part], HYPRE_MEMORY_HOST);
      for (vi = 0; vi < nvars[part]; vi++)
      {
         smatvec_data[part][vi] = hypre_TAlloc(void*,  nvars[part], HYPRE_MEMORY_HOST);
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
         sAH = (HYPRE_StructMatrix) sA;
         sxH = (HYPRE_StructVector) sx;
         syH = (HYPRE_StructVector) sy;
         switch (ssolver)
         {
            default:
               /* If no solver is matched, use Jacobi, but throw and error */
               if (ssolver != HYPRE_Jacobi)
               {
                  hypre_error(HYPRE_ERROR_GENERIC);
               }
            /* fall through */

            case HYPRE_Jacobi:
               HYPRE_StructJacobiCreate(comm, (HYPRE_StructSolver *)&sdata);
               HYPRE_StructJacobiSetMaxIter((HYPRE_StructSolver)sdata, 1);
               HYPRE_StructJacobiSetTol((HYPRE_StructSolver)sdata, 0.0);
               if (solver -> zero_guess)
               {
                  HYPRE_StructJacobiSetZeroGuess((HYPRE_StructSolver)sdata);
               }
               HYPRE_StructJacobiSetup((HYPRE_StructSolver)sdata, sAH, syH, sxH);
               ssolve   = (HYPRE_PtrToVoid4Fcn) HYPRE_StructJacobiSolve;
               sdestroy = (HYPRE_PtrToVoid1Fcn) HYPRE_StructJacobiDestroy;
               break;

            case HYPRE_SMG:
               HYPRE_StructSMGCreate(comm, (HYPRE_StructSolver *)&sdata);
               HYPRE_StructSMGSetMemoryUse((HYPRE_StructSolver)sdata, 0);
               HYPRE_StructSMGSetMaxIter((HYPRE_StructSolver)sdata, 1);
               HYPRE_StructSMGSetTol((HYPRE_StructSolver)sdata, 0.0);
               if (solver -> zero_guess)
               {
                  HYPRE_StructSMGSetZeroGuess((HYPRE_StructSolver)sdata);
               }
               HYPRE_StructSMGSetNumPreRelax((HYPRE_StructSolver)sdata, 1);
               HYPRE_StructSMGSetNumPostRelax((HYPRE_StructSolver)sdata, 1);
               HYPRE_StructSMGSetLogging((HYPRE_StructSolver)sdata, 0);
               HYPRE_StructSMGSetPrintLevel((HYPRE_StructSolver)sdata, 0);
               HYPRE_StructSMGSetup((HYPRE_StructSolver)sdata, sAH, syH, sxH);
               ssolve   = (HYPRE_PtrToVoid4Fcn) HYPRE_StructSMGSolve;
               sdestroy = (HYPRE_PtrToVoid1Fcn) HYPRE_StructSMGDestroy;
               break;

            case HYPRE_PFMG:
               HYPRE_StructPFMGCreate(comm, (HYPRE_StructSolver *)&sdata);
               HYPRE_StructPFMGSetMaxIter((HYPRE_StructSolver)sdata, 1);
               HYPRE_StructPFMGSetTol((HYPRE_StructSolver)sdata, 0.0);
               if (solver -> zero_guess)
               {
                  HYPRE_StructPFMGSetZeroGuess((HYPRE_StructSolver)sdata);
               }
               HYPRE_StructPFMGSetRelaxType((HYPRE_StructSolver)sdata, 1);
               HYPRE_StructPFMGSetNumPreRelax((HYPRE_StructSolver)sdata, 1);
               HYPRE_StructPFMGSetNumPostRelax((HYPRE_StructSolver)sdata, 1);
               HYPRE_StructPFMGSetLogging((HYPRE_StructSolver)sdata, 0);
               HYPRE_StructPFMGSetPrintLevel((HYPRE_StructSolver)sdata, 0);
               HYPRE_StructPFMGSetup((HYPRE_StructSolver)sdata, sAH, syH, sxH);
               ssolve   = (HYPRE_PtrToVoid4Fcn) HYPRE_StructPFMGSolve;
               sdestroy = (HYPRE_PtrToVoid1Fcn) HYPRE_StructPFMGDestroy;
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
   if ((solver -> tol) > 0.0)
   {
      hypre_SStructMatvecCreate(&(solver -> matvec_data));
      hypre_SStructMatvecSetup((solver -> matvec_data), A, x);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSplitSolve( HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x )
{
   hypre_SStructVector     *y                     = (solver -> y);
   HYPRE_Int                nparts                = (solver -> nparts);
   HYPRE_Int               *nvars                 = (solver -> nvars);
   void                 ****smatvec_data          = (solver -> smatvec_data);
   HYPRE_PtrToVoid4Fcn    **ssolver_solve         = (solver -> ssolver_solve);
   void                  ***ssolver_data          = (solver -> ssolver_data);
   HYPRE_Real               tol                   = (solver -> tol);
   HYPRE_Int                max_iter              = (solver -> max_iter);
   HYPRE_Int                zero_guess            = (solver -> zero_guess);
   void                    *matvec_data           = (solver -> matvec_data);

   hypre_SStructPMatrix    *pA;
   hypre_SStructPVector    *px;
   hypre_SStructPVector    *py;
   hypre_StructMatrix      *sA;
   hypre_StructVector      *sx;
   hypre_StructVector      *sy;

   HYPRE_PtrToVoid4Fcn      ssolve;
   void                    *sdata;
   hypre_ParCSRMatrix      *parcsrA;
   hypre_ParVector         *parx;
   hypre_ParVector         *pary;

   HYPRE_Int                iter, part, vi, vj;
   HYPRE_Real               b_dot_b = 0, r_dot_r;

   /* part of convergence check */
   if (tol > 0.0)
   {
      /* eps = (tol^2) */
      hypre_SStructInnerProd(b, b, &b_dot_b);

      /* if rhs is zero, return a zero solution */
      if (b_dot_b == 0.0)
      {
         hypre_SStructVectorSetConstantValues(x, 0.0);
         (solver -> rel_norm) = 0.0;

         return hypre_error_flag;
      }
   }

   for (iter = 0; iter < max_iter; iter++)
   {
      /* convergence check */
      if (tol > 0.0)
      {
         /* compute fine grid residual (b - Ax) */
         hypre_SStructCopy(b, y);
         hypre_SStructMatvecCompute(matvec_data, -1.0, A, x, 1.0, y);
         hypre_SStructInnerProd(y, y, &r_dot_r);
         (solver -> rel_norm) = hypre_sqrt(r_dot_r / b_dot_b);

         if ((solver -> rel_norm) < tol)
         {
            break;
         }
      }

      /* copy b into y */
      hypre_SStructCopy(b, y);

      /* compute y = y + Nx */
      if (!zero_guess || (iter > 0))
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

            ssolve(sdata, (void*) sA, (void*) sy, (void*) sx);
         }
      }
   }

   (solver -> num_iterations) = iter;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSplitSetTol( HYPRE_SStructSolver solver,
                          HYPRE_Real          tol )
{
   (solver -> tol) = tol;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSplitSetMaxIter( HYPRE_SStructSolver solver,
                              HYPRE_Int           max_iter )
{
   (solver -> max_iter) = max_iter;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSplitSetZeroGuess( HYPRE_SStructSolver solver )
{
   (solver -> zero_guess) = 1;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSplitSetNonZeroGuess( HYPRE_SStructSolver solver )
{
   (solver -> zero_guess) = 0;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSplitSetStructSolver( HYPRE_SStructSolver solver,
                                   HYPRE_Int           ssolver )
{
   (solver -> ssolver) = ssolver;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSplitGetNumIterations( HYPRE_SStructSolver  solver,
                                    HYPRE_Int           *num_iterations )
{
   *num_iterations = (solver -> num_iterations);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSplitGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                                HYPRE_Real          *norm )
{
   *norm = (solver -> rel_norm);
   return hypre_error_flag;
}
