/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/





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

   HYPRE_Int                nparts;
   HYPRE_Int               *nvars;

   void                 ****smatvec_data;

   HYPRE_Int            (***ssolver_solve)();
   HYPRE_Int            (***ssolver_destroy)();
   void                  ***ssolver_data;

   double                   tol;
   HYPRE_Int                max_iter;
   HYPRE_Int                zero_guess;
   HYPRE_Int                num_iterations;
   double                   rel_norm;
   HYPRE_Int                ssolver;

   void                    *matvec_data;

} hypre_SStructSolver;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructSplitCreate( MPI_Comm             comm,
                          HYPRE_SStructSolver *solver_ptr )
{
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
   HYPRE_Int            (***ssolver_solve)();
   HYPRE_Int            (***ssolver_destroy)();
   void                  ***ssolver_data;

   HYPRE_Int              (*sdestroy)();
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
      hypre_SStructMatvecDestroy(solver -> matvec_data);
      hypre_TFree(solver);
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
   hypre_SStructVector     *y;
   HYPRE_Int                nparts;
   HYPRE_Int               *nvars;
   void                 ****smatvec_data;
   HYPRE_Int            (***ssolver_solve)();
   HYPRE_Int            (***ssolver_destroy)();
   void                  ***ssolver_data;
   HYPRE_Int                ssolver          = (solver -> ssolver);

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
   HYPRE_Int              (*ssolve)();
   HYPRE_Int              (*sdestroy)();
   void                    *sdata;

   HYPRE_Int                part, vi, vj;

   comm = hypre_SStructVectorComm(b);
   grid = hypre_SStructVectorGrid(b);
   HYPRE_SStructVectorCreate(comm, grid, &y);
   HYPRE_SStructVectorInitialize(y);
   HYPRE_SStructVectorAssemble(y);

   nparts = hypre_SStructMatrixNParts(A);
   nvars = hypre_TAlloc(HYPRE_Int, nparts);
   smatvec_data    = hypre_TAlloc(void ***, nparts);
   ssolver_solve   = (HYPRE_Int (***)()) hypre_MAlloc((sizeof(HYPRE_Int (**)()) * nparts));
   ssolver_destroy = (HYPRE_Int (***)()) hypre_MAlloc((sizeof(HYPRE_Int (**)()) * nparts));
   ssolver_data    = hypre_TAlloc(void **, nparts);
   for (part = 0; part < nparts; part++)
   {
      pA = hypre_SStructMatrixPMatrix(A, part);
      px = hypre_SStructVectorPVector(x, part);
      py = hypre_SStructVectorPVector(y, part);
      nvars[part] = hypre_SStructPMatrixNVars(pA);

      smatvec_data[part]    = hypre_TAlloc(void **, nvars[part]);
      ssolver_solve[part]   =
         (HYPRE_Int (**)()) hypre_MAlloc((sizeof(HYPRE_Int (*)()) * nvars[part]));
      ssolver_destroy[part] =
         (HYPRE_Int (**)()) hypre_MAlloc((sizeof(HYPRE_Int (*)()) * nvars[part]));
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
         sAH = (HYPRE_StructMatrix) sA;
         sxH = (HYPRE_StructVector) sx;
         syH = (HYPRE_StructVector) sy;
         switch(ssolver)
         {
            case HYPRE_Jacobi:
               HYPRE_StructJacobiCreate(comm, (HYPRE_StructSolver *)&sdata);
               HYPRE_StructJacobiSetMaxIter(sdata, 1);
               HYPRE_StructJacobiSetTol(sdata, 0.0);
               if (solver -> zero_guess)
               {
                  HYPRE_StructJacobiSetZeroGuess(sdata);
               }
               HYPRE_StructJacobiSetup(sdata, sAH, syH, sxH);
               ssolve = HYPRE_StructJacobiSolve;
               sdestroy = HYPRE_StructJacobiDestroy;
               break;
            case HYPRE_SMG:
               HYPRE_StructSMGCreate(comm, (HYPRE_StructSolver *)&sdata);
               HYPRE_StructSMGSetMemoryUse(sdata, 0);
               HYPRE_StructSMGSetMaxIter(sdata, 1);
               HYPRE_StructSMGSetTol(sdata, 0.0);
               if (solver -> zero_guess)
               {
                  HYPRE_StructSMGSetZeroGuess(sdata);
               }
               HYPRE_StructSMGSetNumPreRelax(sdata, 1);
               HYPRE_StructSMGSetNumPostRelax(sdata, 1);
               HYPRE_StructSMGSetLogging(sdata, 0);
               HYPRE_StructSMGSetPrintLevel(sdata, 0);
               HYPRE_StructSMGSetup(sdata, sAH, syH, sxH);
               ssolve = HYPRE_StructSMGSolve;
               sdestroy = HYPRE_StructSMGDestroy;
               break;
            case HYPRE_PFMG:
               HYPRE_StructPFMGCreate(comm, (HYPRE_StructSolver *)&sdata);
               HYPRE_StructPFMGSetMaxIter(sdata, 1);
               HYPRE_StructPFMGSetTol(sdata, 0.0);
               if (solver -> zero_guess)
               {
                  HYPRE_StructPFMGSetZeroGuess(sdata);
               }
               HYPRE_StructPFMGSetRelaxType(sdata, 1);
               HYPRE_StructPFMGSetNumPreRelax(sdata, 1);
               HYPRE_StructPFMGSetNumPostRelax(sdata, 1);
               HYPRE_StructPFMGSetLogging(sdata, 0);
               HYPRE_StructPFMGSetPrintLevel(sdata, 0);
               HYPRE_StructPFMGSetup(sdata, sAH, syH, sxH);
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
   hypre_SStructVector     *y                = (solver -> y);
   HYPRE_Int                nparts           = (solver -> nparts);
   HYPRE_Int               *nvars            = (solver -> nvars);
   void                 ****smatvec_data     = (solver -> smatvec_data);
   HYPRE_Int            (***ssolver_solve)() = (solver -> ssolver_solve);
   void                  ***ssolver_data     = (solver -> ssolver_data);
   double                   tol              = (solver -> tol);
   HYPRE_Int                max_iter         = (solver -> max_iter);
   HYPRE_Int                zero_guess       = (solver -> zero_guess);
   void                    *matvec_data      = (solver -> matvec_data);

   hypre_SStructPMatrix    *pA;
   hypre_SStructPVector    *px;
   hypre_SStructPVector    *py;
   hypre_StructMatrix      *sA;
   hypre_StructVector      *sx;
   hypre_StructVector      *sy;
   HYPRE_Int              (*ssolve)();
   void                    *sdata;
   hypre_ParCSRMatrix      *parcsrA;
   hypre_ParVector         *parx;
   hypre_ParVector         *pary;

   HYPRE_Int                iter, part, vi, vj;
   double                   b_dot_b, r_dot_r;



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
         (solver -> rel_norm) = sqrt(r_dot_r/b_dot_b);

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
            ssolve(sdata, sA, sy, sx);
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
                          double              tol )
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
                                                double              *norm )
{
   *norm = (solver -> rel_norm);
   return hypre_error_flag;
}
