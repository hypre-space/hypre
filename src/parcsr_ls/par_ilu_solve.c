/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * ILU solve routine
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"

/*--------------------------------------------------------------------
 * hypre_ILUSolve
 *
 * TODO (VPM): Change variable names of F_array and U_array
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSolve( void               *ilu_vdata,
                hypre_ParCSRMatrix *A,
                hypre_ParVector    *f,
                hypre_ParVector    *u )
{
   MPI_Comm              comm               = hypre_ParCSRMatrixComm(A);
   hypre_ParILUData     *ilu_data           = (hypre_ParILUData*) ilu_vdata;

   /* Matrices */
   hypre_ParCSRMatrix   *matmL              = hypre_ParILUDataMatLModified(ilu_data);
   hypre_ParCSRMatrix   *matmU              = hypre_ParILUDataMatUModified(ilu_data);
   hypre_ParCSRMatrix   *matA               = hypre_ParILUDataMatA(ilu_data);
   hypre_ParCSRMatrix   *matL               = hypre_ParILUDataMatL(ilu_data);
   hypre_ParCSRMatrix   *matU               = hypre_ParILUDataMatU(ilu_data);
   hypre_ParCSRMatrix   *matS               = hypre_ParILUDataMatS(ilu_data);
   HYPRE_Real           *matD               = hypre_ParILUDataMatD(ilu_data);
   HYPRE_Real           *matmD              = hypre_ParILUDataMatDModified(ilu_data);

   /* Vectors */
   HYPRE_Int             ilu_type           = hypre_ParILUDataIluType(ilu_data);
   HYPRE_Int            *perm               = hypre_ParILUDataPerm(ilu_data);
   HYPRE_Int            *qperm              = hypre_ParILUDataQPerm(ilu_data);
   hypre_ParVector      *F_array            = hypre_ParILUDataF(ilu_data);
   hypre_ParVector      *U_array            = hypre_ParILUDataU(ilu_data);

   /* Device data */
#if defined(HYPRE_USING_GPU)
   hypre_CSRMatrix      *matALU_d           = hypre_ParILUDataMatAILUDevice(ilu_data);
   hypre_CSRMatrix      *matBLU_d           = hypre_ParILUDataMatBILUDevice(ilu_data);
   hypre_CSRMatrix      *matE_d             = hypre_ParILUDataMatEDevice(ilu_data);
   hypre_CSRMatrix      *matF_d             = hypre_ParILUDataMatFDevice(ilu_data);
   hypre_ParCSRMatrix   *Aperm              = hypre_ParILUDataAperm(ilu_data);
   hypre_Vector         *Adiag_diag         = hypre_ParILUDataADiagDiag(ilu_data);
   hypre_Vector         *Sdiag_diag         = hypre_ParILUDataSDiagDiag(ilu_data);
   hypre_ParVector      *Ztemp              = hypre_ParILUDataZTemp(ilu_data);
   HYPRE_Int             test_opt           = hypre_ParILUDataTestOption(ilu_data);
#endif

   /* Solver settings */
   HYPRE_Real            tol                = hypre_ParILUDataTol(ilu_data);
   HYPRE_Int             logging            = hypre_ParILUDataLogging(ilu_data);
   HYPRE_Int             print_level        = hypre_ParILUDataPrintLevel(ilu_data);
   HYPRE_Int             max_iter           = hypre_ParILUDataMaxIter(ilu_data);
   HYPRE_Int             tri_solve          = hypre_ParILUDataTriSolve(ilu_data);
   HYPRE_Int             lower_jacobi_iters = hypre_ParILUDataLowerJacobiIters(ilu_data);
   HYPRE_Int             upper_jacobi_iters = hypre_ParILUDataUpperJacobiIters(ilu_data);
   HYPRE_Real           *norms              = hypre_ParILUDataRelResNorms(ilu_data);
   hypre_ParVector      *Ftemp              = hypre_ParILUDataFTemp(ilu_data);
   hypre_ParVector      *Utemp              = hypre_ParILUDataUTemp(ilu_data);
   hypre_ParVector      *Xtemp              = hypre_ParILUDataXTemp(ilu_data);
   hypre_ParVector      *Ytemp              = hypre_ParILUDataYTemp(ilu_data);
   HYPRE_Real           *fext               = hypre_ParILUDataFExt(ilu_data);
   HYPRE_Real           *uext               = hypre_ParILUDataUExt(ilu_data);
   hypre_ParVector      *residual           = NULL;
   HYPRE_Real            alpha              = -1.0;
   HYPRE_Real            beta               = 1.0;
   HYPRE_Real            conv_factor        = 0.0;
   HYPRE_Real            resnorm            = 1.0;
   HYPRE_Real            init_resnorm       = 0.0;
   HYPRE_Real            rel_resnorm;
   HYPRE_Real            rhs_norm           = 0.0;
   HYPRE_Real            old_resnorm;
   HYPRE_Real            ieee_check         = 0.0;
   HYPRE_Real            operat_cmplxty     = hypre_ParILUDataOperatorComplexity(ilu_data);
   HYPRE_Int             Solve_err_flag;
   HYPRE_Int             iter, num_procs, my_id;

   /* problem size */
   HYPRE_Int             n                  = hypre_ParCSRMatrixNumRows(A);
   HYPRE_Int             nLU                = hypre_ParILUDataNLU(ilu_data);
   HYPRE_Int            *u_end              = hypre_ParILUDataUEnd(ilu_data);

   /* Schur system solve */
   HYPRE_Solver          schur_solver       = hypre_ParILUDataSchurSolver(ilu_data);
   HYPRE_Solver          schur_precond      = hypre_ParILUDataSchurPrecond(ilu_data);
   hypre_ParVector      *rhs                = hypre_ParILUDataRhs(ilu_data);
   hypre_ParVector      *x                  = hypre_ParILUDataX(ilu_data);

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_ParCSRMatrixMemoryLocation(A),
                                                      hypre_ParVectorMemoryLocation(f) );

   /* VPM: Placeholder check to avoid -Wunused-variable warning. TODO: remove this */
   if (exec != HYPRE_EXEC_DEVICE && exec != HYPRE_EXEC_HOST)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Need to run either on host or device!");
      return hypre_error_flag;
   }
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;

   if (logging > 1)
   {
      residual = hypre_ParILUDataResidual(ilu_data);
   }

   hypre_ParILUDataNumIterations(ilu_data) = 0;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /*-----------------------------------------------------------------------
    *    Write the solver parameters
    *-----------------------------------------------------------------------*/

   if (my_id == 0 && print_level > 1)
   {
      hypre_ILUWriteSolverParams(ilu_data);
   }

   /*-----------------------------------------------------------------------
    *    Initialize the solver error flag
    *-----------------------------------------------------------------------*/

   Solve_err_flag = 0;

   /*-----------------------------------------------------------------------
    *     write some initial info
    *-----------------------------------------------------------------------*/

   if (my_id == 0 && print_level > 1 && tol > 0.)
   {
      hypre_printf("\n\n ILU SOLVER SOLUTION INFO:\n");
   }

   /*-----------------------------------------------------------------------
    *    Compute initial residual and print
    *-----------------------------------------------------------------------*/

   if (print_level > 1 || logging > 1 || tol > 0.)
   {
      if (logging > 1)
      {
         hypre_ParVectorCopy(f, residual);
         if (tol > 0.0)
         {
            hypre_ParCSRMatrixMatvec(alpha, A, u, beta, residual);
         }
         resnorm = hypre_sqrt(hypre_ParVectorInnerProd(residual, residual));
      }
      else
      {
         hypre_ParVectorCopy(f, Ftemp);
         if (tol > 0.0)
         {
            hypre_ParCSRMatrixMatvec(alpha, A, u, beta, Ftemp);
         }
         resnorm = hypre_sqrt(hypre_ParVectorInnerProd(Ftemp, Ftemp));
      }

      /* Since it does not diminish performance, attempt to return an error flag
         and notify users when they supply bad input. */
      if (resnorm != 0.)
      {
         ieee_check = resnorm / resnorm; /* INF -> NaN conversion */
      }
      if (ieee_check != ieee_check)
      {
         /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
            for ieee_check self-equality works on all IEEE-compliant compilers/
            machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
            by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
            found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
         if (print_level > 0)
         {
            hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
            hypre_printf("ERROR -- hypre_ILUSolve: INFs and/or NaNs detected in input.\n");
            hypre_printf("User probably placed non-numerics in supplied A, x_0, or b.\n");
            hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
         }
         hypre_error(HYPRE_ERROR_GENERIC);
         HYPRE_ANNOTATE_FUNC_END;

         return hypre_error_flag;
      }

      init_resnorm = resnorm;
      rhs_norm = hypre_sqrt(hypre_ParVectorInnerProd(f, f));
      if (rhs_norm > HYPRE_REAL_EPSILON)
      {
         rel_resnorm = init_resnorm / rhs_norm;
      }
      else
      {
         /* rhs is zero, return a zero solution */
         hypre_ParVectorSetConstantValues(U_array, 0.0);
         if (logging > 0)
         {
            rel_resnorm = 0.0;
            hypre_ParILUDataFinalRelResidualNorm(ilu_data) = rel_resnorm;
         }
         HYPRE_ANNOTATE_FUNC_END;

         return hypre_error_flag;
      }
   }
   else
   {
      rel_resnorm = 1.;
   }

   if (my_id == 0 && print_level > 1)
   {
      hypre_printf("                                            relative\n");
      hypre_printf("               residual        factor       residual\n");
      hypre_printf("               --------        ------       --------\n");
      hypre_printf("    Initial    %e                 %e\n", init_resnorm,
                   rel_resnorm);
   }

   matA    = A;
   U_array = u;
   F_array = f;

   /************** Main Solver Loop - always do 1 iteration ************/
   iter = 0;

   while ((rel_resnorm >= tol || iter < 1) &&
          (iter < max_iter))
   {
      /* Do one solve on LU*e = r */
      switch (ilu_type)
      {
      case 0: case 1: default:
            /* TODO (VPM): Encapsulate host and device functions into a single one */
#if defined(HYPRE_USING_GPU)
            if (exec == HYPRE_EXEC_DEVICE)
            {
               /* Apply GPU-accelerated LU solve - BJ-ILU0 */
               if (tri_solve == 1)
               {
                  hypre_ILUSolveLUDevice(matA, matBLU_d, F_array, U_array, perm, Utemp, Ftemp);
               }
               else
               {
                  hypre_ILUSolveLUIterDevice(matA, matBLU_d, F_array, U_array, perm,
                                             Utemp, Ftemp, Ztemp, &Adiag_diag,
                                             lower_jacobi_iters, upper_jacobi_iters);

                  /* Assign this now, in case it was set in method above */
                  hypre_ParILUDataADiagDiag(ilu_data) = Adiag_diag;
               }
            }
            else
#endif
            {
               /* BJ - hypre_ilu */
               if (tri_solve == 1)
               {
                  hypre_ILUSolveLU(matA, F_array, U_array, perm, n,
                                   matL, matD, matU, Utemp, Ftemp);
               }
               else
               {
                  hypre_ILUSolveLUIter(matA, F_array, U_array, perm, n,
                                       matL, matD, matU, Utemp, Ftemp,
                                       lower_jacobi_iters, upper_jacobi_iters);
               }
            }
            break;

         case 10: case 11:
#if defined(HYPRE_USING_GPU)
            if (exec == HYPRE_EXEC_DEVICE)
            {
               /* Apply GPU-accelerated GMRES-ILU solve */
               if (tri_solve == 1)
               {
                  hypre_ILUSolveSchurGMRESDevice(matA, F_array, U_array, perm, nLU, matS,
                                                 Utemp, Ftemp, schur_solver, schur_precond,
                                                 rhs, x, u_end, matBLU_d, matE_d, matF_d);
               }
               else
               {
                  hypre_ILUSolveSchurGMRESJacIterDevice(matA, F_array, U_array, perm, nLU, matS,
                                                        Utemp, Ftemp, schur_solver, schur_precond,
                                                        rhs, x, u_end, matBLU_d, matE_d, matF_d,
                                                        Ztemp, &Adiag_diag, &Sdiag_diag,
                                                        lower_jacobi_iters, upper_jacobi_iters);

                  /* Assign this now, in case it was set in method above */
                  hypre_ParILUDataADiagDiag(ilu_data) = Adiag_diag;
                  hypre_ParILUDataSDiagDiag(ilu_data) = Sdiag_diag;
               }
            }
            else
#endif
            {
               hypre_ILUSolveSchurGMRES(matA, F_array, U_array, perm, perm, nLU,
                                        matL, matD, matU, matS, Utemp, Ftemp,
                                        schur_solver, schur_precond, rhs, x, u_end);
            }
            break;

         case 20: case 21:
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)
            if (exec == HYPRE_EXEC_DEVICE)
            {
               hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                                 "NSH+ILU solve on device runs requires unified memory!");
               return hypre_error_flag;
            }
#endif
            /* NSH+ILU */
            hypre_ILUSolveSchurNSH(matA, F_array, U_array, perm, nLU, matL, matD, matU, matS,
                                   Utemp, Ftemp, schur_solver, rhs, x, u_end);
            break;

         case 30: case 31:
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)
            if (exec == HYPRE_EXEC_DEVICE)
            {
               hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                                 "RAS+ILU solve on device runs requires unified memory!");
               return hypre_error_flag;
            }
#endif
            /* RAS */
            hypre_ILUSolveLURAS(matA, F_array, U_array, perm, matL, matD, matU,
                                Utemp, Utemp, fext, uext);
            break;

         case 40: case 41:
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)
            if (exec == HYPRE_EXEC_DEVICE)
            {
               hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                                 "ddPQ+GMRES+ILU solve on device runs requires unified memory!");
               return hypre_error_flag;
            }
#endif

            /* ddPQ + GMRES + hypre_ilu[k,t]() */
            hypre_ILUSolveSchurGMRES(matA, F_array, U_array, perm, qperm, nLU,
                                     matL, matD, matU, matS, Utemp, Ftemp,
                                     schur_solver, schur_precond, rhs, x, u_end);
            break;

         case 50:
            /* GMRES-RAP */
#if defined(HYPRE_USING_GPU)
            if (exec == HYPRE_EXEC_DEVICE)
            {
               hypre_ILUSolveRAPGMRESDevice(matA, F_array, U_array, perm, nLU, matS, Utemp, Ftemp,
                                            Xtemp, Ytemp, schur_solver, schur_precond, rhs, x,
                                            u_end, Aperm, matALU_d, matBLU_d, matE_d, matF_d,
                                            test_opt);
            }
            else
#endif
            {
               hypre_ILUSolveRAPGMRESHost(matA, F_array, U_array, perm, nLU, matL, matD, matU,
                                          matmL, matmD, matmU, Utemp, Ftemp, Xtemp, Ytemp,
                                          schur_solver, schur_precond, rhs, x, u_end);
            }
            break;
      }

      /*---------------------------------------------------------------
       *    Compute residual and residual norm
       *----------------------------------------------------------------*/

      if (print_level > 1 || logging > 1 || tol > 0.)
      {
         old_resnorm = resnorm;

         if (logging > 1)
         {
            hypre_ParVectorCopy(F_array, residual);
            hypre_ParCSRMatrixMatvec(alpha, matA, U_array, beta, residual);
            resnorm = hypre_sqrt(hypre_ParVectorInnerProd(residual, residual));
         }
         else
         {
            hypre_ParVectorCopy(F_array, Ftemp);
            hypre_ParCSRMatrixMatvec(alpha, matA, U_array, beta, Ftemp);
            resnorm = hypre_sqrt(hypre_ParVectorInnerProd(Ftemp, Ftemp));
         }

         if (old_resnorm)
         {
            conv_factor = resnorm / old_resnorm;
         }
         else
         {
            conv_factor = resnorm;
         }

         if (rhs_norm > HYPRE_REAL_EPSILON)
         {
            rel_resnorm = resnorm / rhs_norm;
         }
         else
         {
            rel_resnorm = resnorm;
         }

         norms[iter] = rel_resnorm;
      }

      ++iter;
      hypre_ParILUDataNumIterations(ilu_data) = iter;
      hypre_ParILUDataFinalRelResidualNorm(ilu_data) = rel_resnorm;

      if (my_id == 0 && print_level > 1)
      {
         hypre_printf("    ILUSolve %2d   %e    %f     %e \n", iter,
                      resnorm, conv_factor, rel_resnorm);
      }
   }

   /* check convergence within max_iter */
   if (iter == max_iter && tol > 0.)
   {
      Solve_err_flag = 1;
      hypre_error(HYPRE_ERROR_CONV);
   }

   /*-----------------------------------------------------------------------
    *    Print closing statistics
    *    Add operator and grid complexity stats
    *-----------------------------------------------------------------------*/

   if (iter > 0 && init_resnorm)
   {
      conv_factor = hypre_pow((resnorm / init_resnorm), (1.0 / (HYPRE_Real) iter));
   }
   else
   {
      conv_factor = 1.;
   }

   if (print_level > 1)
   {
      /*** compute operator and grid complexity (fill factor) here ?? ***/
      if (my_id == 0)
      {
         if (Solve_err_flag == 1)
         {
            hypre_printf("\n\n==============================================");
            hypre_printf("\n NOTE: Convergence tolerance was not achieved\n");
            hypre_printf("      within the allowed %d iterations\n", max_iter);
            hypre_printf("==============================================");
         }
         hypre_printf("\n\n Average Convergence Factor = %f \n", conv_factor);
         hypre_printf("                operator = %f\n", operat_cmplxty);
      }
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_ILUSolveSchurGMRES
 *
 * Schur Complement solve with GMRES on schur complement
 *
 * ParCSRMatrix S is already built in ilu data sturcture, here directly
 * use S, L, D and U factors only have local scope (no off-diag terms)
 * so apart from the residual calculation (which uses A), the solves
 * with the L and U factors are local.
 *
 * S is the global Schur complement
 * schur_solver is a GMRES solver
 * schur_precond is the ILU preconditioner for GMRES
 * rhs and x are helper vector for solving Schur system
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSolveSchurGMRES(hypre_ParCSRMatrix *A,
                         hypre_ParVector    *f,
                         hypre_ParVector    *u,
                         HYPRE_Int          *perm,
                         HYPRE_Int          *qperm,
                         HYPRE_Int           nLU,
                         hypre_ParCSRMatrix *L,
                         HYPRE_Real         *D,
                         hypre_ParCSRMatrix *U,
                         hypre_ParCSRMatrix *S,
                         hypre_ParVector    *ftemp,
                         hypre_ParVector    *utemp,
                         HYPRE_Solver        schur_solver,
                         HYPRE_Solver        schur_precond,
                         hypre_ParVector    *rhs,
                         hypre_ParVector    *x,
                         HYPRE_Int          *u_end)
{
   HYPRE_UNUSED_VAR(schur_precond);

   /* Data objects for L and U */
   hypre_CSRMatrix   *L_diag      = hypre_ParCSRMatrixDiag(L);
   HYPRE_Real        *L_diag_data = hypre_CSRMatrixData(L_diag);
   HYPRE_Int         *L_diag_i    = hypre_CSRMatrixI(L_diag);
   HYPRE_Int         *L_diag_j    = hypre_CSRMatrixJ(L_diag);
   hypre_CSRMatrix   *U_diag      = hypre_ParCSRMatrixDiag(U);
   HYPRE_Real        *U_diag_data = hypre_CSRMatrixData(U_diag);
   HYPRE_Int         *U_diag_i    = hypre_CSRMatrixI(U_diag);
   HYPRE_Int         *U_diag_j    = hypre_CSRMatrixJ(U_diag);

   /* Vectors */
   hypre_Vector      *utemp_local = hypre_ParVectorLocalVector(utemp);
   HYPRE_Real        *utemp_data  = hypre_VectorData(utemp_local);
   hypre_Vector      *ftemp_local = hypre_ParVectorLocalVector(ftemp);
   HYPRE_Real        *ftemp_data  = hypre_VectorData(ftemp_local);
   HYPRE_Real         alpha       = -1.0;
   HYPRE_Real         beta        = 1.0;
   HYPRE_Int          i, j, k1, k2, col;

   /* Problem size */
   HYPRE_Int          n           = hypre_CSRMatrixNumRows(L_diag);
   hypre_Vector      *rhs_local;
   HYPRE_Real        *rhs_data;
   hypre_Vector      *x_local;
   HYPRE_Real        *x_data;

   /* Compute residual */
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* 1st need to solve LBi*xi = fi
    * L solve, solve xi put in u_temp upper
    */
   /* now update with L to solve */
   for (i = 0 ; i < nLU ; i ++)
   {
      utemp_data[qperm[i]] = ftemp_data[perm[i]];
      k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
      for (j = k1 ; j < k2 ; j ++)
      {
         utemp_data[qperm[i]] -= L_diag_data[j] * utemp_data[qperm[L_diag_j[j]]];
      }
   }

   /* 2nd need to compute g'i = gi - Ei*UBi^-1*xi
    * now put g'i into the f_temp lower
    */
   for (i = nLU ; i < n ; i ++)
   {
      k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = L_diag_j[j];
         ftemp_data[perm[i]] -= L_diag_data[j] * utemp_data[qperm[col]];
      }
   }

   /* 3rd need to solve global Schur Complement Sy = g'
    * for now only solve the local system
    * solve y put in u_temp lower
    * only solve whe S is not NULL
    */
   if (S)
   {
      /*initialize solution to zero for residual equation */
      hypre_ParVectorSetConstantValues(x, 0.0);

      /* setup vectors for solve */
      rhs_local   = hypre_ParVectorLocalVector(rhs);
      rhs_data    = hypre_VectorData(rhs_local);
      x_local     = hypre_ParVectorLocalVector(x);
      x_data      = hypre_VectorData(x_local);

      /* set rhs value */
      for (i = nLU ; i < n ; i ++)
      {
         rhs_data[i - nLU] = ftemp_data[perm[i]];
      }

      /* solve */
      HYPRE_GMRESSolve(schur_solver, (HYPRE_Matrix)S, (HYPRE_Vector)rhs, (HYPRE_Vector)x);

      /* copy value back to original */
      for (i = nLU ; i < n ; i ++)
      {
         utemp_data[qperm[i]] = x_data[i - nLU];
      }
   }

   /* 4th need to compute zi = xi - LBi^-1*Fi*yi
    * put zi in f_temp upper
    * only do this computation when nLU < n
    * U is unsorted, search is expensive when unnecessary
    */
   if (nLU < n)
   {
      for (i = 0 ; i < nLU ; i ++)
      {
         ftemp_data[perm[i]] = utemp_data[qperm[i]];
         k1 = u_end[i] ; k2 = U_diag_i[i + 1];
         for (j = k1 ; j < k2 ; j ++)
         {
            col = U_diag_j[j];
            ftemp_data[perm[i]] -= U_diag_data[j] * utemp_data[qperm[col]];
         }
      }
      for (i = 0 ; i < nLU ; i ++)
      {
         utemp_data[qperm[i]] = ftemp_data[perm[i]];
      }
   }

   /* 5th need to solve UBi*ui = zi */
   /* put result in u_temp upper */
   for (i = nLU - 1 ; i >= 0 ; i --)
   {
      k1 = U_diag_i[i] ; k2 = u_end[i];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = U_diag_j[j];
         utemp_data[qperm[i]] -= U_diag_data[j] * utemp_data[qperm[col]];
      }
      utemp_data[qperm[i]] *= D[i];
   }

   /* done, now everything are in u_temp, update solution */
   hypre_ParVectorAxpy(beta, utemp, u);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_ILUSolveSchurNSH
 *
 * Newton-Schulz-Hotelling solve
 *
 * ParCSRMatrix S is already built in ilu data sturcture
 *
 * S here is the INVERSE of Schur Complement
 * L, D and U factors only have local scope (no off-diag terms)
 *  so apart from the residual calculation (which uses A), the solves
 *  with the L and U factors are local.
 * S is the inverse global Schur complement
 * rhs and x are helper vector for solving Schur system
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSolveSchurNSH(hypre_ParCSRMatrix *A,
                       hypre_ParVector    *f,
                       hypre_ParVector    *u,
                       HYPRE_Int          *perm,
                       HYPRE_Int           nLU,
                       hypre_ParCSRMatrix *L,
                       HYPRE_Real         *D,
                       hypre_ParCSRMatrix *U,
                       hypre_ParCSRMatrix *S,
                       hypre_ParVector    *ftemp,
                       hypre_ParVector    *utemp,
                       HYPRE_Solver        schur_solver,
                       hypre_ParVector    *rhs,
                       hypre_ParVector    *x,
                       HYPRE_Int          *u_end)
{
   /* data objects for L and U */
   hypre_CSRMatrix   *L_diag      = hypre_ParCSRMatrixDiag(L);
   HYPRE_Real        *L_diag_data = hypre_CSRMatrixData(L_diag);
   HYPRE_Int         *L_diag_i    = hypre_CSRMatrixI(L_diag);
   HYPRE_Int         *L_diag_j    = hypre_CSRMatrixJ(L_diag);
   hypre_CSRMatrix   *U_diag      = hypre_ParCSRMatrixDiag(U);
   HYPRE_Real        *U_diag_data = hypre_CSRMatrixData(U_diag);
   HYPRE_Int         *U_diag_i    = hypre_CSRMatrixI(U_diag);
   HYPRE_Int         *U_diag_j    = hypre_CSRMatrixJ(U_diag);

   /* Vectors */
   hypre_Vector      *utemp_local = hypre_ParVectorLocalVector(utemp);
   HYPRE_Real        *utemp_data  = hypre_VectorData(utemp_local);
   hypre_Vector      *ftemp_local = hypre_ParVectorLocalVector(ftemp);
   HYPRE_Real        *ftemp_data  = hypre_VectorData(ftemp_local);
   HYPRE_Real         alpha       = -1.0;
   HYPRE_Real         beta        = 1.0;
   HYPRE_Int          i, j, k1, k2, col;

   /* problem size */
   HYPRE_Int         n = hypre_CSRMatrixNumRows(L_diag);

   /* other data objects for computation */
   hypre_Vector      *rhs_local;
   HYPRE_Real        *rhs_data;
   hypre_Vector      *x_local;
   HYPRE_Real        *x_data;

   /* compute residual */
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* 1st need to solve LBi*xi = fi
    * L solve, solve xi put in u_temp upper
    */
   /* now update with L to solve */
   for (i = 0 ; i < nLU ; i ++)
   {
      utemp_data[perm[i]] = ftemp_data[perm[i]];
      k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
      for (j = k1 ; j < k2 ; j ++)
      {
         utemp_data[perm[i]] -= L_diag_data[j] * utemp_data[perm[L_diag_j[j]]];
      }
   }

   /* 2nd need to compute g'i = gi - Ei*UBi^-1*xi
    * now put g'i into the f_temp lower
    */
   for (i = nLU ; i < n ; i ++)
   {
      k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = L_diag_j[j];
         ftemp_data[perm[i]] -= L_diag_data[j] * utemp_data[perm[col]];
      }
   }

   /* 3rd need to solve global Schur Complement Sy = g'
    * for now only solve the local system
    * solve y put in u_temp lower
    * only solve when S is not NULL
    */
   if (S)
   {
      /* Initialize solution to zero for residual equation */
      hypre_ParVectorSetConstantValues(x, 0.0);

      /* Setup vectors for solve */
      rhs_local = hypre_ParVectorLocalVector(rhs);
      rhs_data  = hypre_VectorData(rhs_local);
      x_local   = hypre_ParVectorLocalVector(x);
      x_data    = hypre_VectorData(x_local);

      /* set rhs value */
      for (i = nLU ; i < n ; i ++)
      {
         rhs_data[i - nLU] = ftemp_data[perm[i]];
      }

      /* Solve Schur system with approx inverse
       * x = S*rhs
       */
      hypre_NSHSolve(schur_solver, S, rhs, x);

      /* copy value back to original */
      for (i = nLU ; i < n ; i ++)
      {
         utemp_data[perm[i]] = x_data[i - nLU];
      }
   }

   /* 4th need to compute zi = xi - LBi^-1*yi
    * put zi in f_temp upper
    * only do this computation when nLU < n
    * U is unsorted, search is expensive when unnecessary
    */
   if (nLU < n)
   {
      for (i = 0 ; i < nLU ; i ++)
      {
         ftemp_data[perm[i]] = utemp_data[perm[i]];
         k1 = u_end[i] ; k2 = U_diag_i[i + 1];
         for (j = k1 ; j < k2 ; j ++)
         {
            col = U_diag_j[j];
            ftemp_data[perm[i]] -= U_diag_data[j] * utemp_data[perm[col]];
         }
      }
      for (i = 0 ; i < nLU ; i ++)
      {
         utemp_data[perm[i]] = ftemp_data[perm[i]];
      }
   }

   /* 5th need to solve UBi*ui = zi */
   /* put result in u_temp upper */
   for (i = nLU - 1 ; i >= 0 ; i --)
   {
      k1 = U_diag_i[i] ; k2 = u_end[i];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = U_diag_j[j];
         utemp_data[perm[i]] -= U_diag_data[j] * utemp_data[perm[col]];
      }
      utemp_data[perm[i]] *= D[i];
   }

   /* Done, now everything are in u_temp, update solution */
   hypre_ParVectorAxpy(beta, utemp, u);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_ILUSolveLU
 *
 * Incomplete LU solve
 *
 * L, D and U factors only have local scope (no off-diagterms)
 *  so apart from the residual calculation (which uses A),
 *  the solves with the L and U factors are local.
 *
 * Note: perm contains the permutation of indexes corresponding to
 * user-prescribed reordering strategy. In the block Jacobi case, perm
 * may be NULL if no reordering is done (for performance, (perm == NULL)
 * assumes identity mapping of indexes). Hence we need to check the local
 * solves for this case and avoid segfaults. - DOK
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSolveLU(hypre_ParCSRMatrix *A,
                 hypre_ParVector    *f,
                 hypre_ParVector    *u,
                 HYPRE_Int          *perm,
                 HYPRE_Int           nLU,
                 hypre_ParCSRMatrix *L,
                 HYPRE_Real         *D,
                 hypre_ParCSRMatrix *U,
                 hypre_ParVector    *ftemp,
                 hypre_ParVector    *utemp)
{
   /* data objects for L and U */
   hypre_CSRMatrix *L_diag      = hypre_ParCSRMatrixDiag(L);
   HYPRE_Real      *L_diag_data = hypre_CSRMatrixData(L_diag);
   HYPRE_Int       *L_diag_i    = hypre_CSRMatrixI(L_diag);
   HYPRE_Int       *L_diag_j    = hypre_CSRMatrixJ(L_diag);
   hypre_CSRMatrix *U_diag      = hypre_ParCSRMatrixDiag(U);
   HYPRE_Real      *U_diag_data = hypre_CSRMatrixData(U_diag);
   HYPRE_Int       *U_diag_i    = hypre_CSRMatrixI(U_diag);
   HYPRE_Int       *U_diag_j    = hypre_CSRMatrixJ(U_diag);

   /* Vectors */
   hypre_Vector    *utemp_local = hypre_ParVectorLocalVector(utemp);
   HYPRE_Real      *utemp_data  = hypre_VectorData(utemp_local);
   hypre_Vector    *ftemp_local = hypre_ParVectorLocalVector(ftemp);
   HYPRE_Real      *ftemp_data  = hypre_VectorData(ftemp_local);
   HYPRE_Real       alpha       = -1.0;
   HYPRE_Real       beta        = 1.0;
   HYPRE_Int        i, j, k1, k2;

   /* Initialize Utemp to zero.
    * This is necessary for correctness, when we use optimized
    * vector operations in the case where sizeof(L, D or U) < sizeof(A)
    */
   //hypre_ParVectorSetConstantValues( utemp, 0.);
   /* compute residual */
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* L solve - Forward solve */
   /* copy rhs to account for diagonal of L (which is identity) */
   if (perm)
   {
      for (i = 0; i < nLU; i++)
      {
         utemp_data[perm[i]] = ftemp_data[perm[i]];
      }
   }
   else
   {
      for (i = 0; i < nLU; i++)
      {
         utemp_data[i] = ftemp_data[i];
      }
   }

   /* Update with remaining (off-diagonal) entries of L */
   if (perm)
   {
      for ( i = 0; i < nLU; i++ )
      {
         k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
         for (j = k1; j < k2; j++)
         {
            utemp_data[perm[i]] -= L_diag_data[j] * utemp_data[perm[L_diag_j[j]]];
         }
      }
   }
   else
   {
      for ( i = 0; i < nLU; i++ )
      {
         k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
         for (j = k1; j < k2; j++)
         {
            utemp_data[i] -= L_diag_data[j] * utemp_data[L_diag_j[j]];
         }
      }
   }
   /*-------------------- U solve - Backward substitution */
   if (perm)
   {
      for ( i = nLU - 1; i >= 0; i-- )
      {
         /* first update with the remaining (off-diagonal) entries of U */
         k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
         for (j = k1; j < k2; j++)
         {
            utemp_data[perm[i]] -= U_diag_data[j] * utemp_data[perm[U_diag_j[j]]];
         }

         /* diagonal scaling (contribution from D. Note: D is stored as its inverse) */
         utemp_data[perm[i]] *= D[i];
      }
   }
   else
   {
      for ( i = nLU - 1; i >= 0; i-- )
      {
         /* first update with the remaining (off-diagonal) entries of U */
         k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
         for (j = k1; j < k2; j++)
         {
            utemp_data[i] -= U_diag_data[j] * utemp_data[U_diag_j[j]];
         }

         /* diagonal scaling (contribution from D. Note: D is stored as its inverse) */
         utemp_data[i] *= D[i];
      }
   }
   /* Update solution */
   hypre_ParVectorAxpy(beta, utemp, u);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_ILUSolveLUIter
 *
 * Iterative incomplete LU solve
 *
 * L, D and U factors only have local scope (no off-diag terms)
 *  so apart from the residual calculation (which uses A), the solves
 *  with the L and U factors are local.
 *
 * Note: perm contains the permutation of indexes corresponding to
 * user-prescribed reordering strategy. In the block Jacobi case, perm
 * may be NULL if no reordering is done (for performance, (perm == NULL)
 * assumes identity mapping of indexes). Hence we need to check the local
 * solves for this case and avoid segfaults. - DOK
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSolveLUIter(hypre_ParCSRMatrix *A,
                     hypre_ParVector    *f,
                     hypre_ParVector    *u,
                     HYPRE_Int          *perm,
                     HYPRE_Int           nLU,
                     hypre_ParCSRMatrix *L,
                     HYPRE_Real         *D,
                     hypre_ParCSRMatrix *U,
                     hypre_ParVector    *ftemp,
                     hypre_ParVector    *utemp,
                     HYPRE_Int           lower_jacobi_iters,
                     HYPRE_Int           upper_jacobi_iters)
{
   /* Data objects for L and U */
   hypre_CSRMatrix *L_diag      = hypre_ParCSRMatrixDiag(L);
   HYPRE_Real      *L_diag_data = hypre_CSRMatrixData(L_diag);
   HYPRE_Int       *L_diag_i    = hypre_CSRMatrixI(L_diag);
   HYPRE_Int       *L_diag_j    = hypre_CSRMatrixJ(L_diag);
   hypre_CSRMatrix *U_diag      = hypre_ParCSRMatrixDiag(U);
   HYPRE_Real      *U_diag_data = hypre_CSRMatrixData(U_diag);
   HYPRE_Int       *U_diag_i    = hypre_CSRMatrixI(U_diag);
   HYPRE_Int       *U_diag_j    = hypre_CSRMatrixJ(U_diag);

   /* Vectors */
   hypre_Vector    *utemp_local = hypre_ParVectorLocalVector(utemp);
   HYPRE_Real      *utemp_data  = hypre_VectorData(utemp_local);
   hypre_Vector    *ftemp_local = hypre_ParVectorLocalVector(ftemp);
   HYPRE_Real      *ftemp_data  = hypre_VectorData(ftemp_local);

   /* Local variables */
   HYPRE_Real       alpha       = -1.0;
   HYPRE_Real       beta        = 1.0;
   HYPRE_Real       sum;
   HYPRE_Int        i, j, k1, k2, kk;

   /* Initialize Utemp to zero.
    * This is necessary for correctness, when we use optimized
    * vector operations in the case where sizeof(L, D or U) < sizeof(A)
    */
   //hypre_ParVectorSetConstantValues( utemp, 0.);
   /* compute residual */
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* L solve - Forward solve */
   /* copy rhs to account for diagonal of L (which is identity) */

   /* Initialize iteration to 0 */
   if (perm)
   {
      for ( i = 0; i < nLU; i++ )
      {
         utemp_data[perm[i]] = 0.0;
      }
   }
   else
   {
      for ( i = 0; i < nLU; i++ )
      {
         utemp_data[i] = 0.0;
      }
   }
   /* Jacobi iteration loop */
   for ( kk = 0; kk < lower_jacobi_iters; kk++ )
   {
      /* u^{k+1} = f - Lu^k */

      /* Do a SpMV with L and save the results in xtemp */
      if (perm)
      {
         for ( i = nLU - 1; i >= 0; i-- )
         {
            sum = 0.0;
            k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
            for (j = k1; j < k2; j++)
            {
               sum += L_diag_data[j] * utemp_data[perm[L_diag_j[j]]];
            }
            utemp_data[perm[i]] = ftemp_data[perm[i]] - sum;
         }
      }
      else
      {
         for ( i = nLU - 1; i >= 0; i-- )
         {
            sum = 0.0;
            k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
            for (j = k1; j < k2; j++)
            {
               sum += L_diag_data[j] * utemp_data[L_diag_j[j]];
            }
            utemp_data[i] = ftemp_data[i] - sum;
         }
      }
   } /* end jacobi loop */

   /* Initialize iteration to 0 */
   if (perm)
   {
      for ( i = 0; i < nLU; i++ )
      {
         ftemp_data[perm[i]] = 0.0;
      }
   }
   else
   {
      for ( i = 0; i < nLU; i++ )
      {
         ftemp_data[i] = 0.0;
      }
   }

   /* Jacobi iteration loop */
   for ( kk = 0; kk < upper_jacobi_iters; kk++ )
   {
      /* u^{k+1} = f - Uu^k */

      /* Do a SpMV with U and save the results in xtemp */
      if (perm)
      {
         for ( i = 0; i < nLU; ++i )
         {
            sum = 0.0;
            k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
            for (j = k1; j < k2; j++)
            {
               sum += U_diag_data[j] * ftemp_data[perm[U_diag_j[j]]];
            }
            ftemp_data[perm[i]] = D[i] * (utemp_data[perm[i]] - sum);
         }
      }
      else
      {
         for ( i = 0; i < nLU; ++i )
         {
            sum = 0.0;
            k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
            for (j = k1; j < k2; j++)
            {
               sum += U_diag_data[j] * ftemp_data[U_diag_j[j]];
            }
            ftemp_data[i] = D[i] * (utemp_data[i] - sum);
         }
      }
   } /* end jacobi loop */

   /* Update solution */
   hypre_ParVectorAxpy(beta, ftemp, u);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_ILUSolveLURAS
 *
 * Incomplete LU solve RAS
 *
 * L, D and U factors only have local scope (no off-diag terms)
 *  so apart from the residual calculation (which uses A), the solves
 *  with the L and U factors are local.
 * fext and uext are tempory arrays for external data
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSolveLURAS(hypre_ParCSRMatrix *A,
                    hypre_ParVector    *f,
                    hypre_ParVector    *u,
                    HYPRE_Int          *perm,
                    hypre_ParCSRMatrix *L,
                    HYPRE_Real         *D,
                    hypre_ParCSRMatrix *U,
                    hypre_ParVector    *ftemp,
                    hypre_ParVector    *utemp,
                    HYPRE_Real         *fext,
                    HYPRE_Real         *uext)
{
   /* Parallel info */
   hypre_ParCSRCommPkg        *comm_pkg;
   hypre_ParCSRCommHandle     *comm_handle;
   HYPRE_Int                   num_sends, begin, end;


   /* Data objects for L and U */
   hypre_CSRMatrix            *L_diag      = hypre_ParCSRMatrixDiag(L);
   HYPRE_Real                 *L_diag_data = hypre_CSRMatrixData(L_diag);
   HYPRE_Int                  *L_diag_i    = hypre_CSRMatrixI(L_diag);
   HYPRE_Int                  *L_diag_j    = hypre_CSRMatrixJ(L_diag);
   hypre_CSRMatrix            *U_diag      = hypre_ParCSRMatrixDiag(U);
   HYPRE_Real                 *U_diag_data = hypre_CSRMatrixData(U_diag);
   HYPRE_Int                  *U_diag_i    = hypre_CSRMatrixI(U_diag);
   HYPRE_Int                  *U_diag_j    = hypre_CSRMatrixJ(U_diag);

   /* Vectors */
   HYPRE_Int                   n           = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(A));
   HYPRE_Int                   m           = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));
   HYPRE_Int                   n_total     = m + n;
   hypre_Vector               *utemp_local = hypre_ParVectorLocalVector(utemp);
   HYPRE_Real                 *utemp_data  = hypre_VectorData(utemp_local);
   hypre_Vector               *ftemp_local = hypre_ParVectorLocalVector(ftemp);
   HYPRE_Real                 *ftemp_data  = hypre_VectorData(ftemp_local);

   /* Local variables */
   HYPRE_Int                   idx, jcol, col;
   HYPRE_Int                   i, j, k1, k2;
   HYPRE_Real                  alpha = -1.0;
   HYPRE_Real                  beta  = 1.0;

   /* prepare for communication */
   comm_pkg = hypre_ParCSRMatrixCommPkg(A);

   /* setup if not yet built */
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /* Initialize Utemp to zero.
    * This is necessary for correctness, when we use optimized
    * vector operations in the case where sizeof(L, D or U) < sizeof(A)
    */
   //hypre_ParVectorSetConstantValues( utemp, 0.);
   /* compute residual */
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* communication to get external data */

   /* get total num of send */
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   begin     = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
   end       = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);

   /* copy new index into send_buf */
   for (i = begin ; i < end ; i ++)
   {
      /* all we need is just send out data, we don't need to worry about the
       *    permutation of offd part, actually we don't need to worry about
       *    permutation at all
       * borrow uext as send buffer .
       */
      uext[i - begin] = ftemp_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)];
   }

   /* main communication */
   comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg, uext, fext);
   hypre_ParCSRCommHandleDestroy(comm_handle);

   /* L solve - Forward solve */
   for ( i = 0 ; i < n_total ; i ++)
   {
      k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
      if ( i < n )
      {
         /* diag part */
         utemp_data[perm[i]] = ftemp_data[perm[i]];
         for (j = k1; j < k2; j++)
         {
            col = L_diag_j[j];
            if ( col < n )
            {
               utemp_data[perm[i]] -= L_diag_data[j] * utemp_data[perm[col]];
            }
            else
            {
               jcol = col - n;
               utemp_data[perm[i]] -= L_diag_data[j] * uext[jcol];
            }
         }
      }
      else
      {
         /* offd part */
         idx = i - n;
         uext[idx] = fext[idx];
         for (j = k1; j < k2; j++)
         {
            col = L_diag_j[j];
            if (col < n)
            {
               uext[idx] -= L_diag_data[j] * utemp_data[perm[col]];
            }
            else
            {
               jcol = col - n;
               uext[idx] -= L_diag_data[j] * uext[jcol];
            }
         }
      }
   }

   /*-------------------- U solve - Backward substitution */
   for ( i = n_total - 1; i >= 0; i-- )
   {
      /* first update with the remaining (off-diagonal) entries of U */
      k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
      if ( i < n )
      {
         /* diag part */
         for (j = k1; j < k2; j++)
         {
            col = U_diag_j[j];
            if ( col < n )
            {
               utemp_data[perm[i]] -= U_diag_data[j] * utemp_data[perm[col]];
            }
            else
            {
               jcol = col - n;
               utemp_data[perm[i]] -= U_diag_data[j] * uext[jcol];
            }
         }
         /* diagonal scaling (contribution from D. Note: D is stored as its inverse) */
         utemp_data[perm[i]] *= D[i];
      }
      else
      {
         /* 2nd part of offd */
         idx = i - n;
         for (j = k1; j < k2; j++)
         {
            col = U_diag_j[j];
            if ( col < n )
            {
               uext[idx] -= U_diag_data[j] * utemp_data[perm[col]];
            }
            else
            {
               jcol = col - n;
               uext[idx] -= U_diag_data[j] * uext[jcol];
            }
         }
         /* diagonal scaling (contribution from D. Note: D is stored as its inverse) */
         uext[idx] *= D[i];
      }
   }

   /* Update solution */
   hypre_ParVectorAxpy(beta, utemp, u);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_ILUSolveRAPGMRESHost
 *
 * Solve with GMRES on schur complement, RAP style.
 *
 * ParCSRMatrix S is already built in ilu data sturcture, here directly
 * use S, L, D and U factors only have local scope (no off-diag terms)
 * so apart from the residual calculation (which uses A), the solves
 * with the L and U factors are local.
 *
 * S is the global Schur complement
 * schur_solver is a GMRES solver
 * schur_precond is the ILU preconditioner for GMRES
 * rhs and x are helper vector for solving Schur system
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSolveRAPGMRESHost(hypre_ParCSRMatrix *A,
                           hypre_ParVector    *f,
                           hypre_ParVector    *u,
                           HYPRE_Int          *perm,
                           HYPRE_Int           nLU,
                           hypre_ParCSRMatrix *L,
                           HYPRE_Real         *D,
                           hypre_ParCSRMatrix *U,
                           hypre_ParCSRMatrix *mL,
                           HYPRE_Real         *mD,
                           hypre_ParCSRMatrix *mU,
                           hypre_ParVector    *ftemp,
                           hypre_ParVector    *utemp,
                           hypre_ParVector    *xtemp,
                           hypre_ParVector    *ytemp,
                           HYPRE_Solver        schur_solver,
                           HYPRE_Solver        schur_precond,
                           hypre_ParVector    *rhs,
                           hypre_ParVector    *x,
                           HYPRE_Int          *u_end)
{
   /* data objects for L and U */
   hypre_CSRMatrix   *L_diag       = hypre_ParCSRMatrixDiag(L);
   HYPRE_Real        *L_diag_data  = hypre_CSRMatrixData(L_diag);
   HYPRE_Int         *L_diag_i     = hypre_CSRMatrixI(L_diag);
   HYPRE_Int         *L_diag_j     = hypre_CSRMatrixJ(L_diag);

   hypre_CSRMatrix   *U_diag       = hypre_ParCSRMatrixDiag(U);
   HYPRE_Real        *U_diag_data  = hypre_CSRMatrixData(U_diag);
   HYPRE_Int         *U_diag_i     = hypre_CSRMatrixI(U_diag);
   HYPRE_Int         *U_diag_j     = hypre_CSRMatrixJ(U_diag);

   hypre_CSRMatrix   *mL_diag      = hypre_ParCSRMatrixDiag(mL);
   HYPRE_Real        *mL_diag_data = hypre_CSRMatrixData(mL_diag);
   HYPRE_Int         *mL_diag_i    = hypre_CSRMatrixI(mL_diag);
   HYPRE_Int         *mL_diag_j    = hypre_CSRMatrixJ(mL_diag);

   hypre_CSRMatrix   *mU_diag      = hypre_ParCSRMatrixDiag(mU);
   HYPRE_Real        *mU_diag_data = hypre_CSRMatrixData(mU_diag);
   HYPRE_Int         *mU_diag_i    = hypre_CSRMatrixI(mU_diag);
   HYPRE_Int         *mU_diag_j    = hypre_CSRMatrixJ(mU_diag);

   /* Vectors */
   hypre_Vector      *utemp_local  = hypre_ParVectorLocalVector(utemp);
   HYPRE_Real        *utemp_data   = hypre_VectorData(utemp_local);
   hypre_Vector      *ftemp_local  = hypre_ParVectorLocalVector(ftemp);
   HYPRE_Real        *ftemp_data   = hypre_VectorData(ftemp_local);
   hypre_Vector      *xtemp_local  = NULL;
   HYPRE_Real        *xtemp_data   = NULL;
   hypre_Vector      *ytemp_local  = NULL;
   HYPRE_Real        *ytemp_data   = NULL;

   HYPRE_Real         alpha = -1.0;
   HYPRE_Real         beta  = 1.0;
   HYPRE_Int          i, j, k1, k2, col;

   /* problem size */
   HYPRE_Int          n = hypre_CSRMatrixNumRows(L_diag);
   HYPRE_Int          m = n - nLU;

   /* other data objects for computation */
   hypre_Vector      *rhs_local;
   HYPRE_Real        *rhs_data;
   hypre_Vector      *x_local = NULL;
   HYPRE_Real        *x_data;

   /* xtemp might be null when we have no Schur complement */
   if (xtemp)
   {
      xtemp_local = hypre_ParVectorLocalVector(xtemp);
      xtemp_data  = hypre_VectorData(xtemp_local);
      ytemp_local = hypre_ParVectorLocalVector(ytemp);
      ytemp_data  = hypre_VectorData(ytemp_local);
   }

   /* Setup vectors for solve */
   if (m > 0)
   {
      rhs_local   = hypre_ParVectorLocalVector(rhs);
      rhs_data    = hypre_VectorData(rhs_local);
      x_local     = hypre_ParVectorLocalVector(x);
      x_data      = hypre_VectorData(x_local);
   }

   /* only support RAP with partial factorized W and Z */

   /* compute residual */
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* A-smoothing f_temp = [UA \ LA \ (f_temp[perm])] */
   /* permuted L solve */
   for (i = 0 ; i < n ; i ++)
   {
      utemp_data[i] = ftemp_data[perm[i]];
      k1 = L_diag_i[i] ; k2 = L_diag_i[i + 1];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = L_diag_j[j];
         utemp_data[i] -= L_diag_data[j] * utemp_data[col];
      }
   }

   if (!xtemp)
   {
      /* in this case, we don't have a Schur complement */
      /* U solve */
      for (i = n - 1 ; i >= 0 ; i --)
      {
         ftemp_data[perm[i]] = utemp_data[i];
         k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
         for (j = k1 ; j < k2 ; j ++)
         {
            col = U_diag_j[j];
            ftemp_data[perm[i]] -= U_diag_data[j] * ftemp_data[perm[col]];
         }
         ftemp_data[perm[i]] *= D[i];
      }

      hypre_ParVectorAxpy(beta, ftemp, u);

      return hypre_error_flag;
   }

   /* U solve */
   for (i = n - 1 ; i >= 0 ; i --)
   {
      xtemp_data[perm[i]] = utemp_data[i];
      k1 = U_diag_i[i] ; k2 = U_diag_i[i + 1];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = U_diag_j[j];
         xtemp_data[perm[i]] -= U_diag_data[j] * xtemp_data[perm[col]];
      }
      xtemp_data[perm[i]] *= D[i];
   }

   /* coarse-grid correction */
   /* now f_temp is the result of A-smoothing
    * rhs = R*(b - Ax)
    * */
   // utemp = (ftemp - A*xtemp)
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, xtemp, beta, ftemp, utemp);

   // R = [-L21 L\inv, I]
   if (m > 0)
   {
      /* first is L solve */
      for (i = 0 ; i < nLU ; i ++)
      {
         ytemp_data[i] = utemp_data[perm[i]];
         k1 = mL_diag_i[i] ; k2 = mL_diag_i[i + 1];
         for (j = k1 ; j < k2 ; j ++)
         {
            col = mL_diag_j[j];
            ytemp_data[i] -= mL_diag_data[j] * ytemp_data[col];
         }
      }

      /* apply -W * ytemp on this, and take care of the I part */
      for (i = nLU ; i < n ; i ++)
      {
         rhs_data[i - nLU] = utemp_data[perm[i]];
         k1 = mL_diag_i[i] ; k2 = u_end[i];
         for (j = k1 ; j < k2 ; j ++)
         {
            col = mL_diag_j[j];
            rhs_data[i - nLU] -= mL_diag_data[j] * ytemp_data[col];
         }
      }
   }

   /* now the rhs is ready */
   hypre_SeqVectorSetConstantValues(x_local, 0.0);
   HYPRE_GMRESSolve(schur_solver,
                    (HYPRE_Matrix) schur_precond,
                    (HYPRE_Vector) rhs,
                    (HYPRE_Vector) x);

   if (m > 0)
   {
      /*
      for(i = 0 ; i < m ; i ++)
      {
         x_data[i] = rhs_data[i];
         k1 = u_end[i+nLU] ; k2 = mL_diag_i[i+nLU+1];
         for(j = k1 ; j < k2 ; j ++)
         {
            col = mL_diag_j[j];
            x_data[i] -= mL_diag_data[j] * x_data[col-nLU];
         }
      }

      for(i = m-1 ; i >= 0 ; i --)
      {
         rhs_data[i] = x_data[i];
         k1 = mU_diag_i[i+nLU] ; k2 = mU_diag_i[i+1+nLU];
         for(j = k1 ; j < k2 ; j ++)
         {
            col = mU_diag_j[j];
            rhs_data[i] -= mU_diag_data[j] * rhs_data[col-nLU];
         }
         rhs_data[i] *= mD[i];
      }
      */

      /* after solve, update x = x + Pv
       * that is, xtemp = xtemp + P*x
       */
      /* first compute P*x
       * P = [ -U\inv U_12 ]
       *     [  I          ]
       */
      /* matvec */
      for (i = 0 ; i < nLU ; i ++)
      {
         ytemp_data[i] = 0.0;
         k1 = u_end[i] ; k2 = mU_diag_i[i + 1];
         for (j = k1 ; j < k2 ; j ++)
         {
            col = mU_diag_j[j];
            ytemp_data[i] -= mU_diag_data[j] * x_data[col - nLU];
         }
      }
      /* U solve */
      for (i = nLU - 1 ; i >= 0 ; i --)
      {
         ftemp_data[perm[i]] = ytemp_data[i];
         k1 = mU_diag_i[i] ; k2 = u_end[i];
         for (j = k1 ; j < k2 ; j ++)
         {
            col = mU_diag_j[j];
            ftemp_data[perm[i]] -= mU_diag_data[j] * ftemp_data[perm[col]];
         }
         ftemp_data[perm[i]] *= mD[i];
      }

      /* update with I */
      for (i = nLU ; i < n ; i ++)
      {
         ftemp_data[perm[i]] = x_data[i - nLU];
      }
      hypre_ParVectorAxpy(beta, ftemp, u);
   }

   hypre_ParVectorAxpy(beta, xtemp, u);

   return hypre_error_flag;
}

/******************************************************************************
 *
 * NSH functions.
 *
 *****************************************************************************/

/*--------------------------------------------------------------------
 * hypre_NSHSolve
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_NSHSolve( void               *nsh_vdata,
                hypre_ParCSRMatrix *A,
                hypre_ParVector    *f,
                hypre_ParVector    *u )
{
   MPI_Comm              comm           = hypre_ParCSRMatrixComm(A);
   hypre_ParNSHData     *nsh_data       = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParCSRMatrix   *matA           = hypre_ParNSHDataMatA(nsh_data);
   hypre_ParCSRMatrix   *matM           = hypre_ParNSHDataMatM(nsh_data);
   hypre_ParVector      *F_array        = hypre_ParNSHDataF(nsh_data);
   hypre_ParVector      *U_array        = hypre_ParNSHDataU(nsh_data);

   HYPRE_Real            tol            = hypre_ParNSHDataTol(nsh_data);
   HYPRE_Int             logging        = hypre_ParNSHDataLogging(nsh_data);
   HYPRE_Int             print_level    = hypre_ParNSHDataPrintLevel(nsh_data);
   HYPRE_Int             max_iter       = hypre_ParNSHDataMaxIter(nsh_data);
   HYPRE_Real           *norms          = hypre_ParNSHDataRelResNorms(nsh_data);
   hypre_ParVector      *Ftemp          = hypre_ParNSHDataFTemp(nsh_data);
   hypre_ParVector      *Utemp          = hypre_ParNSHDataUTemp(nsh_data);
   hypre_ParVector      *residual       = NULL;

   HYPRE_Real            alpha          = -1.0;
   HYPRE_Real            beta           = 1.0;
   HYPRE_Real            conv_factor    = 0.0;
   HYPRE_Real            resnorm        = 1.0;
   HYPRE_Real            init_resnorm   = 0.0;
   HYPRE_Real            rel_resnorm;
   HYPRE_Real            rhs_norm       = 0.0;
   HYPRE_Real            old_resnorm;
   HYPRE_Real            ieee_check     = 0.0;
   HYPRE_Real            operat_cmplxty = hypre_ParNSHDataOperatorComplexity(nsh_data);

   HYPRE_Int             iter, num_procs,  my_id;
   HYPRE_Int             Solve_err_flag;

   if (logging > 1)
   {
      residual = hypre_ParNSHDataResidual(nsh_data);
   }

   hypre_ParNSHDataNumIterations(nsh_data) = 0;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /*-----------------------------------------------------------------------
    *    Write the solver parameters
    *-----------------------------------------------------------------------*/
   if (my_id == 0 && print_level > 1)
   {
      hypre_NSHWriteSolverParams(nsh_data);
   }

   /*-----------------------------------------------------------------------
    *    Initialize the solver error flag
    *-----------------------------------------------------------------------*/

   Solve_err_flag = 0;
   /*-----------------------------------------------------------------------
    *     write some initial info
    *-----------------------------------------------------------------------*/

   if (my_id == 0 && print_level > 1 && tol > 0.)
   {
      hypre_printf("\n\n Newton-Schulz-Hotelling SOLVER SOLUTION INFO:\n");
   }


   /*-----------------------------------------------------------------------
    *    Compute initial residual and print
    *-----------------------------------------------------------------------*/
   if (print_level > 1 || logging > 1 || tol > 0.)
   {
      if (logging > 1)
      {
         hypre_ParVectorCopy(f, residual);
         if (tol > 0.0)
         {
            hypre_ParCSRMatrixMatvec(alpha, A, u, beta, residual);
         }
         resnorm = hypre_sqrt(hypre_ParVectorInnerProd(residual, residual));
      }
      else
      {
         hypre_ParVectorCopy(f, Ftemp);
         if (tol > 0.0)
         {
            hypre_ParCSRMatrixMatvec(alpha, A, u, beta, Ftemp);
         }
         resnorm = hypre_sqrt(hypre_ParVectorInnerProd(Ftemp, Ftemp));
      }

      /* Since it does not diminish performance, attempt to return an error flag
         and notify users when they supply bad input. */
      if (resnorm != 0.)
      {
         ieee_check = resnorm / resnorm; /* INF -> NaN conversion */
      }
      if (ieee_check != ieee_check)
      {
         /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
            for ieee_check self-equality works on all IEEE-compliant compilers/
            machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
            by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
            found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
         if (print_level > 0)
         {
            hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
            hypre_printf("ERROR -- hypre_NSHSolve: INFs and/or NaNs detected in input.\n");
            hypre_printf("User probably placed non-numerics in supplied A, x_0, or b.\n");
            hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
         }
         hypre_error(HYPRE_ERROR_GENERIC);
         return hypre_error_flag;
      }

      init_resnorm = resnorm;
      rhs_norm = hypre_sqrt(hypre_ParVectorInnerProd(f, f));
      if (rhs_norm > HYPRE_REAL_EPSILON)
      {
         rel_resnorm = init_resnorm / rhs_norm;
      }
      else
      {
         /* rhs is zero, return a zero solution */
         hypre_ParVectorSetConstantValues(U_array, 0.0);
         if (logging > 0)
         {
            rel_resnorm = 0.0;
            hypre_ParNSHDataFinalRelResidualNorm(nsh_data) = rel_resnorm;
         }
         return hypre_error_flag;
      }
   }
   else
   {
      rel_resnorm = 1.;
   }

   if (my_id == 0 && print_level > 1)
   {
      hypre_printf("                                            relative\n");
      hypre_printf("               residual        factor       residual\n");
      hypre_printf("               --------        ------       --------\n");
      hypre_printf("    Initial    %e                 %e\n", init_resnorm,
                   rel_resnorm);
   }

   matA = A;
   U_array = u;
   F_array = f;

   /************** Main Solver Loop - always do 1 iteration ************/
   iter = 0;

   while ((rel_resnorm >= tol || iter < 1) && iter < max_iter)
   {
      /* Do one solve on e = Mr */
      hypre_NSHSolveInverse(matA, f, u, matM, Utemp, Ftemp);

      /*---------------------------------------------------------------
       *    Compute residual and residual norm
       *----------------------------------------------------------------*/

      if (print_level > 1 || logging > 1 || tol > 0.)
      {
         old_resnorm = resnorm;

         if (logging > 1)
         {
            hypre_ParVectorCopy(F_array, residual);
            hypre_ParCSRMatrixMatvec(alpha, matA, U_array, beta, residual );
            resnorm = hypre_sqrt(hypre_ParVectorInnerProd( residual, residual ));
         }
         else
         {
            hypre_ParVectorCopy(F_array, Ftemp);
            hypre_ParCSRMatrixMatvec(alpha, matA, U_array, beta, Ftemp);
            resnorm = hypre_sqrt(hypre_ParVectorInnerProd(Ftemp, Ftemp));
         }

         if (old_resnorm) { conv_factor = resnorm / old_resnorm; }
         else { conv_factor = resnorm; }
         if (rhs_norm > HYPRE_REAL_EPSILON)
         {
            rel_resnorm = resnorm / rhs_norm;
         }
         else
         {
            rel_resnorm = resnorm;
         }

         norms[iter] = rel_resnorm;
      }

      ++iter;
      hypre_ParNSHDataNumIterations(nsh_data) = iter;
      hypre_ParNSHDataFinalRelResidualNorm(nsh_data) = rel_resnorm;

      if (my_id == 0 && print_level > 1)
      {
         hypre_printf("    NSHSolve %2d   %e    %f     %e \n", iter,
                      resnorm, conv_factor, rel_resnorm);
      }
   }

   /* check convergence within max_iter */
   if (iter == max_iter && tol > 0.)
   {
      Solve_err_flag = 1;
      hypre_error(HYPRE_ERROR_CONV);
   }

   /*-----------------------------------------------------------------------
    *    Print closing statistics
    *    Add operator and grid complexity stats
    *-----------------------------------------------------------------------*/

   if (iter > 0 && init_resnorm)
   {
      conv_factor = hypre_pow((resnorm / init_resnorm), (1.0 / (HYPRE_Real) iter));
   }
   else
   {
      conv_factor = 1.;
   }

   if (print_level > 1)
   {
      /*** compute operator and grid complexity (fill factor) here ?? ***/
      if (my_id == 0)
      {
         if (Solve_err_flag == 1)
         {
            hypre_printf("\n\n==============================================");
            hypre_printf("\n NOTE: Convergence tolerance was not achieved\n");
            hypre_printf("      within the allowed %d iterations\n", max_iter);
            hypre_printf("==============================================");
         }
         hypre_printf("\n\n Average Convergence Factor = %f \n", conv_factor);
         hypre_printf("                operator = %f\n", operat_cmplxty);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------
 * hypre_NSHSolveInverse
 *
 * Simply a matvec on residual with approximate inverse
 *
 * A: original matrix
 * f: rhs
 * u: solution
 * M: approximate inverse
 * ftemp, utemp: working vectors
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_NSHSolveInverse(hypre_ParCSRMatrix *A,
                      hypre_ParVector    *f,
                      hypre_ParVector    *u,
                      hypre_ParCSRMatrix *M,
                      hypre_ParVector    *ftemp,
                      hypre_ParVector    *utemp)
{
   HYPRE_Real  alpha = -1.0;
   HYPRE_Real  beta  = 1.0;
   HYPRE_Real  zero  = 0.0;

   /* r = f - Au */
   hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, u, beta, f, ftemp);

   /* e = Mr */
   hypre_ParCSRMatrixMatvec(beta, M, ftemp, zero, utemp);

   /* u = u + e */
   hypre_ParVectorAxpy(beta, utemp, u);

   return hypre_error_flag;
}
