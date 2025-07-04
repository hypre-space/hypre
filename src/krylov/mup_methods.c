
/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "krylov.h"


/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "krylov.h"



/*--------------------------------------------------------------------------
* HYPRE_PCGSetup
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetup( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetup_flt ( solver, A, b, x);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetup_dbl ( solver, A, b, x);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetup_ldbl ( solver, A, b, x);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSolve
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSolve( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSolve_flt ( solver, A, b, x);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSolve_dbl ( solver, A, b, x);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSolve_ldbl ( solver, A, b, x);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetTol
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetTol( HYPRE_Solver solver, HYPRE_Real tol)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetTol_flt ( solver, tol);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetTol_dbl ( solver, tol);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetTol_ldbl ( solver, tol);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetTol
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetTol( HYPRE_Solver solver, HYPRE_Real *tol)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetTol_flt ( solver, tol);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetTol_dbl ( solver, tol);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetTol_ldbl ( solver, tol);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetAbsoluteTol
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetAbsoluteTol( HYPRE_Solver solver, HYPRE_Real a_tol)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetAbsoluteTol_flt ( solver, a_tol);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetAbsoluteTol_dbl ( solver, a_tol);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetAbsoluteTol_ldbl ( solver, a_tol);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetAbsoluteTol
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetAbsoluteTol( HYPRE_Solver solver, HYPRE_Real *a_tol)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetAbsoluteTol_flt ( solver, a_tol);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetAbsoluteTol_dbl ( solver, a_tol);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetAbsoluteTol_ldbl ( solver, a_tol);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetAbsoluteTolFactor
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetAbsoluteTolFactor( HYPRE_Solver solver, HYPRE_Real abstolf)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetAbsoluteTolFactor_flt ( solver, abstolf);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetAbsoluteTolFactor_dbl ( solver, abstolf);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetAbsoluteTolFactor_ldbl ( solver, abstolf);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetAbsoluteTolFactor
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetAbsoluteTolFactor( HYPRE_Solver solver, HYPRE_Real *abstolf)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetAbsoluteTolFactor_flt ( solver, abstolf);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetAbsoluteTolFactor_dbl ( solver, abstolf);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetAbsoluteTolFactor_ldbl ( solver, abstolf);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetResidualTol
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetResidualTol( HYPRE_Solver solver, HYPRE_Real rtol)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetResidualTol_flt ( solver, rtol);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetResidualTol_dbl ( solver, rtol);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetResidualTol_ldbl ( solver, rtol);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetResidualTol
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetResidualTol( HYPRE_Solver solver, HYPRE_Real *rtol)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetResidualTol_flt ( solver, rtol);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetResidualTol_dbl ( solver, rtol);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetResidualTol_ldbl ( solver, rtol);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetConvergenceFactorTol
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetConvergenceFactorTol( HYPRE_Solver solver, HYPRE_Real cf_tol)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetConvergenceFactorTol_flt ( solver, cf_tol);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetConvergenceFactorTol_dbl ( solver, cf_tol);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetConvergenceFactorTol_ldbl ( solver, cf_tol);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetConvergenceFactorTol
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetConvergenceFactorTol( HYPRE_Solver solver, HYPRE_Real *cf_tol)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetConvergenceFactorTol_flt ( solver, cf_tol);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetConvergenceFactorTol_dbl ( solver, cf_tol);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetConvergenceFactorTol_ldbl ( solver, cf_tol);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetMaxIter
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetMaxIter( HYPRE_Solver solver, HYPRE_Int max_iter)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetMaxIter_flt ( solver, max_iter);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetMaxIter_dbl ( solver, max_iter);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetMaxIter_ldbl ( solver, max_iter);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetMaxIter
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetMaxIter( HYPRE_Solver solver, HYPRE_Int *max_iter)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetMaxIter_flt ( solver, max_iter);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetMaxIter_dbl ( solver, max_iter);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetMaxIter_ldbl ( solver, max_iter);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetStopCrit
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetStopCrit( HYPRE_Solver solver, HYPRE_Int stop_crit)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetStopCrit_flt ( solver, stop_crit);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetStopCrit_dbl ( solver, stop_crit);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetStopCrit_ldbl ( solver, stop_crit);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetStopCrit
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetStopCrit( HYPRE_Solver solver, HYPRE_Int *stop_crit)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetStopCrit_flt ( solver, stop_crit);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetStopCrit_dbl ( solver, stop_crit);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetStopCrit_ldbl ( solver, stop_crit);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetTwoNorm
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetTwoNorm( HYPRE_Solver solver, HYPRE_Int two_norm)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetTwoNorm_flt ( solver, two_norm);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetTwoNorm_dbl ( solver, two_norm);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetTwoNorm_ldbl ( solver, two_norm);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetTwoNorm
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetTwoNorm( HYPRE_Solver solver, HYPRE_Int *two_norm)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetTwoNorm_flt ( solver, two_norm);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetTwoNorm_dbl ( solver, two_norm);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetTwoNorm_ldbl ( solver, two_norm);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetRelChange
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetRelChange( HYPRE_Solver solver, HYPRE_Int rel_change)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetRelChange_flt ( solver, rel_change);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetRelChange_dbl ( solver, rel_change);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetRelChange_ldbl ( solver, rel_change);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetRelChange
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetRelChange( HYPRE_Solver solver, HYPRE_Int *rel_change)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetRelChange_flt ( solver, rel_change);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetRelChange_dbl ( solver, rel_change);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetRelChange_ldbl ( solver, rel_change);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetRecomputeResidual
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetRecomputeResidual( HYPRE_Solver solver, HYPRE_Int recompute_residual)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetRecomputeResidual_flt ( solver, recompute_residual);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetRecomputeResidual_dbl ( solver, recompute_residual);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetRecomputeResidual_ldbl ( solver, recompute_residual);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetRecomputeResidual
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetRecomputeResidual( HYPRE_Solver solver, HYPRE_Int *recompute_residual)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetRecomputeResidual_flt ( solver, recompute_residual);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetRecomputeResidual_dbl ( solver, recompute_residual);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetRecomputeResidual_ldbl ( solver, recompute_residual);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetRecomputeResidualP
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetRecomputeResidualP( HYPRE_Solver solver, HYPRE_Int recompute_residual_p)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetRecomputeResidualP_flt ( solver, recompute_residual_p);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetRecomputeResidualP_dbl ( solver, recompute_residual_p);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetRecomputeResidualP_ldbl ( solver, recompute_residual_p);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetRecomputeResidualP
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetRecomputeResidualP( HYPRE_Solver solver, HYPRE_Int *recompute_residual_p)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetRecomputeResidualP_flt ( solver, recompute_residual_p);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetRecomputeResidualP_dbl ( solver, recompute_residual_p);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetRecomputeResidualP_ldbl ( solver, recompute_residual_p);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetSkipBreak
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetSkipBreak( HYPRE_Solver solver, HYPRE_Int skip_break)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetSkipBreak_flt ( solver, skip_break);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetSkipBreak_dbl ( solver, skip_break);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetSkipBreak_ldbl ( solver, skip_break);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetSkipBreak
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetSkipBreak( HYPRE_Solver solver, HYPRE_Int *skip_break)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetSkipBreak_flt ( solver, skip_break);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetSkipBreak_dbl ( solver, skip_break);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetSkipBreak_ldbl ( solver, skip_break);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetFlex
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetFlex( HYPRE_Solver solver, HYPRE_Int flex)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetFlex_flt ( solver, flex);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetFlex_dbl ( solver, flex);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetFlex_ldbl ( solver, flex);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetFlex
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetFlex( HYPRE_Solver solver, HYPRE_Int *flex)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetFlex_flt ( solver, flex);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetFlex_dbl ( solver, flex);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetFlex_ldbl ( solver, flex);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetPrecond
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetPrecond(void *solver,
 		HYPRE_Int (*precond)(void*,void*,void*,void*),
 		HYPRE_Int (*precond_setup)(void*,void*,void*,void*),
 		void *precond_data)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetPrecond_flt (solver, precond, precond_setup, precond_data);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetPrecond_dbl (solver, precond, precond_setup, precond_data);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetPrecond_ldbl (solver, precond, precond_setup, precond_data);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetPrecondMatrix
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetPrecondMatrix(void *solver,
 		HYPRE_Int (*precond)(void*,void*,void*,void*),
 		HYPRE_Int (*precond_setup)(void*,void*,void*,void*),
 		void *precond_data)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetPrecondMatrix_flt (solver, precond, precond_setup, precond_data);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetPrecondMatrix_dbl (solver, precond, precond_setup, precond_data);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetPrecondMatrix_ldbl (solver, precond, precond_setup, precond_data);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetPreconditioner
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetPreconditioner(void *solver,
 		HYPRE_Int (*precond)(void*,void*,void*,void*),
 		HYPRE_Int (*precond_setup)(void*,void*,void*,void*),
 		void *precond_data)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetPreconditioner_flt (solver, precond, precond_setup, precond_data);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetPreconditioner_dbl (solver, precond, precond_setup, precond_data);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetPreconditioner_ldbl (solver, precond, precond_setup, precond_data);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetPrecond
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetPrecond( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetPrecond_flt ( solver, precond_data_ptr);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetPrecond_dbl ( solver, precond_data_ptr);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetPrecond_ldbl ( solver, precond_data_ptr);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetPrecondMatrix
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetPrecondMatrix( HYPRE_Solver solver, HYPRE_Matrix *precond_matrix_ptr)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetPrecondMatrix_flt ( solver, precond_matrix_ptr);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetPrecondMatrix_dbl ( solver, precond_matrix_ptr);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetPrecondMatrix_ldbl ( solver, precond_matrix_ptr);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetLogging
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetLogging( HYPRE_Solver solver, HYPRE_Int level)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetLogging_flt ( solver, level);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetLogging_dbl ( solver, level);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetLogging_ldbl ( solver, level);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetLogging
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetLogging( HYPRE_Solver solver, HYPRE_Int *level)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetLogging_flt ( solver, level);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetLogging_dbl ( solver, level);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetLogging_ldbl ( solver, level);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetPrintLevel
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetPrintLevel( HYPRE_Solver solver, HYPRE_Int level)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetPrintLevel_flt ( solver, level);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetPrintLevel_dbl ( solver, level);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetPrintLevel_ldbl ( solver, level);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetPrintLevel
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetPrintLevel( HYPRE_Solver solver, HYPRE_Int *level)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetPrintLevel_flt ( solver, level);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetPrintLevel_dbl ( solver, level);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetPrintLevel_ldbl ( solver, level);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetNumIterations
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetNumIterations( HYPRE_Solver solver, HYPRE_Int *num_iterations)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetNumIterations_flt ( solver, num_iterations);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetNumIterations_dbl ( solver, num_iterations);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetNumIterations_ldbl ( solver, num_iterations);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetConverged
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetConverged( HYPRE_Solver solver, HYPRE_Int *converged)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetConverged_flt ( solver, converged);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetConverged_dbl ( solver, converged);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetConverged_ldbl ( solver, converged);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetFinalRelativeResidualNorm
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetFinalRelativeResidualNorm( HYPRE_Solver solver, HYPRE_Real *norm)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetFinalRelativeResidualNorm_flt ( solver, norm);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetFinalRelativeResidualNorm_dbl ( solver, norm);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetFinalRelativeResidualNorm_ldbl ( solver, norm);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetResidual
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetResidual( HYPRE_Solver solver, void *residual)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetResidual_flt ( solver, residual);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetResidual_dbl ( solver, residual);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetResidual_ldbl ( solver, residual);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "krylov.h"



/*--------------------------------------------------------------------------
* HYPRE_PCGSetup
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetup( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetup_flt ( solver, A, b, x);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetup_dbl ( solver, A, b, x);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetup_ldbl ( solver, A, b, x);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSolve
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSolve( HYPRE_Solver solver, HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSolve_flt ( solver, A, b, x);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSolve_dbl ( solver, A, b, x);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSolve_ldbl ( solver, A, b, x);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetTol
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetTol( HYPRE_Solver solver, HYPRE_Real tol)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetTol_flt ( solver, tol);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetTol_dbl ( solver, tol);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetTol_ldbl ( solver, tol);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetTol
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetTol( HYPRE_Solver solver, HYPRE_Real *tol)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetTol_flt ( solver, tol);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetTol_dbl ( solver, tol);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetTol_ldbl ( solver, tol);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetAbsoluteTol
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetAbsoluteTol( HYPRE_Solver solver, HYPRE_Real a_tol)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetAbsoluteTol_flt ( solver, a_tol);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetAbsoluteTol_dbl ( solver, a_tol);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetAbsoluteTol_ldbl ( solver, a_tol);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetAbsoluteTol
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetAbsoluteTol( HYPRE_Solver solver, HYPRE_Real *a_tol)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetAbsoluteTol_flt ( solver, a_tol);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetAbsoluteTol_dbl ( solver, a_tol);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetAbsoluteTol_ldbl ( solver, a_tol);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetAbsoluteTolFactor
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetAbsoluteTolFactor( HYPRE_Solver solver, HYPRE_Real abstolf)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetAbsoluteTolFactor_flt ( solver, abstolf);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetAbsoluteTolFactor_dbl ( solver, abstolf);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetAbsoluteTolFactor_ldbl ( solver, abstolf);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetAbsoluteTolFactor
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetAbsoluteTolFactor( HYPRE_Solver solver, HYPRE_Real *abstolf)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetAbsoluteTolFactor_flt ( solver, abstolf);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetAbsoluteTolFactor_dbl ( solver, abstolf);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetAbsoluteTolFactor_ldbl ( solver, abstolf);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetResidualTol
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetResidualTol( HYPRE_Solver solver, HYPRE_Real rtol)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetResidualTol_flt ( solver, rtol);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetResidualTol_dbl ( solver, rtol);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetResidualTol_ldbl ( solver, rtol);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetResidualTol
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetResidualTol( HYPRE_Solver solver, HYPRE_Real *rtol)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetResidualTol_flt ( solver, rtol);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetResidualTol_dbl ( solver, rtol);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetResidualTol_ldbl ( solver, rtol);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetConvergenceFactorTol
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetConvergenceFactorTol( HYPRE_Solver solver, HYPRE_Real cf_tol)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetConvergenceFactorTol_flt ( solver, cf_tol);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetConvergenceFactorTol_dbl ( solver, cf_tol);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetConvergenceFactorTol_ldbl ( solver, cf_tol);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetConvergenceFactorTol
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetConvergenceFactorTol( HYPRE_Solver solver, HYPRE_Real *cf_tol)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetConvergenceFactorTol_flt ( solver, cf_tol);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetConvergenceFactorTol_dbl ( solver, cf_tol);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetConvergenceFactorTol_ldbl ( solver, cf_tol);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetMaxIter
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetMaxIter( HYPRE_Solver solver, HYPRE_Int max_iter)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetMaxIter_flt ( solver, max_iter);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetMaxIter_dbl ( solver, max_iter);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetMaxIter_ldbl ( solver, max_iter);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetMaxIter
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetMaxIter( HYPRE_Solver solver, HYPRE_Int *max_iter)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetMaxIter_flt ( solver, max_iter);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetMaxIter_dbl ( solver, max_iter);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetMaxIter_ldbl ( solver, max_iter);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetStopCrit
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetStopCrit( HYPRE_Solver solver, HYPRE_Int stop_crit)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetStopCrit_flt ( solver, stop_crit);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetStopCrit_dbl ( solver, stop_crit);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetStopCrit_ldbl ( solver, stop_crit);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetStopCrit
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetStopCrit( HYPRE_Solver solver, HYPRE_Int *stop_crit)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetStopCrit_flt ( solver, stop_crit);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetStopCrit_dbl ( solver, stop_crit);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetStopCrit_ldbl ( solver, stop_crit);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetTwoNorm
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetTwoNorm( HYPRE_Solver solver, HYPRE_Int two_norm)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetTwoNorm_flt ( solver, two_norm);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetTwoNorm_dbl ( solver, two_norm);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetTwoNorm_ldbl ( solver, two_norm);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetTwoNorm
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetTwoNorm( HYPRE_Solver solver, HYPRE_Int *two_norm)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetTwoNorm_flt ( solver, two_norm);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetTwoNorm_dbl ( solver, two_norm);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetTwoNorm_ldbl ( solver, two_norm);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetRelChange
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetRelChange( HYPRE_Solver solver, HYPRE_Int rel_change)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetRelChange_flt ( solver, rel_change);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetRelChange_dbl ( solver, rel_change);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetRelChange_ldbl ( solver, rel_change);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetRelChange
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetRelChange( HYPRE_Solver solver, HYPRE_Int *rel_change)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetRelChange_flt ( solver, rel_change);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetRelChange_dbl ( solver, rel_change);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetRelChange_ldbl ( solver, rel_change);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetRecomputeResidual
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetRecomputeResidual( HYPRE_Solver solver, HYPRE_Int recompute_residual)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetRecomputeResidual_flt ( solver, recompute_residual);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetRecomputeResidual_dbl ( solver, recompute_residual);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetRecomputeResidual_ldbl ( solver, recompute_residual);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetRecomputeResidual
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetRecomputeResidual( HYPRE_Solver solver, HYPRE_Int *recompute_residual)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetRecomputeResidual_flt ( solver, recompute_residual);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetRecomputeResidual_dbl ( solver, recompute_residual);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetRecomputeResidual_ldbl ( solver, recompute_residual);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetRecomputeResidualP
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetRecomputeResidualP( HYPRE_Solver solver, HYPRE_Int recompute_residual_p)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetRecomputeResidualP_flt ( solver, recompute_residual_p);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetRecomputeResidualP_dbl ( solver, recompute_residual_p);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetRecomputeResidualP_ldbl ( solver, recompute_residual_p);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetRecomputeResidualP
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetRecomputeResidualP( HYPRE_Solver solver, HYPRE_Int *recompute_residual_p)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetRecomputeResidualP_flt ( solver, recompute_residual_p);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetRecomputeResidualP_dbl ( solver, recompute_residual_p);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetRecomputeResidualP_ldbl ( solver, recompute_residual_p);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetSkipBreak
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetSkipBreak( HYPRE_Solver solver, HYPRE_Int skip_break)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetSkipBreak_flt ( solver, skip_break);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetSkipBreak_dbl ( solver, skip_break);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetSkipBreak_ldbl ( solver, skip_break);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetSkipBreak
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetSkipBreak( HYPRE_Solver solver, HYPRE_Int *skip_break)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetSkipBreak_flt ( solver, skip_break);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetSkipBreak_dbl ( solver, skip_break);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetSkipBreak_ldbl ( solver, skip_break);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetFlex
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetFlex( HYPRE_Solver solver, HYPRE_Int flex)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetFlex_flt ( solver, flex);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetFlex_dbl ( solver, flex);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetFlex_ldbl ( solver, flex);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetFlex
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetFlex( HYPRE_Solver solver, HYPRE_Int *flex)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetFlex_flt ( solver, flex);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetFlex_dbl ( solver, flex);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetFlex_ldbl ( solver, flex);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetPrecond
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetPrecond(void *solver,
 		HYPRE_Int (*precond)(void*,void*,void*,void*),
 		HYPRE_Int (*precond_setup)(void*,void*,void*,void*),
 		void *precond_data)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetPrecond_flt (solver, precond, precond_setup, precond_data);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetPrecond_dbl (solver, precond, precond_setup, precond_data);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetPrecond_ldbl (solver, precond, precond_setup, precond_data);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetPrecondMatrix
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetPrecondMatrix(void *solver,
 		HYPRE_Int (*precond)(void*,void*,void*,void*),
 		HYPRE_Int (*precond_setup)(void*,void*,void*,void*),
 		void *precond_data)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetPrecondMatrix_flt (solver, precond, precond_setup, precond_data);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetPrecondMatrix_dbl (solver, precond, precond_setup, precond_data);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetPrecondMatrix_ldbl (solver, precond, precond_setup, precond_data);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetPreconditioner
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetPreconditioner(void *solver,
 		HYPRE_Int (*precond)(void*,void*,void*,void*),
 		HYPRE_Int (*precond_setup)(void*,void*,void*,void*),
 		void *precond_data)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetPreconditioner_flt (solver, precond, precond_setup, precond_data);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetPreconditioner_dbl (solver, precond, precond_setup, precond_data);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetPreconditioner_ldbl (solver, precond, precond_setup, precond_data);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetPrecond
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetPrecond( HYPRE_Solver solver, HYPRE_Solver *precond_data_ptr)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetPrecond_flt ( solver, precond_data_ptr);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetPrecond_dbl ( solver, precond_data_ptr);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetPrecond_ldbl ( solver, precond_data_ptr);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetPrecondMatrix
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetPrecondMatrix( HYPRE_Solver solver, HYPRE_Matrix *precond_matrix_ptr)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetPrecondMatrix_flt ( solver, precond_matrix_ptr);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetPrecondMatrix_dbl ( solver, precond_matrix_ptr);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetPrecondMatrix_ldbl ( solver, precond_matrix_ptr);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetLogging
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetLogging( HYPRE_Solver solver, HYPRE_Int level)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetLogging_flt ( solver, level);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetLogging_dbl ( solver, level);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetLogging_ldbl ( solver, level);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetLogging
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetLogging( HYPRE_Solver solver, HYPRE_Int *level)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetLogging_flt ( solver, level);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetLogging_dbl ( solver, level);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetLogging_ldbl ( solver, level);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGSetPrintLevel
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGSetPrintLevel( HYPRE_Solver solver, HYPRE_Int level)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGSetPrintLevel_flt ( solver, level);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGSetPrintLevel_dbl ( solver, level);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGSetPrintLevel_ldbl ( solver, level);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetPrintLevel
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetPrintLevel( HYPRE_Solver solver, HYPRE_Int *level)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetPrintLevel_flt ( solver, level);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetPrintLevel_dbl ( solver, level);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetPrintLevel_ldbl ( solver, level);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetNumIterations
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetNumIterations( HYPRE_Solver solver, HYPRE_Int *num_iterations)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetNumIterations_flt ( solver, num_iterations);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetNumIterations_dbl ( solver, num_iterations);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetNumIterations_ldbl ( solver, num_iterations);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetConverged
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetConverged( HYPRE_Solver solver, HYPRE_Int *converged)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetConverged_flt ( solver, converged);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetConverged_dbl ( solver, converged);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetConverged_ldbl ( solver, converged);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetFinalRelativeResidualNorm
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetFinalRelativeResidualNorm( HYPRE_Solver solver, HYPRE_Real *norm)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetFinalRelativeResidualNorm_flt ( solver, norm);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetFinalRelativeResidualNorm_dbl ( solver, norm);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetFinalRelativeResidualNorm_ldbl ( solver, norm);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
* HYPRE_PCGGetResidual
*--------------------------------------------------------------------------*/
HYPRE_Int 
HYPRE_PCGGetResidual( HYPRE_Solver solver, void *residual)
{
   hypre_PCGData *pcg_data = (hypre_PCGData *)solver;
   switch (pcg_data-> solver_precision)
   {
      case HYPRE_REAL_SINGLE:
         HYPRE_PCGGetResidual_flt ( solver, residual);
         break;
      case HYPRE_REAL_DOUBLE:
         HYPRE_PCGGetResidual_dbl ( solver, residual);
         break;
      case HYPRE_REAL_LONGDOUBLE:
         HYPRE_PCGGetResidual_ldbl ( solver, residual);
         break;
      default:
         hypre_printf("Unknown solver precision" );
   }
   return hypre_error_flag;
}
