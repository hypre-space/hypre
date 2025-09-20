/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_LOBPCG interface
 *
 *****************************************************************************/

#include "_hypre_utilities.h"
#include "HYPRE_config.h"

#include "HYPRE_lobpcg.h"
#include "_hypre_lobpcg.h"
#include "_hypre_lobpcg_interpreter.h"

HYPRE_Int
HYPRE_LOBPCGCreate( mv_InterfaceInterpreter* ii, HYPRE_MatvecFunctions* mv,
                    HYPRE_Solver* solver )
{
   hypre_LOBPCGData *pcg_data;

   pcg_data = hypre_CTAlloc(hypre_LOBPCGData, 1, HYPRE_MEMORY_HOST);

   (pcg_data->precondFunctions).Precond = NULL;
   (pcg_data->precondFunctions).PrecondSetup = NULL;

   /* set defaults */

   (pcg_data->interpreter)               = ii;
   pcg_data->matvecFunctions             = mv;

   (pcg_data->matvecData)           = NULL;
   (pcg_data->B)                 = NULL;
   (pcg_data->matvecDataB)          = NULL;
   (pcg_data->T)                 = NULL;
   (pcg_data->matvecDataT)          = NULL;
   (pcg_data->precondData)          = NULL;

   lobpcg_initialize( &(pcg_data->lobpcgData) );

   *solver = (HYPRE_Solver)pcg_data;

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_LOBPCGDestroy( HYPRE_Solver solver )
{
   return ( hypre_LOBPCGDestroy( (void *) solver ) );
}

HYPRE_Int
HYPRE_LOBPCGSetup( HYPRE_Solver solver,
                   HYPRE_Matrix A,
                   HYPRE_Vector b,
                   HYPRE_Vector x      )
{
   return ( hypre_LOBPCGSetup( solver, A, b, x ) );
}

HYPRE_Int
HYPRE_LOBPCGSetupB( HYPRE_Solver solver,
                    HYPRE_Matrix B,
                    HYPRE_Vector x      )
{
   return ( hypre_LOBPCGSetupB( solver, B, x ) );
}

HYPRE_Int
HYPRE_LOBPCGSetupT( HYPRE_Solver solver,
                    HYPRE_Matrix T,
                    HYPRE_Vector x      )
{
   return ( hypre_LOBPCGSetupT( solver, T, x ) );
}

HYPRE_Int
HYPRE_LOBPCGSolve( HYPRE_Solver solver, mv_MultiVectorPtr con,
                   mv_MultiVectorPtr vec, HYPRE_Real* val )
{
   return ( hypre_LOBPCGSolve( (void *) solver, con, vec, val ) );
}

HYPRE_Int
HYPRE_LOBPCGSetTol( HYPRE_Solver solver, HYPRE_Real tol )
{
   return ( hypre_LOBPCGSetTol( (void *) solver, tol ) );
}

HYPRE_Int
HYPRE_LOBPCGSetRTol( HYPRE_Solver solver, HYPRE_Real tol )
{
   return ( hypre_LOBPCGSetRTol( (void *) solver, tol ) );
}

HYPRE_Int
HYPRE_LOBPCGSetMaxIter( HYPRE_Solver solver, HYPRE_Int max_iter )
{
   return ( hypre_LOBPCGSetMaxIter( (void *) solver, max_iter ) );
}

HYPRE_Int
HYPRE_LOBPCGSetPrecondUsageMode( HYPRE_Solver solver, HYPRE_Int mode )
{
   return ( hypre_LOBPCGSetPrecondUsageMode( (void *) solver, mode ) );
}

HYPRE_Int
HYPRE_LOBPCGSetPrecond( HYPRE_Solver         solver,
                        HYPRE_PtrToSolverFcn precond,
                        HYPRE_PtrToSolverFcn precond_setup,
                        HYPRE_Solver         precond_solver )
{
   return ( hypre_LOBPCGSetPrecond( (void *) solver,
                                    (HYPRE_Int (*)(void*, void*, void*, void*))precond,
                                    (HYPRE_Int (*)(void*, void*, void*, void*))precond_setup,
                                    (void *) precond_solver ) );
}

HYPRE_Int
HYPRE_LOBPCGGetPrecond( HYPRE_Solver  solver,
                        HYPRE_Solver *precond_data_ptr )
{
   return ( hypre_LOBPCGGetPrecond( (void *)     solver,
                                    (HYPRE_Solver *) precond_data_ptr ) );
}

HYPRE_Int
HYPRE_LOBPCGSetPrintLevel( HYPRE_Solver solver, HYPRE_Int level )
{
   return ( hypre_LOBPCGSetPrintLevel( (void*)solver, level ) );
}

utilities_FortranMatrix*
HYPRE_LOBPCGResidualNorms( HYPRE_Solver solver )
{
   return ( hypre_LOBPCGResidualNorms( (void*)solver ) );
}

utilities_FortranMatrix*
HYPRE_LOBPCGResidualNormsHistory( HYPRE_Solver solver )
{
   return ( hypre_LOBPCGResidualNormsHistory( (void*)solver ) );
}

utilities_FortranMatrix*
HYPRE_LOBPCGEigenvaluesHistory( HYPRE_Solver solver )
{
   return ( hypre_LOBPCGEigenvaluesHistory( (void*)solver ) );
}

HYPRE_Int
HYPRE_LOBPCGIterations( HYPRE_Solver solver )
{
   return ( hypre_LOBPCGIterations( (void*)solver ) );
}
