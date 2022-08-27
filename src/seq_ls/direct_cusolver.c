/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_seq_ls.h"
#include "_hypre_utilities.hpp"

#if defined (HYPRE_USING_CUSOLVER)

/*--------------------------------------------------------------------------
 * hypre_DirectSolverCreateCuSolver
 *--------------------------------------------------------------------------*/

void*
hypre_DirectSolverCreateCuSolver()
{
   cusolverDnHandle_t handle = NULL;

   HYPRE_CUSOLVER_CALL(cusolverDnCreate(&handle));

   return (void*) handle;
}

/*--------------------------------------------------------------------------
 * hypre_DirectSolverDestroyCuSolver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverDestroyCuSolver(void* direct_vdata)
{
   hypre_DirectSolverData  *data   = (hypre_DirectSolverData*) direct_vdata;
   cusolverDnHandle_t       handle = hypre_DirectSolverDataCuSolverHandle(data);

   if (handle)
   {
      HYPRE_CUSOLVER_CALL(cusolverDnDestroy(handle));
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DirectSolverSetupCuSolver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverSetupCuSolver( void               *direct_vdata,
                                 hypre_DenseMatrix  *A,
                                 hypre_Vector       *f,
                                 hypre_Vector       *u )
{
   /* TODO: Call specific setup routines for each solver option */

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DirectSolverSolveCuSolver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverSolveCuSolver( void               *direct_vdata,
                                 hypre_DenseMatrix  *A,
                                 hypre_Vector       *f,
                                 hypre_Vector       *u )
{
   /* TODO: Call specific solve routines for each solver option */

   return hypre_error_flag;
}

#endif
