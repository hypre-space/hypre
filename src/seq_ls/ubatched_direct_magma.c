/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_seq_ls.h"
#include "_hypre_utilities.hpp"

#if defined (HYPRE_USING_MAGMA)

/*--------------------------------------------------------------------------
 * hypre_DirectSolverCreateMagma
 *--------------------------------------------------------------------------*/

void*
hypre_DirectSolverCreateMagma()
{
   void *handle = NULL;

   return (void*) handle;
}

/*--------------------------------------------------------------------------
 * hypre_DirectSolverDestroyMagma
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverDestroyMagma(void* direct_vdata)
{
   /* TODO: Call specific destroy routines for each solver option */

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DirectSolverSetupMagma
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverSetupMagma( void               *direct_vdata,
                              hypre_DenseMatrix  *A,
                              hypre_Vector       *f,
                              hypre_Vector       *u )
{
   /* TODO: Call specific setup routines for each solver option */

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DirectSolverSolveMagma
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverSolveMagma( void               *direct_vdata,
                              hypre_DenseMatrix  *A,
                              hypre_Vector       *f,
                              hypre_Vector       *u )
{
   /* TODO: Call specific solve routines for each solver option */

   return hypre_error_flag;
}

#endif
