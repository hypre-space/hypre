/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_seq_ls.h"
#include "_hypre_utilities.hpp"

/*--------------------------------------------------------------------------
 * hypre_DirectSolverCreate
 *--------------------------------------------------------------------------*/

void*
hypre_DirectSolverCreate(HYPRE_Int option)
{
   hypre_DirectSolverData  *data;

   data = hypre_TAlloc(hypre_DirectSolverData, 1, HYPRE_MEMORY_HOST);

   /* Set default values */
   hypre_DirectSolverDataOption(data) = option;
   hypre_DirectSolverDataInfo(data)   = 0;

   /* TODO: Call specific create routines for each solver option */
   switch (option)
   {
      case 1:
         hypre_printf("Call Create for option 1\n");
         break;

      case 2:
         hypre_printf("Call Create for option 2\n");
         break;

      default:
         hypre_printf("Unknown solver option!\n");
   }

   return (void*) data;
}

/*--------------------------------------------------------------------------
 * hypre_DirectSolverDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverDestroy(void* vdata)
{
   hypre_DirectSolverData  *data = (hypre_DirectSolverData*) vdata;

   if (data)
   {
      /* TODO: Call specific destroy routines for each solver option */
      switch (hypre_DirectSolverDataOption(data))
      {
         case 1:
            hypre_printf("Call Destroy for option 1\n");
            break;

         case 2:
            hypre_printf("Call Destroy for option 2\n");
            break;

         default:
            hypre_printf("Unknown solver option!\n");
      }

      hypre_TFree(data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DirectSolverSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverSetup( void               *direct_vdata,
                         hypre_DenseMatrix  *A,
                         hypre_Vector       *f,
                         hypre_Vector       *u )
{
   /* TODO: Call specific setup routines for each solver option */

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DirectSolverSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DirectSolverSolve( void               *direct_vdata,
                         hypre_DenseMatrix  *A,
                         hypre_Vector       *f,
                         hypre_Vector       *u )
{
   /* TODO: Call specific solve routines for each solver option */

   return hypre_error_flag;
}
