/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_fsai.h"

/******************************************************************************
 * HYPRE_FSAICreate
 ******************************************************************************/

void *
hypre_FSAICreate( void )
{
   hypre_ParFSAIData    *fsai_data;

   /* setup params */
   HYPRE_Int            algo_type;
   HYPRE_Int            local_solve_type;
   HYPRE_Int            max_steps;
   HYPRE_Int            max_step_size;
   HYPRE_Int            max_nnz_row;
   HYPRE_Int            num_levels;
   HYPRE_Real           kap_tolerance;

   /* solver params */
   HYPRE_Int            eig_max_iters;
   HYPRE_Int            max_iterations;
   HYPRE_Int            num_iterations;
   HYPRE_Real           tolerance;
   HYPRE_Real           omega;

   /* log info */
   HYPRE_Int            logging;

   /* output params */
   HYPRE_Int            print_level;

   /*-----------------------------------------------------------------------
    * Setup default values for parameters
    *-----------------------------------------------------------------------*/
   fsai_data = hypre_CTAlloc(hypre_ParFSAIData, 1, HYPRE_MEMORY_HOST);

   /* setup params */
   local_solve_type = 0;
   max_steps = 3;
   max_step_size = 5;
   max_nnz_row = max_steps * max_step_size;
   num_levels = 2;
   kap_tolerance = 1.0e-3;

   /* parameters that depend on the execution policy */
#if defined (HYPRE_USING_CUDA) || defined (HYPRE_USING_HIP)
   HYPRE_MemoryLocation memory_location = hypre_HandleMemoryLocation(hypre_handle());

   if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
   {
      algo_type = 3;
   }
   else
#endif
   {
      algo_type = hypre_NumThreads() > 4 ? 2 : 1;
   }

   /* solver params */
   eig_max_iters = 0;
   max_iterations = 20;
   tolerance = 1.0e-6;
   omega = 1.0;

   /* log info */
   logging = 0;
   num_iterations = 0;

   /* output params */
   print_level = 0;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /*-----------------------------------------------------------------------
    * Create the hypre_ParFSAIData structure and return
    *-----------------------------------------------------------------------*/

   hypre_ParFSAIDataGmat(fsai_data)      = NULL;
   hypre_ParFSAIDataGTmat(fsai_data)     = NULL;
   hypre_ParFSAIDataRWork(fsai_data)     = NULL;
   hypre_ParFSAIDataZWork(fsai_data)     = NULL;
   hypre_ParFSAIDataZeroGuess(fsai_data) = 0;

   hypre_FSAISetAlgoType(fsai_data, algo_type);
   hypre_FSAISetLocalSolveType(fsai_data, local_solve_type);
   hypre_FSAISetMaxSteps(fsai_data, max_steps);
   hypre_FSAISetMaxStepSize(fsai_data, max_step_size);
   hypre_FSAISetMaxNnzRow(fsai_data, max_nnz_row);
   hypre_FSAISetNumLevels(fsai_data, num_levels);
   hypre_FSAISetKapTolerance(fsai_data, kap_tolerance);

   hypre_FSAISetMaxIterations(fsai_data, max_iterations);
   hypre_FSAISetEigMaxIters(fsai_data, eig_max_iters);
   hypre_FSAISetTolerance(fsai_data, tolerance);
   hypre_FSAISetOmega(fsai_data, omega);

   hypre_FSAISetLogging(fsai_data, logging);
   hypre_FSAISetNumIterations(fsai_data, num_iterations);

   hypre_FSAISetPrintLevel(fsai_data, print_level);

   HYPRE_ANNOTATE_FUNC_END;

   return (void *) fsai_data;
}

/******************************************************************************
 * HYPRE_FSAIDestroy
 ******************************************************************************/

HYPRE_Int
hypre_FSAIDestroy( void *data )
{
   hypre_ParFSAIData *fsai_data = (hypre_ParFSAIData*)data;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   if (fsai_data)
   {
      if (hypre_ParFSAIDataGmat(fsai_data))
      {
         hypre_ParCSRMatrixDestroy(hypre_ParFSAIDataGmat(fsai_data));
      }

      if (hypre_ParFSAIDataGTmat(fsai_data))
      {
         hypre_ParCSRMatrixDestroy(hypre_ParFSAIDataGTmat(fsai_data));
      }

      hypre_ParVectorDestroy(hypre_ParFSAIDataRWork(fsai_data));
      hypre_ParVectorDestroy(hypre_ParFSAIDataZWork(fsai_data));

      hypre_TFree(fsai_data, HYPRE_MEMORY_HOST);
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/******************************************************************************
 * Routines to SET the setup phase parameters
 ******************************************************************************/

HYPRE_Int
hypre_FSAISetAlgoType( void      *data,
                       HYPRE_Int  algo_type )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (algo_type < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataAlgoType(fsai_data) = algo_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAISetLocalSolveType( void      *data,
                             HYPRE_Int  local_solve_type )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (local_solve_type < 0 || local_solve_type > 2)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataLocalSolveType(fsai_data) = local_solve_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAISetMaxSteps( void      *data,
                       HYPRE_Int  max_steps )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (max_steps < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataMaxSteps(fsai_data) = max_steps;

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAISetMaxStepSize( void      *data,
                          HYPRE_Int  max_step_size )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (max_step_size < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataMaxStepSize(fsai_data) = max_step_size;

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAISetMaxNnzRow( void      *data,
                        HYPRE_Int  max_nnz_row )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (max_nnz_row < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataMaxNnzRow(fsai_data) = max_nnz_row;

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAISetNumLevels( void      *data,
                        HYPRE_Int  num_levels )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (num_levels < 1)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataNumLevels(fsai_data) = num_levels;

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAISetThreshold( void       *data,
                        HYPRE_Real  threshold )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (threshold < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataThreshold(fsai_data) = threshold;

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAISetKapTolerance( void       *data,
                           HYPRE_Real  kap_tolerance )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (kap_tolerance < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataKapTolerance(fsai_data) = kap_tolerance;

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAISetMaxIterations( void      *data,
                            HYPRE_Int  max_iterations )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (max_iterations < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataMaxIterations(fsai_data) = max_iterations;

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAISetEigMaxIters( void      *data,
                          HYPRE_Int  eig_max_iters )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (eig_max_iters < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataEigMaxIters(fsai_data) = eig_max_iters;

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAISetZeroGuess( void     *data,
                        HYPRE_Int zero_guess )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (zero_guess != 0)
   {
      hypre_ParFSAIDataZeroGuess(fsai_data) = 1;
   }
   else
   {
      hypre_ParFSAIDataZeroGuess(fsai_data) = 0;
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAISetTolerance( void       *data,
                        HYPRE_Real  tolerance )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (tolerance < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataTolerance(fsai_data) = tolerance;

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAISetOmega( void       *data,
                    HYPRE_Real  omega )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (omega < 0)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Negative omega not allowed!");
      return hypre_error_flag;
   }

   hypre_ParFSAIDataOmega(fsai_data) = omega;

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAISetLogging( void      *data,
                      HYPRE_Int  logging )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (logging < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataLogging(fsai_data) = logging;

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAISetNumIterations( void      *data,
                            HYPRE_Int  num_iterations )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (num_iterations < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataNumIterations(fsai_data) = num_iterations;

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAISetPrintLevel( void      *data,
                         HYPRE_Int  print_level )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (print_level < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataPrintLevel(fsai_data) = print_level;

   return hypre_error_flag;
}

/******************************************************************************
 * Routines to GET the setup phase parameters
 ******************************************************************************/

HYPRE_Int
hypre_FSAIGetAlgoType( void      *data,
                       HYPRE_Int *algo_type )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *algo_type = hypre_ParFSAIDataAlgoType(fsai_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAIGetLocalSolveType( void      *data,
                             HYPRE_Int *local_solve_type )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *local_solve_type = hypre_ParFSAIDataLocalSolveType(fsai_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAIGetMaxSteps( void      *data,
                       HYPRE_Int *algo_type )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *algo_type = hypre_ParFSAIDataMaxSteps(fsai_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAIGetMaxStepSize( void      *data,
                          HYPRE_Int *max_step_size )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *max_step_size = hypre_ParFSAIDataMaxStepSize(fsai_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAIGetMaxNnzRow( void      *data,
                        HYPRE_Int *max_nnz_row )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *max_nnz_row = hypre_ParFSAIDataMaxNnzRow(fsai_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAIGetNumLevels( void      *data,
                        HYPRE_Int *num_levels )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *num_levels = hypre_ParFSAIDataNumLevels(fsai_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAIGetThreshold( void       *data,
                        HYPRE_Real *threshold )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *threshold = hypre_ParFSAIDataThreshold(fsai_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAIGetKapTolerance( void       *data,
                           HYPRE_Real *kap_tolerance )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *kap_tolerance = hypre_ParFSAIDataKapTolerance(fsai_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAIGetMaxIterations( void      *data,
                            HYPRE_Int *max_iterations )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *max_iterations = hypre_ParFSAIDataMaxIterations(fsai_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAIGetEigMaxIters( void      *data,
                          HYPRE_Int *eig_max_iters )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *eig_max_iters = hypre_ParFSAIDataEigMaxIters(fsai_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAIGetZeroGuess( void      *data,
                        HYPRE_Int *zero_guess )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *zero_guess = hypre_ParFSAIDataZeroGuess(fsai_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAIGetTolerance( void       *data,
                        HYPRE_Real *tolerance )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *tolerance = hypre_ParFSAIDataTolerance(fsai_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAIGetOmega( void       *data,
                    HYPRE_Real *omega )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *omega = hypre_ParFSAIDataOmega(fsai_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAIGetLogging( void      *data,
                      HYPRE_Int *logging )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *logging = hypre_ParFSAIDataLogging(fsai_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAIGetNumIterations( void      *data,
                            HYPRE_Int *num_iterations )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *num_iterations = hypre_ParFSAIDataNumIterations(fsai_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAIGetPrintLevel( void      *data,
                         HYPRE_Int *print_level )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *print_level = hypre_ParFSAIDataPrintLevel(fsai_data);

   return hypre_error_flag;
}
