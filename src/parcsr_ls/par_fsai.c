/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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
hypre_FSAICreate()
{

   hypre_ParFSAIData    *fsai_data;

   /* setup params */
   HYPRE_Int            max_steps;
   HYPRE_Int            max_step_size;
   HYPRE_Real           tolerance;

   /* solver params */
   HYPRE_Int            min_iterations;
   HYPRE_Int            max_iterations;
   HYPRE_Real           solver_tolerance;

   /* log info */
   HYPRE_Int            logging;
   HYPRE_Int            num_iterations;

   /* output params */
   HYPRE_Int            print_level;
   HYPRE_Int            debug_flag;
   char                 log_file_name[256];

   /*-----------------------------------------------------------------------
    * Setup default values for parameters
    *-----------------------------------------------------------------------*/ 

   /* setup params */
   max_steps = 10;
   max_step_size = 3;
   tolerance = 1.0e-3;

   /* solver params */
   min_iterations = 0;
   max_iterations = 20;
   solver_tolerance = 1.0e-6;

   /* log info */
   logging = 0;
   num_iterations = 0;

   /* output params */  
   print_level = 0;
   debug_flag = 0;
   hypre_sprintf(log_file_name, "%s", "fsai.out.log");
 
   #if defined(HYPRE_USING_GPU)
   #endif

   HYPRE_ANNOTATE_FUNC_BEGIN; 

   /*-----------------------------------------------------------------------
    * Create the hypre_ParFSAIData structure and return
    *-----------------------------------------------------------------------*/ 

   fsai_data = hypre_CTAlloc(hypre_ParFSAIData, 1, HYPRE_MEMORY_HOST);

   hypre_ParFSAIDataMemoryLocation(fsai_data)   = HYPRE_MEMORY_UNDEFINED;
 
   hypre_ParFSAIDataAmat(fsai_data)             = NULL;
   hypre_ParFSAIDataAinv(fsai_data)             = NULL;
   hypre_ParFSAIDataSPattern(fsai_data)         = NULL;
   hypre_ParFSAIDataAArray(fsai_data)           = NULL;
   hypre_ParFSAIDataGArray(fsai_data)           = NULL;
   hypre_ParFSAIDataPArray(fsai_data)           = NULL;
   hypre_ParFSAIDatabvec(fsai_data)             = NULL;
   hypre_ParFSAIDataGmat(fsai_data)             = NULL;
   hypre_ParFSAIDataKaporinGradient(fsai_data)  = NULL; 
   hypre_ParFSAIDataNnzPerRow(fsai_data)        = NULL;
   hypre_ParFSAIDataNnzCumSum(fsai_data)        = NULL;
   hypre_ParFSAIDataCommInfo(fsai_data)         = NULL;
   hypre_ParFSAIDataNewComm(fsai_data)          = hypre_MPI_COMM_NULL;

   hypre_FSAISetTolerance(fsai_data, tolerance);
   hypre_FSAISetMaxSteps(fsai_data, max_steps);
   hypre_FSAISetMaxStepSize(fsai_data, max_step_size);

   hypre_FSAISetMinIterations(fsai_data, min_iterations);
   hypre_FSAISetMaxIterations(fsai_data, max_iterations);
   hypre_FSAISetSolverTolerance(fsai_data, solver_tolerance);

   hypre_FSAISetLogging(fsai_data, logging);
   hypre_FSAISetNumIterations(fsai_data, num_iterations);

   hypre_FSAISetPrintLevel(fsai_data, print_level);
   hypre_FSAISetPrintFileName(fsai_data, log_file_name);
   hypre_FSAISetDebugFlag(fsai_data, debug_flag);

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
   MPI_comm new_comm = hypre_ParFSAIDataNewComm(fsai_data);

   HYPRE_ANNOTATE_FUNC_BEGIN;

   if (hypre_ParFSAIDataCommInfo(fsai_data)) hypre_TFree(hypre_ParFSAIDataCommInfo(fsai_data), HYPRE_MEMORY_HOST);
   if (hypre_ParFSAIDataAmat(fsai_data)) hypre_TFree(hypre_ParFSAIDataAmat(fsai_data), HYPRE_MEMORY_HOST);
   if (hypre_ParFSAIDataMinv(fsai_data)) hypre_TFree(hypre_ParFSAIDataMinv(fsai_data), HYPRE_MEMORY_HOST);
   if (hypre_ParFSAIDataSPattern(fsai_data)) hypre_TFree(hypre_ParFSAIDataSPattern(fsai_data), HYPRE_MEMORY_HOST);
   if (hypre_ParFSAIDatabvec(fsai_data)) hypre_TFree(hypre_ParFSAIDatabvec(fsai_data), HYPRE_MEMORY_HOST);
   if (hypre_ParFSAIDataGmat(fsai_data)) hypre_TFree(hypre_ParFSAIDataAinv(fsai_data), HYPRE_MEMORY_HOST);
   if (hypre_ParFSAIDataCommInfo(fsai_data)) hypre_TFree(hypre_ParFSAIDataCommInfo(fsai_data), HYPRE_MEMORY_HOST);
   if (hypre_ParFSAIDataAArray(fsai_data)) hypre_TFree(hypre_ParFSAIDataAArray(fsai_data), HYPRE_MEMORY_HOST);
   if (hypre_ParFSAIDataGArray(fsai_data)) hypre_TFree(hypre_ParFSAIDataGArray(fsai_data), HYPRE_MEMORY_HOST);
   if (hypre_ParFSAIDataPArray(fsai_data)) hypre_TFree(hypre_ParFSAIDataPArray(fsai_data), HYPRE_MEMORY_HOST);

   if( hypre_ParFSAIDataResidual(fsai_data) )
   {
      hypre_ParVectorDestroy( hypre_ParFSAIDataResidual(fsai_data) );
      hypre_ParFSAIDataResidual(fsai_data) = NULL; 
   }
   if( hypre_ParFSAIDataKaporinGradient(fsai_data) )
   {
      hypre_ParVectorDestroy( hypre_ParFSAIDataKaporinGradient(fsai_data) );
      hypre_ParFSAIDataKaporinGradient(fsai_data) = NULL; 
   }
   if( hypre_ParFSAIDataNnzPerRow(fsai_data) )
   {
      hypre_ParVectorDestroy( hypre_ParFSAIDataNnzPerRow(fsai_data) );
      hypre_ParFSAIDataNnzPerRow(fsai_data) = NULL; 
   }
   if( hypre_ParFSAIDataNnzCumSum(fsai_data) )
   {
      hypre_ParVectorDestroy( hypre_ParFSAIDataNnzCumSum(fsai_data) );
      hypre_ParFSAIDataNnzCumSum(fsai_data) = NULL; 
   }

   if( new_comm != hypre_MPI_COMM_NULL )
      hypre_MPI_Comm_free(&new_comm);

   hypre_TFree(fsai_data, HYPRE_MEMORY_HOST);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag; 
}

/******************************************************************************
 * Routines to SET the setup phase parameters
 ******************************************************************************/

HYPRE_Int
hypre_FSAISetTolerance( void *data,
                          HYPRE_Real tolerance   )
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
hypre_FSAISetMaxSteps( void *data,
                          HYPRE_Int max_steps   )
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
hypre_FSAISetMaxStepSize( void *data,
                          HYPRE_Int max_step_size   )
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
hypre_FSAISetMinIterations( void *data,
                          HYPRE_Int min_iterations   )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (min_iterations < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataMinIterations(fsai_data) = min_iterations;

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAISetMaxIterations( void *data,
                          HYPRE_Int max_iterations   )
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

   hypre_ParFSAIDataMaxIterations(fsai_data) = min_iterations;

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAISetSolverTolerance( void *data,
                          HYPRE_Real solver_tolerance   )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (solver_tolerance < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataSTolerance(fsai_data) = solver_tolerance;

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAISetLogging( void *data,
                          HYPRE_Int logging   )
{
/*   This function should be called before Setup.  Logging changes
 *    may require allocation or freeing of arrays, which is presently
 *    only done there.
 *    It may be possible to support logging changes at other times,
 *    but there is little need.
 */
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
hypre_FSAISetNumIterations( void *data,
                          HYPRE_Int num_iterations   )
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
hypre_FSAISetPrintLevel( void *data,
                          HYPRE_Int print_level   )
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

HYPRE_Int
hypre_FSAISetPrintFileName( void       *data,
                               const char *print_file_name )
{
   hypre_ParFSAIData  *fsai_data =  (hypre_ParFSAIData*) data;
   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   if ( strlen(print_file_name) > 256 )
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_sprintf(hypre_ParFSAIDataLogFileName(fsai_data), "%s", print_file_name;

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAISetDebugFlag( void *data,
                          HYPRE_Int debug_flag   )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (debug_flag < 0)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   hypre_ParFSAIDataDebugFlag(fsai_data) = debug_flag;

   return hypre_error_flag;
}

/******************************************************************************
 * Routines to GET the setup phase parameters
 ******************************************************************************/

HYPRE_Int
hypre_FSAIGetTolerance( void     *data,
                          HYPRE_Int     *tolerance )
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
hypre_FSAIGetMaxSteps( void     *data,
                          HYPRE_Int     *max_steps )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *max_steps = hypre_ParFSAIDataMaxSteps(fsai_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAIGetMaxStepSize( void     *data,
                          HYPRE_Int     *max_step_size )
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
hypre_FSAIGetMinIterations( void     *data,
                          HYPRE_Int     *min_iterations )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *min_iterations = hypre_ParFSAIDataMinIterations(fsai_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAIGetMaxIterations( void     *data,
                          HYPRE_Int     *max_iterations )
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
hypre_FSAIGetSolverTolerance( void     *data,
                          HYPRE_Real     *solver_tolerance )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *solver_tolerance = hypre_ParFSAIDataSolverTolerance(fsai_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAIGetLogging( void     *data,
                          HYPRE_Int     *logging )
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
hypre_FSAIGetNumIterations( void     *data,
                          HYPRE_Int     *num_iterations )
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
hypre_FSAIGetPrintLevel( void     *data,
                          HYPRE_Int     *print_level )
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

HYPRE_Int
hypre_FSAIGetPrintFileName( void       *data,
                                 char ** print_file_name )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   hypre_sprintf( *print_file_name, "%s", hypre_ParFSAIDataLogFileName(fsai_data) );

   return hypre_error_flag;
}

HYPRE_Int
hypre_FSAIGetDebugFlag( void     *data,
                          HYPRE_Int     *debug_flag )
{
   hypre_ParFSAIData  *fsai_data = (hypre_ParFSAIData*) data;

   if (!fsai_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *debug_flag = hypre_ParFSAIDataDebugFlag(fsai_data);

   return hypre_error_flag;
}
