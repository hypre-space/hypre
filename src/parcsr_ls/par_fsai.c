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
   HYPRE_Real           S_tolerance;

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
   S_tolerance = 1.0e-6;

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
   hypre_ParFSAIDataGmat(fsai_data)             = NULL;
   hypre_ParFSAIDataSPattern(fsai_data);        = NULL;
   hypre_ParFSAIDataKaporinGradient(fsai_data)  = NULL; 
   hypre_ParFSAIDataNnzPerRow(fsai_data)        = NULL;
   hypre_ParFSAIDataNnzCumSum(fsai_data)        = NULL;


   hypre_ParFSAIDataSetTolerance(fsai_data, tolerance);
   hypre_ParFSAIDataSetMaxSteps(fsai_data, max_steps);
   hypre_ParFSAIDataSetMaxStepSize(fsai_data, max_step_size);

   hypre_ParFSAIDataSetMinIterations(fsai_data, min_iterations);
   hypre_ParFSAIDataSetMaxIterations(fsai_data, max_iterations);
   hypre_ParFSAIDataSetSTolerance(fsai_data, S_tolerance);

   hypre_ParFSAIDataSetLogging(fsai_data, logging);
   hypre_ParFSAIDataSetNumIterations(fsai_data, num_iterations);

   hypre_ParFSAIDataSetPrintLevel(fsai_data, print_level);
   hypre_ParFSAIDataSetLogFileName(fsai_data, log_file_name);
   hypre_ParFSAIDataSetDebugFlag(fsai_data, debug_flag);

}

/******************************************************************************
 * HYPRE_FSAIDestroy
 ******************************************************************************/

HYPRE_Int
HYPRE_FSAIDestroy( HYPRE_Solver solver )
{
   return( hypre_FSAIDestroy( (void*) solver ) );
}

/******************************************************************************
 * HYPRE_FSAISetup
 ******************************************************************************/


