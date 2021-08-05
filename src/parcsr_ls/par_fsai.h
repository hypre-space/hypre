/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_ParFSAI_DATA_HEADER
#define hypre_ParFSAI_DATA_HEADER

/*--------------------------------------------------------------------------
 * hypre_ParFSAIData
 *--------------------------------------------------------------------------*/
typedef struct
{

   HYPRE_MemoryLocation memory_location;     /* memory location of matrices/vectors in FSAIData */
   MPI_Comm             new_comm;
   HYPRE_Int            *comm_info;

   /* FSAI Setup data */
   HYPRE_Int            max_steps;           /* Maximum iterations run per row */
   HYPRE_Int            max_step_size;       /* Maximum number of nonzero elements added to a row of G per step */
   HYPRE_Real           kap_tolerance;       /* Minimum amount of change between two steps */
   hypre_ParCSRMatrix  *Gmat;                /* Matrix holding FSAI factor. M^(-1) = G'G */
   hypre_ParCSRMatrix  *GTmat;               /* Matrix holding the transpose of the FSAI factor */

   /* FSAI Setup info */
   HYPRE_Real           density;             /* Density of matrix G wrt A */

   /* Solver Problem Data */
   HYPRE_Int            max_iterations;      /* Maximum iterations run for the solver */
   HYPRE_Int            num_iterations;      /* Number of iterations the solver ran */
   HYPRE_Real           omega;               /* Step size for Preconditioned Richardson Solver */
   HYPRE_Real           tolerance;    	      /* Tolerance for the solver */
   HYPRE_Real           rel_resnorm;         /* available if logging > 1 */
   hypre_ParVector      *residual;           /* available if logging > 1 */
   hypre_ParVector      *r_work;
   hypre_ParVector      *x_work;
   hypre_ParVector      *z_work;

   /* log info */
   HYPRE_Int            logging;

   /* output params */
   char                 log_file_name[256];
   HYPRE_Int            print_level;
   HYPRE_Int            debug_flag;

} hypre_ParFSAIData;

/*--------------------------------------------------------------------------
 *  Accessor functions for the hypre_ParFSAIData structure
 *--------------------------------------------------------------------------*/

#define hypre_ParFSAIDataMemoryLocation(fsai_data)          ((fsai_data) -> memory_location)
#define hypre_ParFSAIDataNewComm(fsai_data)                 ((fsai_data) -> new_comm)
#define hypre_ParFSAIDataCommInfo(fsai_data)                ((fsai_data) -> comm_info)

/* FSAI Setup data */
#define hypre_ParFSAIDataMaxSteps(fsai_data)                ((fsai_data) -> max_steps)
#define hypre_ParFSAIDataMaxStepSize(fsai_data)             ((fsai_data) -> max_step_size)
#define hypre_ParFSAIDataKapTolerance(fsai_data)            ((fsai_data) -> kap_tolerance)
#define hypre_ParFSAIDataGmat(fsai_data)                    ((fsai_data) -> Gmat)
#define hypre_ParFSAIDataGTmat(fsai_data)                   ((fsai_data) -> GTmat)
#define hypre_ParFSAIDataDensity(fsai_data)                 ((fsai_data) -> density)

/* Solver problem data */
#define hypre_ParFSAIDataMaxIterations(fsai_data)           ((fsai_data) -> max_iterations)
#define hypre_ParFSAIDataNumIterations(fsai_data)           ((fsai_data) -> num_iterations)
#define hypre_ParFSAIDataOmega(fsai_data)                   ((fsai_data) -> omega)
#define hypre_ParFSAIDataRelResNorm(fsai_data)              ((fsai_data) -> rel_resnorm)
#define hypre_ParFSAIDataTolerance(fsai_data)               ((fsai_data) -> tolerance)
#define hypre_ParFSAIDataResidual(fsai_data)                ((fsai_data) -> residual)
#define hypre_ParFSAIDataRWork(fsai_data)                   ((fsai_data) -> r_work)
#define hypre_ParFSAIDataXWork(fsai_data)                   ((fsai_data) -> x_work)
#define hypre_ParFSAIDataZWork(fsai_data)                   ((fsai_data) -> z_work)

/* log info data */
#define hypre_ParFSAIDataLogging(fsai_data)                 ((fsai_data) -> logging)

/* output parameters */
#define hypre_ParFSAIDataLogFileName(fsai_data)             ((fsai_data) -> log_file_name)
#define hypre_ParFSAIDataPrintLevel(fsai_data)              ((fsai_data) -> print_level)
#define hypre_ParFSAIDataDebugFlag(fsai_data)               ((fsai_data) -> debug_flag)

#endif
