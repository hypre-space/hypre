/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_ParFSAI_DATA_HEADER
#define hypre_ParFSAI_DATA_HEADER

//#define DEBUG_FSAI

/*--------------------------------------------------------------------------
 * hypre_ParFSAIData
 *--------------------------------------------------------------------------*/

typedef struct hypre_ParFSAIData_struct
{
   /* FSAI Setup input data */
   HYPRE_Int             algo_type;        /* FSAI algorithm type */
   HYPRE_Int             local_solve_type; /* Local linear solver type */
   HYPRE_Int             max_steps;        /* Max. iterations run per row */
   HYPRE_Int             max_step_size;    /* Max. number of nonzeros added to a row of G per step */
   HYPRE_Int             max_nnz_row;      /* Max. number of nonzeros per row of G */
   HYPRE_Int             num_levels;       /* Number of levels for computing the candidate pattern */
   HYPRE_Real            threshold;        /* Filtering threshold for the candidate pattern */
   HYPRE_Real            kap_tolerance;    /* Min. amount of change between two steps */

   /* FSAI Setup data */
   HYPRE_Real            density;          /* Density of matrix G wrt. A */
   hypre_ParCSRMatrix   *Gmat;             /* Matrix holding FSAI factor. M^(-1) = G'G */
   hypre_ParCSRMatrix   *GTmat;            /* Matrix holding the transpose of the FSAI factor */

   /* Solver data */
   HYPRE_Int             zero_guess;       /* Flag indicating x0 = 0 */
   HYPRE_Int             eig_max_iters;    /* Iters for computing max. eigenvalue of G^T*G*A */
   HYPRE_Int             max_iterations;   /* Maximum iterations run for the solver */
   HYPRE_Int             num_iterations;   /* Number of iterations the solver ran */
   HYPRE_Real            omega;            /* Step size for Preconditioned Richardson Solver */
   HYPRE_Real            tolerance;        /* Tolerance for the solver */
   HYPRE_Real            rel_resnorm;      /* available if logging > 1 */

   /* Work data */
   hypre_ParVector      *r_work;           /* work vector used to compute the residual */
   hypre_ParVector      *z_work;           /* work vector used for applying FSAI */

   /* Log info data */
   HYPRE_Int             logging;
   HYPRE_Int             print_level;
} hypre_ParFSAIData;

/*--------------------------------------------------------------------------
 *  Accessor functions for the hypre_ParFSAIData structure
 *--------------------------------------------------------------------------*/

/* FSAI Setup input data */
#define hypre_ParFSAIDataAlgoType(fsai_data)                ((fsai_data) -> algo_type)
#define hypre_ParFSAIDataLocalSolveType(fsai_data)          ((fsai_data) -> local_solve_type)
#define hypre_ParFSAIDataMaxSteps(fsai_data)                ((fsai_data) -> max_steps)
#define hypre_ParFSAIDataMaxStepSize(fsai_data)             ((fsai_data) -> max_step_size)
#define hypre_ParFSAIDataMaxNnzRow(fsai_data)               ((fsai_data) -> max_nnz_row)
#define hypre_ParFSAIDataNumLevels(fsai_data)               ((fsai_data) -> num_levels)
#define hypre_ParFSAIDataThreshold(fsai_data)               ((fsai_data) -> threshold)
#define hypre_ParFSAIDataKapTolerance(fsai_data)            ((fsai_data) -> kap_tolerance)

/* FSAI Setup data */
#define hypre_ParFSAIDataGmat(fsai_data)                    ((fsai_data) -> Gmat)
#define hypre_ParFSAIDataGTmat(fsai_data)                   ((fsai_data) -> GTmat)
#define hypre_ParFSAIDataDensity(fsai_data)                 ((fsai_data) -> density)

/* Solver problem data */
#define hypre_ParFSAIDataZeroGuess(fsai_data)               ((fsai_data) -> zero_guess)
#define hypre_ParFSAIDataEigMaxIters(fsai_data)             ((fsai_data) -> eig_max_iters)
#define hypre_ParFSAIDataMaxIterations(fsai_data)           ((fsai_data) -> max_iterations)
#define hypre_ParFSAIDataNumIterations(fsai_data)           ((fsai_data) -> num_iterations)
#define hypre_ParFSAIDataOmega(fsai_data)                   ((fsai_data) -> omega)
#define hypre_ParFSAIDataRelResNorm(fsai_data)              ((fsai_data) -> rel_resnorm)
#define hypre_ParFSAIDataTolerance(fsai_data)               ((fsai_data) -> tolerance)

/* Work data */
#define hypre_ParFSAIDataRWork(fsai_data)                   ((fsai_data) -> r_work)
#define hypre_ParFSAIDataZWork(fsai_data)                   ((fsai_data) -> z_work)

/* Log info data */
#define hypre_ParFSAIDataLogging(fsai_data)                 ((fsai_data) -> logging)
#define hypre_ParFSAIDataPrintLevel(fsai_data)              ((fsai_data) -> print_level)

#endif
