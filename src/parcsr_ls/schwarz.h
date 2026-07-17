/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_Schwarz_DATA_HEADER
#define hypre_Schwarz_DATA_HEADER

/*--------------------------------------------------------------------------
 * Schwarz Variant Definitions
 *
 * Domain-based Schwarz variants (0-4): Schwarz with LU factorization
 *   0 = Multiplicative Schwarz (MPSchwarz)
 *   1 = Additive Schwarz with scaling (AdSchwarz)
 *   2 = Parallel Additive Schwarz (ParAdSchwarz)
 *   3 = Parallel Multiplicative Schwarz with boundary (ParMPSchwarz)
 *   4 = Forward Multiplicative Schwarz (MPSchwarzFW)
 *
 * Overlapping Schwarz variants (10+): True overlapping Schwarz
 *   10 = RAS + ILU(k)
 *   11 = AS + ILU(k)
 *   20 = RAS + ILUT
 *   21 = AS + ILUT
 *   30 = RAS + AMG
 *   31 = AS + AMG
 *   40 = RAS + SuperLU
 *   41 = AS + SuperLU
 *--------------------------------------------------------------------------*/

/* Domain-based Schwarz variants */
#define HYPRE_SCHWARZ_VARIANT_MP          0   /* Multiplicative Schwarz */
#define HYPRE_SCHWARZ_VARIANT_AD          1   /* Additive Schwarz with scaling */
#define HYPRE_SCHWARZ_VARIANT_PAR_AD      2   /* Parallel Additive Schwarz */
#define HYPRE_SCHWARZ_VARIANT_PAR_MP      3   /* Parallel Multiplicative Schwarz */
#define HYPRE_SCHWARZ_VARIANT_MP_FW       4   /* Forward Multiplicative Schwarz */

/* Overlapping Schwarz variants */
#define HYPRE_SCHWARZ_VARIANT_OVERLAP_BASE   10

#define HYPRE_SCHWARZ_VARIANT_RAS_ILUK    10  /* RAS + ILU(k) */
#define HYPRE_SCHWARZ_VARIANT_AS_ILUK     11  /* AS  + ILU(k) */
#define HYPRE_SCHWARZ_VARIANT_RAS_ILUT    20  /* RAS + ILUT */
#define HYPRE_SCHWARZ_VARIANT_AS_ILUT     21  /* AS  + ILUT */
#define HYPRE_SCHWARZ_VARIANT_RAS_AMG     30  /* RAS + AMG */
#define HYPRE_SCHWARZ_VARIANT_AS_AMG      31  /* AS  + AMG */
#define HYPRE_SCHWARZ_VARIANT_RAS_SUPERLU 40  /* RAS + SuperLU */
#define HYPRE_SCHWARZ_VARIANT_AS_SUPERLU  41  /* AS  + SuperLU */

/* Internal sub-variant codes for overlapping Schwarz (0=RAS, 1=AS) */
#define HYPRE_SCHWARZ_OVERLAP_VARIANT_RAS     0
#define HYPRE_SCHWARZ_OVERLAP_VARIANT_AS      1

/* Local solver types for overlapping Schwarz */
#define HYPRE_SCHWARZ_LOCAL_SOLVER_ILUK       0
#define HYPRE_SCHWARZ_LOCAL_SOLVER_ILUT       1
#define HYPRE_SCHWARZ_LOCAL_SOLVER_AMG        2
#define HYPRE_SCHWARZ_LOCAL_SOLVER_SUPERLU    3

/* Check if variant uses overlapping Schwarz implementation */
#define HYPRE_SCHWARZ_IS_OVERLAPPING(variant) ((variant) >= HYPRE_SCHWARZ_VARIANT_OVERLAP_BASE)

/*--------------------------------------------------------------------------
 * hypre_SchwarzData
 *
 * Unified data structure for both domain-based and overlapping Schwarz.
 * Uses the variant field to determine which implementation to use.
 *--------------------------------------------------------------------------*/

typedef struct
{
   /* Common parameters */
   HYPRE_Int      variant;
   HYPRE_Int      domain_type;
   HYPRE_Int      overlap;           /* minimal overlap (domain-based); delta (overlap) */
   HYPRE_Int      num_functions;
   HYPRE_Int      use_nonsymm;
   HYPRE_Real     relax_weight;

   /* Domain-based Schwarz-specific data */
   hypre_CSRMatrix *domain_structure;
   hypre_CSRMatrix *A_boundary;
   hypre_ParVector *Vtemp;            /* Temporary global vector (used by both variants) */
   HYPRE_Real     *scale;
   HYPRE_Int      *dof_func;
   HYPRE_Int      *pivots;

   /* Overlapping Schwarz-specific parameters */
   HYPRE_Int       local_solver_type;   /* Subdomain solver type */
   HYPRE_Int       iluk_level_of_fill;  /* Level of fill for ILU(k) */
   HYPRE_Int       ilut_max_nnz_row;    /* Max nonzeros per row for ILUT */
   HYPRE_Real      ilut_droptol;        /* Drop tolerance for ILUT */
   HYPRE_Int       max_iter;            /* Max iterations (for iterative use) */
   HYPRE_Real      tol;                 /* Tolerance */
   HYPRE_Int       print_level;
   HYPRE_Int       logging;

   /* Overlapping Schwarz setup-time data */
   hypre_ParCSRMatrix *A;               /* Original matrix reference */
   hypre_OverlapData  *overlap_data;    /* Overlap computation data */
   HYPRE_Int           num_cols_local;  /* Number of local columns */
   HYPRE_Int          *row_to_col_map;  /* Map: extended row -> local col for u */

   /* Local solver */
   HYPRE_Solver    local_solver;
   HYPRE_PtrToSolverFcn   local_solver_setup;
   HYPRE_PtrToSolverFcn   local_solver_solve;
   HYPRE_PtrToDestroyFcn  local_solver_destroy;
   HYPRE_Int       local_solver_owner;  /* 1 if we own the solver */

   /* For local solver - wrapped ParCSR matrix */
   hypre_ParCSRMatrix *A_local_parcsr;

   /* Overlapping Schwarz work vectors */
   hypre_ParVector *f_local_par;        /* Local RHS (ParVector wrapper) */
   hypre_ParVector *u_local_par;        /* Local solution (ParVector wrapper) */

   /* Statistics */
   HYPRE_Int       num_iterations;
   HYPRE_Real      final_res_norm;
   HYPRE_Real     *res_norms;
   HYPRE_Int       cfsolve_warning_issued;

} hypre_SchwarzData;

/*--------------------------------------------------------------------------
 * Accessor macros for the hypre_SchwarzData structure
 *--------------------------------------------------------------------------*/

#define hypre_SchwarzDataVariant(schwarz_data)       ((schwarz_data)->variant)
#define hypre_SchwarzDataDomainType(schwarz_data)      ((schwarz_data)->domain_type)
#define hypre_SchwarzDataOverlap(schwarz_data)       ((schwarz_data)->overlap)
#define hypre_SchwarzDataNumFunctions(schwarz_data)    ((schwarz_data)->num_functions)
#define hypre_SchwarzDataUseNonSymm(schwarz_data)      ((schwarz_data)->use_nonsymm)
#define hypre_SchwarzDataRelaxWeight(schwarz_data)   ((schwarz_data)->relax_weight)

/* Domain-based Schwarz accessors */
#define hypre_SchwarzDataDomainStructure(schwarz_data) ((schwarz_data)->domain_structure)
#define hypre_SchwarzDataABoundary(schwarz_data)       ((schwarz_data)->A_boundary)
#define hypre_SchwarzDataVtemp(schwarz_data)           ((schwarz_data)->Vtemp)
#define hypre_SchwarzDataScale(schwarz_data)           ((schwarz_data)->scale)
#define hypre_SchwarzDataDofFunc(schwarz_data)         ((schwarz_data)->dof_func)
#define hypre_SchwarzDataPivots(schwarz_data)          ((schwarz_data)->pivots)

/* Overlapping Schwarz parameter accessors */
#define hypre_SchwarzDataLocalSolverType(schwarz_data)   ((schwarz_data)->local_solver_type)
#define hypre_SchwarzDataILUKLevelOfFill(schwarz_data)   ((schwarz_data)->iluk_level_of_fill)
#define hypre_SchwarzDataILUTMaxNnzRow(schwarz_data)     ((schwarz_data)->ilut_max_nnz_row)
#define hypre_SchwarzDataILUTDroptol(schwarz_data)       ((schwarz_data)->ilut_droptol)
#define hypre_SchwarzDataMaxIter(schwarz_data)           ((schwarz_data)->max_iter)
#define hypre_SchwarzDataTol(schwarz_data)               ((schwarz_data)->tol)
#define hypre_SchwarzDataPrintLevel(schwarz_data)        ((schwarz_data)->print_level)
#define hypre_SchwarzDataLogging(schwarz_data)           ((schwarz_data)->logging)

/* Overlapping Schwarz setup-time data accessors */
#define hypre_SchwarzDataA(schwarz_data)                 ((schwarz_data)->A)
#define hypre_SchwarzDataOverlapData(schwarz_data)       ((schwarz_data)->overlap_data)
#define hypre_SchwarzDataNumColsLocal(schwarz_data)      ((schwarz_data)->num_cols_local)
#define hypre_SchwarzDataRowToColMap(schwarz_data)       ((schwarz_data)->row_to_col_map)
#define hypre_SchwarzDataLocalSolver(schwarz_data)       ((schwarz_data)->local_solver)
#define hypre_SchwarzDataLocalSolverSetup(schwarz_data)   ((schwarz_data)->local_solver_setup)
#define hypre_SchwarzDataLocalSolverSolve(schwarz_data)   ((schwarz_data)->local_solver_solve)
#define hypre_SchwarzDataLocalSolverDestroy(schwarz_data) ((schwarz_data)->local_solver_destroy)
#define hypre_SchwarzDataLocalSolverOwner(schwarz_data)   ((schwarz_data)->local_solver_owner)
#define hypre_SchwarzDataALocalParCSR(schwarz_data)     ((schwarz_data)->A_local_parcsr)
#define hypre_SchwarzDataFLocalPar(schwarz_data)         ((schwarz_data)->f_local_par)
#define hypre_SchwarzDataULocalPar(schwarz_data)         ((schwarz_data)->u_local_par)
#define hypre_SchwarzDataNumIterations(schwarz_data)     ((schwarz_data)->num_iterations)
#define hypre_SchwarzDataFinalResNorm(schwarz_data)      ((schwarz_data)->final_res_norm)
#define hypre_SchwarzDataResNorms(schwarz_data)          ((schwarz_data)->res_norms)
#define hypre_SchwarzDataCFSolveWarningIssued(schwarz_data) ((schwarz_data)->cfsolve_warning_issued)

#endif
